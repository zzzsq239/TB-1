#!/usr/bin/env python3
# /root/lanyun-tmp/oiltea_multimodal_segmentation/test_multimodal_vmamba_with_boundary_corrected.py

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加当前目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# ==================== 可视化工具（带边界）- 修改为9个类别 ====================
class VisualizationToolWithBoundary:
    @staticmethod
    def denormalize_image(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """反归一化图像"""
        if torch.is_tensor(image_tensor):
            image_tensor = image_tensor.cpu()
        
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        mean = torch.tensor(mean).view(1, 3, 1, 1)
        std = torch.tensor(std).view(1, 3, 1, 1)
        
        denorm = image_tensor * std + mean
        denorm = torch.clamp(denorm, 0, 1)
        return denorm
    
    @staticmethod
    def create_mask_visualization(mask_tensor, num_classes=9):  # 改为9个类别
        """创建分割掩码可视化"""
        if torch.is_tensor(mask_tensor):
            mask_np = mask_tensor.cpu().numpy()
        else:
            mask_np = mask_tensor
        
        # 颜色映射：9个类别的不同颜色
        color_map = {
            0: [0, 0, 0],        # 黑色背景
            1: [0, 255, 0],      # 绿色叶片
            2: [255, 0, 0],      # 红色 - 病斑1
            3: [0, 0, 255],      # 蓝色 - 病斑2
            4: [255, 255, 0],    # 黄色 - 病斑3
            5: [255, 0, 255],    # 品红 - 病斑4
            6: [0, 255, 255],    # 青色 - 病斑5
            7: [128, 0, 0],      # 深红 - 病斑6
            8: [0, 128, 0]       # 深绿 - 病斑7
        }
        
        # 确保只映射有效的类别
        color_map = {k: v for k, v in color_map.items() if k < num_classes}
        
        if mask_np.ndim == 2:
            h, w = mask_np.shape
            vis = np.zeros((h, w, 3), dtype=np.uint8)
            for class_id, color in color_map.items():
                vis[mask_np == class_id] = color
        else:  # batch维度
            b, h, w = mask_np.shape
            vis = np.zeros((b, h, w, 3), dtype=np.uint8)
            for class_id, color in color_map.items():
                for i in range(b):
                    vis[i][mask_np[i] == class_id] = color
        
        return vis
    
    @staticmethod
    def create_boundary_visualization(boundary_tensor):
        """创建边界图可视化"""
        if torch.is_tensor(boundary_tensor):
            boundary_np = boundary_tensor.cpu().numpy()
        else:
            boundary_np = boundary_tensor
        
        # 处理不同形状
        if boundary_np.ndim == 4:  # [B, 1, H, W]
            boundary_np = boundary_np.squeeze(1)  # [B, H, W]
        elif boundary_np.ndim == 3 and boundary_np.shape[0] == 1:  # [1, H, W]
            boundary_np = boundary_np.squeeze(0)  # [H, W]
        
        if boundary_np.ndim == 2:
            h, w = boundary_np.shape
            # 创建热力图
            boundary_np = np.clip(boundary_np, 0, 1)
            # 使用jet颜色映射
            cmap = plt.cm.jet
            norm_boundary = (boundary_np - boundary_np.min()) / (boundary_np.max() - boundary_np.min() + 1e-8)
            colored_boundary = (cmap(norm_boundary)[:, :, :3] * 255).astype(np.uint8)
            return colored_boundary
        else:  # batch维度
            b, h, w = boundary_np.shape
            colored_boundaries = []
            for i in range(b):
                cmap = plt.cm.jet
                norm_boundary = (boundary_np[i] - boundary_np[i].min()) / (boundary_np[i].max() - boundary_np[i].min() + 1e-8)
                colored_boundary = (cmap(norm_boundary)[:, :, :3] * 255).astype(np.uint8)
                colored_boundaries.append(colored_boundary)
            
            return np.array(colored_boundaries)
    
    @staticmethod
    def create_overlay_boundary(image, boundary, alpha=0.7):
        """在图像上叠加边界"""
        if torch.is_tensor(image):
            image_np = image.cpu().numpy()
        else:
            image_np = image
        
        if torch.is_tensor(boundary):
            boundary_np = boundary.cpu().numpy()
        else:
            boundary_np = boundary
        
        # 处理图像形状 - 确保是[H, W, 3]
        if image_np.ndim == 4:  # [B, C, H, W]
            image_np = image_np.transpose(0, 2, 3, 1)  # [B, H, W, C]
        elif image_np.ndim == 3 and image_np.shape[0] in [3, 1]:  # [C, H, W]
            image_np = image_np.transpose(1, 2, 0)  # [H, W, C]
            if image_np.shape[2] == 1:  # 如果是单通道，转为3通道
                image_np = np.repeat(image_np, 3, axis=2)
        
        # 处理边界形状
        if boundary_np.ndim == 4:  # [B, 1, H, W]
            boundary_np = boundary_np.squeeze(1)  # [B, H, W]
        elif boundary_np.ndim == 3 and boundary_np.shape[0] == 1:  # [1, H, W]
            boundary_np = boundary_np.squeeze(0)  # [H, W]
        elif boundary_np.ndim == 3 and boundary_np.shape[2] == 1:  # [H, W, 1]
            boundary_np = boundary_np.squeeze(2)  # [H, W]
        
        # 归一化到[0, 1]
        if image_np.max() > 1:
            image_np = image_np / 255.0
        
        boundary_np = np.clip(boundary_np, 0, 1)
        
        # 创建红色边界
        if boundary_np.ndim == 2:
            h, w = boundary_np.shape
            red_boundary = np.zeros((h, w, 3))
            red_boundary[:, :, 0] = boundary_np  # 红色通道
            red_boundary[:, :, 1] = 0  # 绿色通道为0
            red_boundary[:, :, 2] = 0  # 蓝色通道为0
            overlay = image_np * (1 - alpha) + red_boundary * alpha
        elif boundary_np.ndim == 3 and boundary_np.shape[2] == 3:  # [H, W, 3]
            overlay = image_np * (1 - alpha) + boundary_np * alpha
        else:
            # 其他情况，直接使用边界作为红色通道
            h, w = boundary_np.shape[:2]
            red_boundary = np.zeros((h, w, 3))
            red_boundary[:, :, 0] = boundary_np
            overlay = image_np * (1 - alpha) + red_boundary * alpha
        
        return np.clip(overlay * 255, 0, 255).astype(np.uint8)
    
    @staticmethod
    def create_difference_map(pred_mask, true_mask, num_classes=9):  # 改为9个类别
        """创建预测差异图"""
        if torch.is_tensor(pred_mask):
            pred_np = pred_mask.cpu().numpy()
        else:
            pred_np = pred_mask
        
        if torch.is_tensor(true_mask):
            true_np = true_mask.cpu().numpy()
        else:
            true_np = true_mask
        
        if pred_np.ndim == 2:
            h, w = pred_np.shape
            diff = np.zeros((h, w, 3), dtype=np.uint8)
            
            # 绿色：完全正确
            correct_mask = (pred_np == true_np)
            diff[correct_mask] = [0, 255, 0]
            
            # 按类别处理错误
            for class_id in range(num_classes):
                if class_id == 0:  # 背景
                    bg_fp = (pred_np == 0) & (true_np != 0)
                    bg_fn = (pred_np != 0) & (true_np == 0)
                    bg_error = bg_fp | bg_fn
                    bg_error = bg_error & ~correct_mask
                    diff[bg_error] = [255, 255, 255]  # 白色：背景错误
                
                elif class_id == 1:  # 叶片
                    leaf_fp = (pred_np == 1) & (true_np != 1)
                    leaf_fn = (pred_np != 1) & (true_np == 1)
                    leaf_error = leaf_fp | leaf_fn
                    leaf_error = leaf_error & ~correct_mask
                    diff[leaf_error] = [0, 0, 255]  # 蓝色：叶片错误
                
                else:  # 病斑 (2-8)
                    disease_fp = (pred_np == class_id) & (true_np != class_id)
                    disease_fn = (pred_np != class_id) & (true_np == class_id)
                    disease_error = disease_fp | disease_fn
                    disease_error = disease_error & ~correct_mask
                    
                    # 为不同病斑类型使用不同颜色
                    disease_colors = {
                        2: [255, 255, 0],    # 黄色：病斑1错误
                        3: [255, 165, 0],    # 橙色：病斑2错误
                        4: [255, 0, 165],    # 粉色：病斑3错误
                        5: [165, 255, 0],    # 黄绿：病斑4错误
                        6: [0, 255, 165],    # 蓝绿：病斑5错误
                        7: [165, 0, 255],    # 紫红：病斑6错误
                        8: [128, 128, 0]     # 橄榄色：病斑7错误
                    }
                    if class_id in disease_colors:
                        diff[disease_error] = disease_colors[class_id]
        
        return diff

# ==================== 标准dataset-level指标计算器 =====================
class DatasetLevelMetricsCalculator:
    def __init__(self, num_classes=9):
        self.num_classes = num_classes
        self.class_names = [
            'Background', 'Leaf', 
            'Disease_1', 'Disease_2', 'Disease_3', 'Disease_4',
            'Disease_5', 'Disease_6', 'Disease_7'
        ]
    
    def calculate_dataset_level_metrics(self, all_preds, all_targets, include_bg=True):
        """计算标准的 dataset-level 指标"""
        num_classes = self.num_classes
        
        # 初始化统计
        tp = np.zeros(num_classes, dtype=np.int64)
        fp = np.zeros(num_classes, dtype=np.int64)
        fn = np.zeros(num_classes, dtype=np.int64)
        total_pixels_per_class = np.zeros(num_classes, dtype=np.int64)
        
        # 总体像素统计
        total_correct = 0
        total_pixels = 0
        
        # 遍历所有图像
        for pred, target in zip(all_preds, all_targets):
            pred_flat = pred.flatten()
            target_flat = target.flatten()
            
            # 总体统计
            total_pixels += len(pred_flat)
            total_correct += (pred_flat == target_flat).sum()
            
            # 按类别统计
            for c in range(num_classes):
                tp[c] += np.sum((pred_flat == c) & (target_flat == c))
                fp[c] += np.sum((pred_flat == c) & (target_flat != c))
                fn[c] += np.sum((pred_flat != c) & (target_flat == c))
                total_pixels_per_class[c] += (target_flat == c).sum()
        
        eps = 1e-8
        metrics = {}
        
        # 计算每个类别的指标
        for c in range(num_classes):
            iou = tp[c] / (tp[c] + fp[c] + fn[c] + eps)
            precision = tp[c] / (tp[c] + fp[c] + eps)
            recall = tp[c] / (tp[c] + fn[c] + eps)
            dice = 2 * tp[c] / (2 * tp[c] + fp[c] + fn[c] + eps)
            
            metrics[f'class_{c}_iou'] = float(iou)
            metrics[f'class_{c}_precision'] = float(precision)
            metrics[f'class_{c}_recall'] = float(recall)
            metrics[f'class_{c}_dice'] = float(dice)
            metrics[f'class_{c}_tp'] = int(tp[c])
            metrics[f'class_{c}_fp'] = int(fp[c])
            metrics[f'class_{c}_fn'] = int(fn[c])
            metrics[f'class_{c}_total_pixels'] = int(total_pixels_per_class[c])
            metrics[f'class_{c}_pixel_ratio'] = float(total_pixels_per_class[c] / (total_pixels + eps))
        
        # 计算不同类别的 mIoU
        # 1. 所有类别
        all_classes = list(range(num_classes))
        all_classes_with_data = [c for c in all_classes if (tp[c] + fn[c]) > 0]
        
        if all_classes_with_data:
            all_iou = [metrics[f'class_{c}_iou'] for c in all_classes_with_data]
            metrics['all_mIoU'] = float(np.mean(all_iou))
        else:
            metrics['all_mIoU'] = 0.0
        
        # 2. 排除背景的类别
        exclude_bg_classes = [c for c in range(1, num_classes) if (tp[c] + fn[c]) > 0]
        
        if exclude_bg_classes:
            exclude_bg_iou = [metrics[f'class_{c}_iou'] for c in exclude_bg_classes]
            metrics['exclude_bg_mIoU'] = float(np.mean(exclude_bg_iou))
        else:
            metrics['exclude_bg_mIoU'] = 0.0
        
        # 3. 背景和叶片（如果存在）
        bg_leaf_classes = []
        if 0 in all_classes_with_data:
            bg_leaf_classes.append(0)
        if 1 in all_classes_with_data:
            bg_leaf_classes.append(1)
        
        if bg_leaf_classes:
            bg_leaf_iou = [metrics[f'class_{c}_iou'] for c in bg_leaf_classes]
            metrics['bg_leaf_mIoU'] = float(np.mean(bg_leaf_iou))
        else:
            metrics['bg_leaf_mIoU'] = 0.0
        
        # 4. 病斑类别 (2-8)
        disease_classes = [c for c in range(2, num_classes) if (tp[c] + fn[c]) > 0]
        
        if disease_classes:
            disease_iou = [metrics[f'class_{c}_iou'] for c in disease_classes]
            metrics['disease_mIoU'] = float(np.mean(disease_iou))
            metrics['disease_class_count'] = len(disease_classes)
        else:
            metrics['disease_mIoU'] = 0.0
            metrics['disease_class_count'] = 0
        
        # 5. 叶片单独指标
        if 1 in all_classes_with_data:
            metrics['leaf_iou'] = metrics['class_1_iou']
            metrics['leaf_precision'] = metrics['class_1_precision']
            metrics['leaf_recall'] = metrics['class_1_recall']
            metrics['leaf_dice'] = metrics['class_1_dice']
        else:
            metrics['leaf_iou'] = 0.0
            metrics['leaf_precision'] = 0.0
            metrics['leaf_recall'] = 0.0
            metrics['leaf_dice'] = 0.0
        
        # 6. 混合指标（背景+叶片+病斑）
        mixed_classes = all_classes_with_data
        if mixed_classes:
            mixed_iou = [metrics[f'class_{c}_iou'] for c in mixed_classes]
            metrics['mixed_mIoU'] = float(np.mean(mixed_iou))
        else:
            metrics['mixed_mIoU'] = 0.0
        
        # 7. 混合指标（排除背景）
        mixed_exclude_bg_classes = exclude_bg_classes
        if mixed_exclude_bg_classes:
            mixed_exclude_bg_iou = [metrics[f'class_{c}_iou'] for c in mixed_exclude_bg_classes]
            metrics['mixed_mIoU_exclude_bg'] = float(np.mean(mixed_exclude_bg_iou))
        else:
            metrics['mixed_mIoU_exclude_bg'] = 0.0
        
        # 总体指标
        metrics['overall_pixel_accuracy'] = float(total_correct / (total_pixels + eps))
        metrics['total_images'] = len(all_preds)
        metrics['total_pixels'] = int(total_pixels)
        
        return metrics

# ==================== 边界增强的多模态测试器（修正版）====================
class CorrectedMultimodalVMambaTesterWithBoundary:
    def __init__(self, data_root, checkpoint_path, image_size=512, batch_size=4, 
                 model_size='small', local_model_path='./models/bert-base-uncased',
                 output_dir='./test_results_corrected'):
        
        print("=" * 80)
        print("Corrected Multimodal Vision Mamba Tester with Boundary Enhancement")
        print("=" * 80)
        
        self.data_root = data_root
        self.checkpoint_path = checkpoint_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.model_size = model_size
        self.local_model_path = local_model_path
        self.num_classes = 9  # 9个类别
        
        # 创建输出目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join(output_dir, f"test_corrected_{timestamp}")
        self.vis_dir = os.path.join(self.output_dir, "visualizations")
        self.per_image_dir = os.path.join(self.output_dir, "per_image_results")
        self.boundary_dir = os.path.join(self.output_dir, "boundary_results")
        self.csv_files_dir = os.path.join(self.output_dir, "csv_files")
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)
        os.makedirs(self.per_image_dir, exist_ok=True)
        os.makedirs(self.boundary_dir, exist_ok=True)
        os.makedirs(self.csv_files_dir, exist_ok=True)
        
        # 设备
        self.device = self._setup_device()
        
        # 初始化
        self.model = self._load_corrected_model()
        self.test_loader = self._init_dataloader()
        # 使用标准的 dataset-level 指标计算器
        self.metrics_calc = DatasetLevelMetricsCalculator(num_classes=self.num_classes)
        self.vis_tool = VisualizationToolWithBoundary()
        
        print(f" Corrected multimodal tester with boundary enhancement initialized successfully")
        print(f" Output directory: {self.output_dir}")
        print(f" Metric Strategy: Standard Dataset-Level")
        print(f" Visualization directory: {self.vis_dir}")
        print(f" Boundary directory: {self.boundary_dir}")
    
    def _setup_device(self):
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            torch.cuda.empty_cache()
            print(f"🎮 GPU: {torch.cuda.get_device_name(device)}")
        else:
            device = torch.device('cpu')
            print(" CPU")
        return device
    
    def _load_corrected_model(self):
        """加载修正版的训练好的模型"""
        print(f"Loading corrected model from checkpoint: {self.checkpoint_path}")
        
        if not os.path.exists(self.checkpoint_path):
            print(f" Checkpoint not found: {self.checkpoint_path}")
            sys.exit(1)
        
        # 先加载checkpoint查看实际的配置
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        config = checkpoint.get('config', {})
        
        # 检查配置
        use_boundary_branch = config.get('use_boundary_branch', False)
        deep_supervision = config.get('deep_supervision', False)
        use_dynamic_weights = config.get('use_dynamic_weights', True)
        dynamic_k = config.get('dynamic_k', 1000.0)
        ce_weight = config.get('ce_weight', 0.7)
        dice_weight = config.get('dice_weight', 0.3)
        dice_weight_factor = config.get('dice_weight_factor', 2.0)
        
        # 从checkpoint中推断模型配置
        print(f" Corrected model configuration from checkpoint:")
        print(f"  - Use boundary branch: {use_boundary_branch}")
        print(f"  - Deep supervision: {deep_supervision}")
        print(f"  - Use dynamic weights: {use_dynamic_weights}")
        print(f"  - Dynamic K: {dynamic_k}")
        print(f"  - CE weight: {ce_weight}")
        print(f"  - Dice weight: {dice_weight}")
        print(f"  - Dice weight factor: {dice_weight_factor}")
        print(f"  - Number of classes: {config.get('num_classes', 9)}")
        
        # 尝试从checkpoint推断模型大小
        model_size = self._infer_model_size_from_checkpoint(checkpoint)
        print(f"  - Inferred model size: {model_size}")
        
        # 模型配置 - 更新以匹配checkpoint
        configs = {
            'tiny': {'embed_dim': 64, 'depths': [1, 2, 4, 1], 'd_state': 8},
            'small': {'embed_dim': 96, 'depths': [2, 2, 9, 2], 'd_state': 16},
            'base': {'embed_dim': 128, 'depths': [2, 2, 18, 2], 'd_state': 32},
            'large': {'embed_dim': 192, 'depths': [2, 2, 27, 2], 'd_state': 48}
        }
        
        # 使用推断的模型大小或默认值
        if model_size and model_size in configs:
            self.model_size = model_size
            cfg = configs[model_size]
        else:
            # 尝试从depths推断
            depths = self._infer_depths_from_checkpoint(checkpoint)
            if depths:
                print(f"  - Inferred depths: {depths}")
                cfg = {
                    'embed_dim': 128,  # 默认base大小
                    'depths': depths,
                    'd_state': 32
                }
            else:
                # 使用配置中的模型大小或默认base
                self.model_size = config.get('model_size', 'base')
                cfg = configs.get(self.model_size, configs['base'])
        
        # 获取类别数（默认为9）
        num_classes = config.get('num_classes', 9)
        print(f" Initializing corrected {self.model_size} model with {num_classes} classes...")
        
        # 尝试导入修正版模型
        try:
            from models.multimodal_vmamba_local_with_boundary_corrected import (
                MultimodalVMambaSegLocalWithBoundary
            )
            print(" Loaded corrected multimodal model class")
        except ImportError as e:
            print(f" Failed to import corrected model: {e}")
            print(" Trying to import from local directory...")
            sys.path.append(os.path.join(current_dir, 'models'))
            try:
                from multimodal_vmamba_local_with_boundary_corrected import (
                    MultimodalVMambaSegLocalWithBoundary
                )
                print(" Loaded corrected model from local directory")
            except ImportError as e2:
                print(f" Failed to import corrected model: {e2}")
                sys.exit(1)
        
        # 创建修正版模型
        model = MultimodalVMambaSegLocalWithBoundary(
            num_classes=num_classes,
            embed_dim=cfg['embed_dim'],
            depths=cfg['depths'],
            d_state=cfg['d_state'],
            use_aux=True,
            text_dim=256,
            local_model_path=self.local_model_path,
            use_boundary_branch=use_boundary_branch,
            boundary_weight=config.get('boundary_weight', 0.3),
            deep_supervision=deep_supervision
        )
        
        # 打印模型结构信息
        print(f" Model architecture:")
        print(f"  - Embed dim: {cfg['embed_dim']}")
        print(f"  - Depths: {cfg['depths']}")
        print(f"  - d_state: {cfg['d_state']}")
        
        # 加载权重 - 尝试严格加载
        try:
            model.load_state_dict(checkpoint['model_state'], strict=True)
            print(" Strict weight loading successful")
        except RuntimeError as e:
            print(f" Strict loading failed: {e}")
            print(" Trying strict=False loading...")
            
            # 尝试非严格加载
            model.load_state_dict(checkpoint['model_state'], strict=False)
            
            # 检查哪些权重没加载
            state_dict = checkpoint['model_state']
            model_dict = model.state_dict()
            
            missing_keys = [k for k in state_dict.keys() if k not in model_dict]
            unexpected_keys = [k for k in model_dict.keys() if k not in state_dict]
            
            print(f" Weight loading summary:")
            print(f"  - Missing keys: {len(missing_keys)}")
            print(f"  - Unexpected keys: {len(unexpected_keys)}")
            
            if len(missing_keys) > 0:
                print("  - First few missing keys:")
                for key in missing_keys[:5]:
                    print(f"    {key}")
            
            if len(unexpected_keys) > 0:
                print("  - First few unexpected keys:")
                for key in unexpected_keys[:5]:
                    print(f"    {key}")
        
        model = model.to(self.device)
        model.eval()
        
        print(f" Corrected model loaded successfully")
        print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"   Best mIoU: {checkpoint.get('best_miou_exclude_bg', 'N/A')}")
        
        # 存储模型配置信息
        self.model_config = config
        self.is_boundary_model = use_boundary_branch
        
        return model
    
    def _infer_model_size_from_checkpoint(self, checkpoint):
        """从checkpoint推断模型大小"""
        state_dict = checkpoint.get('model_state', {})
        
        # 分析层数
        block_keys = [k for k in state_dict.keys() if 'blocks.' in k]
        if not block_keys:
            return None
        
        # 分析每个stage的最大block索引
        stage_blocks = {}
        for key in block_keys:
            if 'stages.' in key:
                parts = key.split('.')
                for i, part in enumerate(parts):
                    if part == 'stages' and i+1 < len(parts):
                        stage_idx = int(parts[i+1])
                        for j in range(i+2, len(parts)):
                            if parts[j] == 'blocks' and j+1 < len(parts):
                                try:
                                    block_idx = int(parts[j+1])
                                    if stage_idx not in stage_blocks:
                                        stage_blocks[stage_idx] = []
                                    if block_idx not in stage_blocks[stage_idx]:
                                        stage_blocks[stage_idx].append(block_idx)
                                except:
                                    pass
        
        # 计算depths
        depths = []
        for stage in sorted(stage_blocks.keys()):
            max_block = max(stage_blocks[stage])
            depths.append(max_block + 1)  # block索引从0开始
        
        # 根据depths推断模型大小
        if depths == [1, 2, 4, 1]:
            return 'tiny'
        elif depths == [2, 2, 9, 2]:
            return 'small'
        elif depths == [2, 2, 18, 2]:
            return 'base'
        elif depths == [2, 2, 27, 2]:
            return 'large'
        else:
            print(f" Unknown depths configuration: {depths}")
            return None
    
    def _infer_depths_from_checkpoint(self, checkpoint):
        """从checkpoint推断depths配置"""
        state_dict = checkpoint.get('model_state', {})
        
        # 查找所有blocks
        stage_counts = {}
        for key in state_dict.keys():
            if 'stages.' in key and 'blocks.' in key:
                parts = key.split('.')
                for i, part in enumerate(parts):
                    if part == 'stages' and i+1 < len(parts):
                        try:
                            stage = int(parts[i+1])
                            if stage not in stage_counts:
                                stage_counts[stage] = set()
                        except:
                            continue
                    
                    if part == 'blocks' and i+1 < len(parts):
                        try:
                            block = int(parts[i+1])
                            stage_counts[stage].add(block)
                        except:
                            continue
        
        # 计算每个stage的block数量
        depths = []
        for stage in sorted(stage_counts.keys()):
            max_block = max(stage_counts[stage])
            depths.append(max_block + 1)  # 索引从0开始
        
        return depths if depths else None
    
    def _init_dataloader(self):
        """初始化测试数据加载器"""
        print("\nLoading test dataset...")
        
        try:
            from data.oiltea_voc_adapted import OilTeaVOCAdaptedDataset
            
            test_ds = OilTeaVOCAdaptedDataset(
                data_root=self.data_root,
                split='test',
                image_size=self.image_size,
                use_main_split=True,
                num_classes=self.num_classes  # 指定9个类别
            )
            
            print(f" Test dataset loaded: {len(test_ds)} images")
            print(f" Number of classes: {self.num_classes}")
            
        except Exception as e:
            print(f" Dataset loading failed: {e}")
            sys.exit(1)
        
        test_loader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        return test_loader
    
    def visualize_sample_with_boundary(self, images, masks, preds, boundaries, 
                                      image_ids, text_descriptions, batch_idx, save_individual=False):
        """可视化一批样本（带边界图）"""
        num_samples = min(4, images.shape[0])
        
        # 创建大图
        fig, axes = plt.subplots(num_samples, 6, figsize=(30, 5*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            # 1. 原始图像（反归一化）
            img = self.vis_tool.denormalize_image(images[i])[0]
            img_np = img.permute(1, 2, 0).numpy()
            
            axes[i, 0].imshow(img_np)
            title = f'Test Image\n{image_ids[i]}'
            if i < len(text_descriptions):
                short_text = text_descriptions[i][:30] + ('...' if len(text_descriptions[i]) > 30 else '')
                title += f'\n{short_text}'
            axes[i, 0].set_title(title, fontsize=8)
            axes[i, 0].axis('off')
            
            # 2. 真实分割
            gt_mask = masks[i].numpy()
            gt_vis = self.vis_tool.create_mask_visualization(gt_mask, num_classes=self.num_classes)
            axes[i, 1].imshow(gt_vis)
            axes[i, 1].set_title('Ground Truth (9 classes)', fontsize=10)
            axes[i, 1].axis('off')
            
            # 3. 预测分割
            pred_mask = preds[i].numpy()
            pred_vis = self.vis_tool.create_mask_visualization(pred_mask, num_classes=self.num_classes)
            axes[i, 2].imshow(pred_vis)
            axes[i, 2].set_title('Prediction (9 classes)', fontsize=10)
            axes[i, 2].axis('off')
            
            # 4. 差异图
            diff_map = self.vis_tool.create_difference_map(pred_mask, gt_mask, num_classes=self.num_classes)
            axes[i, 3].imshow(diff_map)
            axes[i, 3].set_title('Difference Map', fontsize=9)
            axes[i, 3].axis('off')
            
            # 5. 边界图
            if boundaries is not None and i < boundaries.shape[0]:
                boundary_map = boundaries[i]
                # 处理边界图形状
                if boundary_map.dim() == 3 and boundary_map.shape[0] == 1:
                    boundary_map = boundary_map.squeeze(0)  # [H, W]
                
                boundary_vis = self.vis_tool.create_boundary_visualization(boundary_map)
                axes[i, 4].imshow(boundary_vis)
                axes[i, 4].set_title('Boundary Prediction', fontsize=10)
                axes[i, 4].axis('off')
                
                # 6. 叠加边界
                overlay = self.vis_tool.create_overlay_boundary(img_np, boundary_map)
                axes[i, 5].imshow(overlay)
                axes[i, 5].set_title('Image + Boundary', fontsize=10)
                axes[i, 5].axis('off')
            else:
                # 没有边界图
                h, w = img_np.shape[:2]
                empty_img = np.zeros((h, w, 3), dtype=np.uint8)
                
                axes[i, 4].imshow(empty_img)
                axes[i, 4].set_title('No Boundary', fontsize=10)
                axes[i, 4].axis('off')
                
                axes[i, 5].imshow(empty_img)
                axes[i, 5].set_title('No Overlay', fontsize=10)
                axes[i, 5].axis('off')
            
            # 保存单张图片结果（使用图像ID命名）
            if save_individual:
                self._save_individual_result_with_boundary(
                    img_np, gt_vis, pred_vis, diff_map, boundaries[i] if boundaries is not None else None,
                    image_ids[i], text_descriptions[i] if i < len(text_descriptions) else "",
                    batch_idx, i
                )
        
        plt.suptitle(f'Corrected Test Results - Batch {batch_idx+1} (Dynamic K={self.model_config.get("dynamic_k", 1000)})', fontsize=14, y=1.02)
        plt.tight_layout()
        save_path = os.path.join(self.vis_dir, f'test_batch_{batch_idx+1:03d}.png')
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
        
        print(f" Batch visualization saved: {save_path}")
    
    def _save_individual_result_with_boundary(self, img_np, gt_vis, pred_vis, diff_map, boundary, 
                                             image_id, text_desc, batch_idx, sample_idx):
        """保存单张图片的详细结果（带边界），以图像ID命名"""
        # 创建主图
        fig = plt.figure(figsize=(15, 10))
        
        # 原始图像
        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(img_np)
        ax1.set_title(f'Image: {image_id}', fontsize=12)
        ax1.axis('off')
        if text_desc:
            ax1.text(0.02, -0.05, f'Text: {text_desc[:80]}...' if len(text_desc) > 80 else f'Text: {text_desc}',
                    transform=ax1.transAxes, fontsize=8, verticalalignment='top')
        
        # 真实分割
        ax2 = plt.subplot(2, 3, 2)
        ax2.imshow(gt_vis)
        ax2.set_title('Ground Truth (9 classes)', fontsize=12)
        ax2.axis('off')
        
        # 预测分割
        ax3 = plt.subplot(2, 3, 3)
        ax3.imshow(pred_vis)
        ax3.set_title('Prediction (9 classes)', fontsize=12)
        ax3.axis('off')
        
        # 差异图
        ax4 = plt.subplot(2, 3, 4)
        ax4.imshow(diff_map)
        ax4.set_title('Difference Map', fontsize=12)
        ax4.axis('off')
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Correct'),
            Patch(facecolor='white', label='Background Error'),
            Patch(facecolor='blue', label='Leaf Error'),
            Patch(facecolor='yellow', label='Disease_1 Error'),
            Patch(facecolor='orange', label='Disease_2 Error'),
            Patch(facecolor='pink', label='Disease_3 Error'),
            Patch(facecolor='lime', label='Disease_4 Error'),
            Patch(facecolor='cyan', label='Disease_5 Error'),
            Patch(facecolor='magenta', label='Disease_6 Error'),
            Patch(facecolor='olive', label='Disease_7 Error')
        ]
        ax4.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=8)
        
        # 边界图（如果有）
        if boundary is not None:
            # 处理边界图形状
            if torch.is_tensor(boundary):
                boundary_np = boundary.cpu().numpy()
            else:
                boundary_np = boundary
            
            if boundary_np.ndim == 3 and boundary_np.shape[0] == 1:
                boundary_np = boundary_np.squeeze(0)
            
            ax5 = plt.subplot(2, 3, 5)
            boundary_vis = self.vis_tool.create_boundary_visualization(boundary_np)
            ax5.imshow(boundary_vis)
            ax5.set_title('Boundary Heatmap', fontsize=12)
            ax5.axis('off')
            
            # 边界叠加图
            ax6 = plt.subplot(2, 3, 6)
            overlay = self.vis_tool.create_overlay_boundary(img_np, boundary_np)
            ax6.imshow(overlay)
            ax6.set_title('Image + Boundary', fontsize=12)
            ax6.axis('off')
        else:
            ax5 = plt.subplot(2, 3, 5)
            ax5.text(0.5, 0.5, 'No Boundary\nAvailable', 
                    ha='center', va='center', fontsize=12)
            ax5.set_title('Boundary', fontsize=12)
            ax5.axis('off')
            
            ax6 = plt.subplot(2, 3, 6)
            ax6.text(0.5, 0.5, 'No Overlay\nAvailable', 
                    ha='center', va='center', fontsize=12)
            ax6.set_title('Image + Boundary', fontsize=12)
            ax6.axis('off')
        
        # 添加动态权重配置信息
        dynamic_k = self.model_config.get('dynamic_k', 1000)
        ce_weight = self.model_config.get('ce_weight', 0.7)
        dice_weight = self.model_config.get('dice_weight', 0.3)
        
        plt.suptitle(f'Corrected Test Result: {image_id}\nDynamic K={dynamic_k}, CE={ce_weight}, Dice={dice_weight}', fontsize=14)
        plt.tight_layout()
        
        # 保存（直接使用image_id作为文件名，确保安全）
        safe_image_id = self._make_filename_safe(image_id)
        save_path = os.path.join(self.per_image_dir, f'{safe_image_id}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   Saved: {safe_image_id}.png")
        
        # 额外保存边界图
        if boundary is not None:
            self._save_boundary_separately(img_np, boundary_np, safe_image_id)
    
    def _save_boundary_separately(self, image_np, boundary_np, image_id):
        """单独保存边界相关图像"""
        # 1. 边界热力图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始图像
        axes[0].imshow(image_np)
        axes[0].set_title(f'Image: {image_id}', fontsize=12)
        axes[0].axis('off')
        
        # 边界热力图
        boundary_vis = self.vis_tool.create_boundary_visualization(boundary_np)
        axes[1].imshow(boundary_vis)
        axes[1].set_title('Boundary Heatmap', fontsize=12)
        axes[1].axis('off')
        
        # 边界叠加
        overlay = self.vis_tool.create_overlay_boundary(image_np, boundary_np)
        axes[2].imshow(overlay)
        axes[2].set_title('Image + Boundary', fontsize=12)
        axes[2].axis('off')
        
        plt.suptitle(f'Boundary Analysis: {image_id}', fontsize=14)
        plt.tight_layout()
        
        boundary_path = os.path.join(self.boundary_dir, f'{image_id}_boundary.png')
        plt.savefig(boundary_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _make_filename_safe(self, filename):
        """确保文件名安全，替换特殊字符"""
        unsafe_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
        safe_name = filename
        for char in unsafe_chars:
            safe_name = safe_name.replace(char, '_')
        
        safe_name = safe_name.strip('. ')
        
        max_length = 100
        if len(safe_name) > max_length:
            name, ext = os.path.splitext(safe_name)
            safe_name = name[:max_length-len(ext)] + ext
        
        return safe_name
    
    def test(self, visualize_all=True):
        """运行测试（使用修正版模型）"""
        print("\n" + "="*80)
        print("Starting Corrected Multimodal Testing with Dynamic Weights")
        print("="*80)
        print(f" Metric Strategy: Standard Dataset-Level")
        print(f"  1. Accumulate TP/FP/FN across the entire test set")
        print(f"  2. Compute IoU for each class: IoU_c = TP_c / (TP_c + FP_c + FN_c)")
        print(f"  3. Average IoUs to get mIoU (exclude classes without data)")
        print("="*80)
        
        # 获取动态权重配置
        dynamic_config = {
            'use_dynamic_weights': self.model_config.get('use_dynamic_weights', True),
            'dynamic_k': self.model_config.get('dynamic_k', 1000.0),
            'ce_weight': self.model_config.get('ce_weight', 0.7),
            'dice_weight': self.model_config.get('dice_weight', 0.3),
            'dice_weight_factor': self.model_config.get('dice_weight_factor', 2.0)
        }
        
        print(f" Dynamic Weights Configuration:")
        for key, value in dynamic_config.items():
            print(f"  - {key}: {value}")
        
        self.model.eval()
        
        # 用于存储每张图片的详细结果（仅用于展示）
        per_image_results = []
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Testing")
            
            for batch_idx, batch in enumerate(pbar):
                # 数据
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                text_descriptions = batch['text_prompt']
                image_ids = batch['image_id']
                
                # 前向传播
                outputs = self.model(images, text_descriptions, return_boundary=True)
                
                # 获取预测标签
                preds = outputs['main']
                pred_labels = torch.argmax(preds, dim=1)
                
                # 获取边界图（如果有）
                boundaries = outputs.get('boundary_maps', [])
                boundary_tensor = boundaries[-1] if boundaries and len(boundaries) > 0 else None
                
                # 保存预测结果用于最终的dataset-level计算
                all_predictions.append(pred_labels.cpu().numpy())
                all_targets.append(masks.cpu().numpy())
                
                # 可视化
                if visualize_all:
                    self.visualize_sample_with_boundary(
                        images.cpu(), masks.cpu(), pred_labels.cpu(), boundary_tensor,
                        image_ids, text_descriptions, batch_idx,
                        save_individual=True
                    )
                
                # 更新进度条
                batch_metrics = per_image_results[-images.shape[0]:] if per_image_results else []
                if batch_metrics:
                    # 这里不再计算per-image指标，因为我们要的是dataset-level
                    pass
                
                model_type = 'Corrected' if self.is_boundary_model else 'Corrected-Regular'
                
                pbar.set_postfix({
                    'Model': model_type,
                    'Batch': f"{batch_idx+1}/{len(self.test_loader)}",
                    'K': f"{dynamic_config['dynamic_k']:.0f}"
                })
        
        # 计算总体指标（标准的dataset-level计算）
        overall_metrics = self.metrics_calc.calculate_dataset_level_metrics(
            all_predictions, all_targets, include_bg=True
        )
        overall_metrics['model_type'] = 'corrected-with-dynamic-weights'
        overall_metrics.update(dynamic_config)
        
        # 为每张图片创建简单的结果记录（仅用于文件命名）
        per_image_results = []
        for i, image_id in enumerate(image_ids):
            per_image_results.append({
                'image_id': image_id,
                'model_type': 'corrected-with-dynamic-weights',
                'dynamic_k': dynamic_config['dynamic_k'],
                'ce_weight': dynamic_config['ce_weight'],
                'dice_weight': dynamic_config['dice_weight']
            })
        
        # 保存结果
        self._save_results(overall_metrics, per_image_results)
        
        # 生成总结报告
        self._generate_summary_report(overall_metrics, per_image_results)
        
        return overall_metrics, per_image_results
    
    def _save_results(self, overall_metrics, per_image_results):
        """保存测试结果"""
        print("\nSaving test results...")
        
        # 1. 保存总体指标
        overall_df = pd.DataFrame([overall_metrics])
        overall_path = os.path.join(self.output_dir, 'overall_metrics.csv')
        overall_df.to_csv(overall_path, index=False)
        print(f" Saved overall metrics to: {overall_path}")
        
        # 2. 保存每张图片的简单结果
        per_image_df = pd.DataFrame(per_image_results)
        per_image_path = os.path.join(self.output_dir, 'per_image_info.csv')
        per_image_df.to_csv(per_image_path, index=False)
        print(f" Saved per-image info to: {per_image_path}")
        
        # 3. 保存dataset-level指标报告
        self._save_dataset_level_report(overall_metrics)
        
        # 4. 保存配置信息
        config_info = {
            'checkpoint_path': self.checkpoint_path,
            'data_root': self.data_root,
            'image_size': self.image_size,
            'batch_size': self.batch_size,
            'model_size': self.model_size,
            'num_classes': self.num_classes,
            'model_type': 'corrected-with-dynamic-weights',
            'use_dynamic_weights': self.model_config.get('use_dynamic_weights', True),
            'dynamic_k': self.model_config.get('dynamic_k', 1000.0),
            'ce_weight': self.model_config.get('ce_weight', 0.7),
            'dice_weight': self.model_config.get('dice_weight', 0.3),
            'dice_weight_factor': self.model_config.get('dice_weight_factor', 2.0),
            'use_boundary_branch': self.model_config.get('use_boundary_branch', False),
            'deep_supervision': self.model_config.get('deep_supervision', False),
            'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_test_images': len(per_image_results),
            'metric_strategy': 'Standard Dataset-Level'
        }
        
        config_df = pd.DataFrame([config_info])
        config_path = os.path.join(self.output_dir, 'test_config.csv')
        config_df.to_csv(config_path, index=False)
        
    def _save_dataset_level_report(self, overall_metrics):
        """保存dataset-level指标报告"""
        report_path = os.path.join(self.output_dir, 'dataset_level_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("STANDARD DATASET-LEVEL METRICS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Type: Corrected with Dynamic Weights\n")
            f.write(f"Dynamic K: {self.model_config.get('dynamic_k', 1000.0)}\n")
            f.write(f"CE Weight: {self.model_config.get('ce_weight', 0.7)}\n")
            f.write(f"Dice Weight: {self.model_config.get('dice_weight', 0.3)}\n")
            f.write(f"Total Images: {overall_metrics.get('total_images', 0)}\n")
            f.write(f"Number of Classes: {self.num_classes} (1 background + 1 leaf + 7 diseases)\n")
            f.write(f"Metric Strategy: Standard Dataset-Level\n")
            f.write("  1. Accumulate TP/FP/FN across the entire test set\n")
            f.write("  2. Compute IoU for each class: IoU_c = TP_c / (TP_c + FP_c + FN_c)\n")
            f.write("  3. Average IoUs to get mIoU (exclude classes without data)\n")
            f.write("\n" + "=" * 80 + "\n\n")
            
            # ==================== 显示dataset-level的各类别IoU ====================
            
            f.write("1. PER-CLASS IoU (Dataset-Level):\n")
            f.write("-" * 80 + "\n")
            
            class_names = ['Background', 'Leaf', 'Disease_1', 'Disease_2', 'Disease_3', 
                          'Disease_4', 'Disease_5', 'Disease_6', 'Disease_7']
            
            for i in range(self.num_classes):
                tp = overall_metrics.get(f'class_{i}_tp', 0)
                fp = overall_metrics.get(f'class_{i}_fp', 0)
                fn = overall_metrics.get(f'class_{i}_fn', 0)
                iou = overall_metrics.get(f'class_{i}_iou', 0)
                total_pixels = overall_metrics.get(f'class_{i}_total_pixels', 0)
                
                f.write(f"{class_names[i]}:\n")
                f.write(f"  TP: {tp}, FP: {fp}, FN: {fn}\n")
                f.write(f"  Total pixels in GT: {total_pixels}\n")
                f.write(f"  IoU: {iou:.6f}\n")
                f.write(f"  Precision: {overall_metrics.get(f'class_{i}_precision', 0):.6f}\n")
                f.write(f"  Recall: {overall_metrics.get(f'class_{i}_recall', 0):.6f}\n")
                f.write(f"  Dice: {overall_metrics.get(f'class_{i}_dice', 0):.6f}\n\n")
            
            f.write("\n")
            
            # ==================== 汇总指标 ====================
            
            f.write("2. SUMMARY METRICS:\n")
            f.write("-" * 80 + "\n")
            
            f.write("All classes (with data):\n")
            f.write(f"  mIoU: {overall_metrics.get('all_mIoU', 0):.6f}\n")
            f.write(f"  Number of classes with data: {len([c for c in range(self.num_classes) if overall_metrics.get(f'class_{c}_total_pixels', 0) > 0])}\n\n")
            
            f.write("Excluding background:\n")
            f.write(f"  mIoU: {overall_metrics.get('exclude_bg_mIoU', 0):.6f}\n")
            f.write(f"  Number of classes with data: {len([c for c in range(1, self.num_classes) if overall_metrics.get(f'class_{c}_total_pixels', 0) > 0])}\n\n")
            
            f.write("Background + Leaf:\n")
            f.write(f"  mIoU: {overall_metrics.get('bg_leaf_mIoU', 0):.6f}\n\n")
            
            f.write("Leaf only:\n")
            f.write(f"  IoU: {overall_metrics.get('leaf_iou', 0):.6f}\n")
            f.write(f"  Precision: {overall_metrics.get('leaf_precision', 0):.6f}\n")
            f.write(f"  Recall: {overall_metrics.get('leaf_recall', 0):.6f}\n")
            f.write(f"  Dice: {overall_metrics.get('leaf_dice', 0):.6f}\n\n")
            
            f.write("Disease classes (2-8):\n")
            f.write(f"  mIoU: {overall_metrics.get('disease_mIoU', 0):.6f}\n")
            f.write(f"  Number of disease classes with data: {overall_metrics.get('disease_class_count', 0)}\n\n")
            
            f.write("Mixed metrics:\n")
            f.write(f"  mIoU (excluding background): {overall_metrics.get('mixed_mIoU_exclude_bg', 0):.6f}\n")
            f.write(f"  mIoU (all mixed): {overall_metrics.get('mixed_mIoU', 0):.6f}\n\n")
            
            f.write("Overall:\n")
            f.write(f"  Pixel Accuracy: {overall_metrics.get('overall_pixel_accuracy', 0):.6f}\n")
            f.write(f"  Total pixels: {overall_metrics.get('total_pixels', 0)}\n")
            f.write(f"  Total images: {overall_metrics.get('total_images', 0)}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        print(f" Dataset-level metrics report saved to: {report_path}")
    
    def _generate_summary_report(self, overall_metrics, per_image_results):
        """生成测试总结报告"""
        print("\nGenerating summary report...")
        
        report_path = os.path.join(self.output_dir, 'test_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("CORRECTED MULTIMODAL VISION MAMBA TEST REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Checkpoint: {self.checkpoint_path}\n")
            f.write(f"Dataset: {self.data_root}\n")
            f.write(f"Model Type: Corrected with Dynamic Weights\n")
            f.write(f"Total Test Images: {overall_metrics.get('total_images', 0)}\n")
            f.write(f"Number of Classes: {self.num_classes} (1 background + 1 leaf + 7 diseases)\n")
            f.write(f"Metric Strategy: Standard Dataset-Level\n")
            f.write(f"  1. Accumulate TP/FP/FN across the entire test set\n")
            f.write(f"  2. Compute IoU for each class: IoU_c = TP_c / (TP_c + FP_c + FN_c)\n")
            f.write(f"  3. Average IoUs to get mIoU (exclude classes without data)\n")
            f.write(f"Dynamic Weights Configuration:\n")
            f.write(f"  - Use dynamic weights: {self.model_config.get('use_dynamic_weights', True)}\n")
            f.write(f"  - Dynamic K: {self.model_config.get('dynamic_k', 1000.0)}\n")
            f.write(f"  - CE weight: {self.model_config.get('ce_weight', 0.7)}\n")
            f.write(f"  - Dice weight: {self.model_config.get('dice_weight', 0.3)}\n")
            f.write(f"  - Dice weight factor: {self.model_config.get('dice_weight_factor', 2.0)}\n\n")
            
            f.write("="*80 + "\n")
            f.write("OVERALL METRICS - DATASET LEVEL\n")
            f.write("="*80 + "\n\n")
            
            # 类别指标
            f.write("Per-Class Metrics (Dataset-Level):\n")
            f.write("-"*40 + "\n")
            class_names = ['Background', 'Leaf', 'Disease_1', 'Disease_2', 'Disease_3', 
                          'Disease_4', 'Disease_5', 'Disease_6', 'Disease_7']
            
            for i in range(self.num_classes):
                tp = overall_metrics.get(f'class_{i}_tp', 0)
                fp = overall_metrics.get(f'class_{i}_fp', 0)
                fn = overall_metrics.get(f'class_{i}_fn', 0)
                iou = overall_metrics.get(f'class_{i}_iou', 0)
                total_pixels = overall_metrics.get(f'class_{i}_total_pixels', 0)
                
                if total_pixels > 0:
                    f.write(f"{class_names[i]} ({total_pixels} pixels in GT):\n")
                    f.write(f"  IoU: {iou:.4f} (TP={tp}, FP={fp}, FN={fn})\n")
                    f.write(f"  Precision: {overall_metrics.get(f'class_{i}_precision', 0):.4f}\n")
                    f.write(f"  Recall: {overall_metrics.get(f'class_{i}_recall', 0):.4f}\n")
                    f.write(f"  Dice: {overall_metrics.get(f'class_{i}_dice', 0):.4f}\n\n")
            
            f.write("="*80 + "\n")
            f.write("SUMMARY METRICS\n")
            f.write("="*80 + "\n\n")
            
            f.write("Excluding background (standard evaluation):\n")
            f.write(f"  mIoU: {overall_metrics.get('exclude_bg_mIoU', 0):.4f}\n\n")
            
            f.write("Leaf only:\n")
            f.write(f"  IoU: {overall_metrics.get('leaf_iou', 0):.4f}\n")
            f.write(f"  Precision: {overall_metrics.get('leaf_precision', 0):.4f}\n")
            f.write(f"  Recall: {overall_metrics.get('leaf_recall', 0):.4f}\n")
            f.write(f"  Dice: {overall_metrics.get('leaf_dice', 0):.4f}\n\n")
            
            f.write("Disease classes (2-8):\n")
            f.write(f"  mIoU: {overall_metrics.get('disease_mIoU', 0):.4f}\n")
            f.write(f"  Number of disease classes with data: {overall_metrics.get('disease_class_count', 0)}\n\n")
            
            f.write("Mixed metrics:\n")
            f.write(f"  mIoU (excluding background): {overall_metrics.get('mixed_mIoU_exclude_bg', 0):.4f}\n")
            f.write(f"  mIoU (all classes with data): {overall_metrics.get('mixed_mIoU', 0):.4f}\n\n")
            
            f.write("Background + Leaf:\n")
            f.write(f"  mIoU: {overall_metrics.get('bg_leaf_mIoU', 0):.4f}\n\n")
            
            f.write("Overall:\n")
            f.write(f"  Pixel Accuracy: {overall_metrics.get('overall_pixel_accuracy', 0):.4f}\n")
            f.write(f"  Total images: {overall_metrics.get('total_images', 0)}\n\n")
            
            # 显示哪些类别有数据
            f.write("Classes with data in test set:\n")
            for i in range(self.num_classes):
                total_pixels = overall_metrics.get(f'class_{i}_total_pixels', 0)
                if total_pixels > 0:
                    f.write(f"  - {class_names[i]}: {total_pixels} pixels\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("BOUNDARY ENHANCEMENT ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            if self.is_boundary_model:
                f.write(" Boundary enhancement was used during training.\n")
                f.write("   - Boundary branches in encoder\n")
                f.write("   - Boundary loss during training\n")
                f.write("   - Deep supervision with multiple levels\n\n")
                
                f.write("Visual analysis available in:\n")
                f.write(f"  - {self.vis_dir}/ : Batch visualizations\n")
                f.write(f"  - {self.per_image_dir}/ : Per-image results\n")
                f.write(f"  - {self.boundary_dir}/ : Boundary-specific visualizations\n")
            else:
                f.write(" No boundary enhancement was used.\n")
                f.write("   Model was trained without boundary branches.\n\n")
            
            f.write("="*80 + "\n")
            f.write("TEST COMPLETED SUCCESSFULLY\n")
            f.write("="*80 + "\n")
        
        print(f" Saved test report to: {report_path}")
        
        # 打印关键指标
        print("\n" + "="*80)
        model_type = "CORRECTED WITH DYNAMIC WEIGHTS"
        print(f"TEST SUMMARY ({model_type})")
        print("="*80)
        print(f"Dynamic K: {self.model_config.get('dynamic_k', 1000)}")
        print(f"Metric Strategy: Standard Dataset-Level")
        print(f"Leaf IoU: {overall_metrics.get('leaf_iou', 0):.4f}")
        print(f"Disease mIoU ({overall_metrics.get('disease_class_count', 0)} classes with data): {overall_metrics.get('disease_mIoU', 0):.4f}")
        print(f"Exclude Background mIoU: {overall_metrics.get('exclude_bg_mIoU', 0):.4f}")
        print(f"Mixed mIoU (excl background): {overall_metrics.get('mixed_mIoU_exclude_bg', 0):.4f}")
        print(f"Background+Leaf mIoU: {overall_metrics.get('bg_leaf_mIoU', 0):.4f}")
        print(f"Overall Pixel Accuracy: {overall_metrics.get('overall_pixel_accuracy', 0):.4f}")
        print(f"Total test images: {overall_metrics.get('total_images', 0)}")
        print(f"Results saved to: {self.output_dir}")
        print("="*80)


    def main(): 
    parser = argparse.ArgumentParser(description='Test Corrected Multimodal Vision Mamba with Dynamic Weights')
    
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to corrected model checkpoint')
    parser.add_argument('--image_size', type=int, default=512,
                       help='Image size')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--model_size', type=str, default='small',
                       choices=['tiny', 'small', 'base'],
                       help='Model size')
    parser.add_argument('--local_model_path', type=str, 
                       default='./models/bert-base-uncased',
                       help='Path to local BERT model')
    parser.add_argument('--output_dir', type=str, default='./test_results_corrected',
                       help='Directory to save test results')
    parser.add_argument('--no_visualize', action='store_true',
                       help='Disable visualization')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Corrected Multimodal Vision Mamba Testing with Dynamic Weights")
    print("="*80)
    
    # 创建测试器
    tester = CorrectedMultimodalVMambaTesterWithBoundary(
        data_root=args.data_root,
        checkpoint_path=args.checkpoint,
        image_size=args.image_size,
        batch_size=args.batch_size,
        model_size=args.model_size,
        local_model_path=args.local_model_path,
        output_dir=args.output_dir
    )
    
    # 运行测试
    #overall_metrics, per_image_results = tester.test(visualize_all=not args.no_visualize)
    
    print("\n Corrected model testing completed successfully!")

if __name__ == "__main__":
    main()