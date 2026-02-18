import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import argparse
import random
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f" Set random seed: {seed}")

# 导入修正版的多模态模型
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from models.multimodal_vmamba_local_with_boundary_corrected import (
        MultimodalVMambaSegLocalWithBoundary,
        CorrectDynamicWeightsLoss
    )
    print(" Successfully imported corrected multimodal model with dynamic weights")
except ImportError as e:
    print(f" Failed to import corrected multimodal model: {e}")
    # 尝试从同一目录导入
    sys.path.append(os.path.join(current_dir, 'models'))
    try:
        from multimodal_vmamba_local_with_boundary_corrected import (
            MultimodalVMambaSegLocalWithBoundary,
            CorrectDynamicWeightsLoss
        )
        print(" Successfully imported from models directory")
    except ImportError as e2:
        print(f" Failed to import: {e2}")
        sys.exit(1)

# 导入数据集
try:
    from data.oiltea_voc_adapted import OilTeaVOCAdaptedDataset
    print(" Successfully imported dataset class")
except ImportError as e:
    print(f" Failed to import dataset: {e}")
    sys.exit(1)

# ==================== 可视化工具 - 9个类别 ====================
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
    def create_mask_visualization(mask_tensor, num_classes=9):
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
            boundary_np = boundary_tensor.squeeze().cpu().numpy()
        else:
            boundary_np = boundary_tensor
        
        if boundary_np.ndim == 2:
            h, w = boundary_np.shape
            vis = np.zeros((h, w, 3), dtype=np.uint8)
            vis[boundary_np > 0.5] = [255, 255, 255]
        else:  # batch维度
            b, h, w = boundary_np.shape
            vis = np.zeros((b, h, w, 3), dtype=np.uint8)
            for i in range(b):
                vis[i][boundary_np[i] > 0.5] = [255, 255, 255]
        
        return vis
    
    @staticmethod
    def create_difference_map(pred_mask, true_mask, num_classes=9):
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
            
            correct_mask = (pred_np == true_np)
            diff[correct_mask] = [0, 255, 0]
            
            for class_id in range(num_classes):
                if class_id == 0:  # 背景
                    bg_fp = (pred_np == 0) & (true_np != 0)
                    bg_fn = (pred_np != 0) & (true_np == 0)
                    bg_error = bg_fp | bg_fn
                    bg_error = bg_error & ~correct_mask
                    diff[bg_error] = [255, 255, 255]
                
                elif class_id == 1:  # 叶片
                    leaf_fp = (pred_np == 1) & (true_np != 1)
                    leaf_fn = (pred_np != 1) & (true_np == 1)
                    leaf_error = leaf_fp | leaf_fn
                    leaf_error = leaf_error & ~correct_mask
                    diff[leaf_error] = [0, 0, 255]
                
                else:  # 病斑 (2-8)
                    disease_fp = (pred_np == class_id) & (true_np != class_id)
                    disease_fn = (pred_np != class_id) & (true_np == class_id)
                    disease_error = disease_fp | disease_fn
                    disease_error = disease_error & ~correct_mask
                    
                    disease_colors = {
                        2: [255, 255, 0],
                        3: [255, 165, 0],
                        4: [255, 0, 165],
                        5: [165, 255, 0],
                        6: [0, 255, 165],
                        7: [165, 0, 255],
                        8: [128, 128, 0]
                    }
                    if class_id in disease_colors:
                        diff[disease_error] = disease_colors[class_id]
        
        return diff

# ==================== 指标计算器（排除背景）- 基于图像真实存在的类别 ====================
class MetricsCalculatorExcludingBackground:
    def __init__(self, num_classes=9):
        self.num_classes = num_classes
        self.class_names = [
            'Background', 'Leaf', 
            'Disease_1', 'Disease_2', 'Disease_3', 'Disease_4',
            'Disease_5', 'Disease_6', 'Disease_7'
        ]
    
    def calculate_metrics(self, pred, target):
        """计算指标，只计算图像中真实存在的类别"""
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        metrics = {}
        
        # 找出图像中真实存在的类别（排除背景类别0）
        present_classes = np.unique(target_flat)
        present_classes = present_classes[present_classes > 0]
        
        # 为所有类别计算基础指标
        for i in range(self.num_classes):
            tp = ((pred_flat == i) & (target_flat == i)).sum()
            fp = ((pred_flat == i) & (target_flat != i)).sum()
            fn = ((pred_flat != i) & (target_flat == i)).sum()
            
            eps = 1e-8
            iou = tp / (tp + fp + fn + eps)
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            dice = 2 * tp / (2 * tp + fp + fn + eps)
            pa = tp / (tp + fn + eps)
            
            metrics[f'class_{i}_iou'] = float(iou)
            metrics[f'class_{i}_precision'] = float(precision)
            metrics[f'class_{i}_recall'] = float(recall)
            metrics[f'class_{i}_dice'] = float(dice)
            metrics[f'class_{i}_pa'] = float(pa)
        
        # 只计算图像中真实存在的类别的指标（排除背景）
        foreground_iou = []
        foreground_pa = []
        foreground_precision = []
        foreground_recall = []
        foreground_dice = []
        
        for i in range(1, self.num_classes):
            if i in present_classes:
                foreground_iou.append(metrics[f'class_{i}_iou'])
                foreground_pa.append(metrics[f'class_{i}_pa'])
                foreground_precision.append(metrics[f'class_{i}_precision'])
                foreground_recall.append(metrics[f'class_{i}_recall'])
                foreground_dice.append(metrics[f'class_{i}_dice'])
        
        metrics['mIoU_exclude_bg'] = float(np.mean(foreground_iou)) if foreground_iou else 0.0
        metrics['mPA_exclude_bg'] = float(np.mean(foreground_pa)) if foreground_pa else 0.0
        metrics['mPrecision_exclude_bg'] = float(np.mean(foreground_precision)) if foreground_precision else 0.0
        metrics['mRecall_exclude_bg'] = float(np.mean(foreground_recall)) if foreground_recall else 0.0
        metrics['mDice_exclude_bg'] = float(np.mean(foreground_dice)) if foreground_dice else 0.0
        
        # 病斑特定指标（类别2-8）- 只计算真实存在的病斑类别
        disease_iou = []
        for i in range(2, self.num_classes):
            if i in present_classes:
                disease_iou.append(metrics[f'class_{i}_iou'])
        metrics['disease_mIoU'] = float(np.mean(disease_iou)) if disease_iou else 0.0
        
        # 叶片指标
        if 1 in present_classes:
            metrics['leaf_iou'] = metrics['class_1_iou']
            metrics['leaf_precision'] = metrics['class_1_precision']
            metrics['leaf_recall'] = metrics['class_1_recall']
            metrics['leaf_pa'] = metrics['class_1_pa']
            metrics['leaf_dice'] = metrics['class_1_dice']
        else:
            metrics['leaf_iou'] = 0.0
            metrics['leaf_precision'] = 0.0
            metrics['leaf_recall'] = 0.0
            metrics['leaf_pa'] = 0.0
            metrics['leaf_dice'] = 0.0
        
        eps = 1e-8
        if metrics['mPrecision_exclude_bg'] + metrics['mRecall_exclude_bg'] > 0:
            metrics['mF1_exclude_bg'] = float(2 * metrics['mPrecision_exclude_bg'] * metrics['mRecall_exclude_bg'] / 
                                          (metrics['mPrecision_exclude_bg'] + metrics['mRecall_exclude_bg'] + eps))
        else:
            metrics['mF1_exclude_bg'] = 0.0
        
        metrics['present_classes_count'] = int(len(present_classes))
        
        return metrics

# ==================== 修正版的多模态训练器（使用CorrectDynamicWeightsLoss） ====================
class MultimodalVMambaTrainerCorrected:
    def __init__(self, data_root, image_size=512, batch_size=4, 
                 lr=0.0002, epochs=100, save_dir='./checkpoints_corrected',
                 seed=42, use_cuda=True, model_size='base',
                 local_model_path='./models/bert-base-uncased',
                 visualize_freq=5, boundary_weight=0.3,
                 use_boundary_branch=True, deep_supervision=True,
                 # 新增：动态权重参数
                 use_dynamic_weights=True,
                 dynamic_k=1000.0,
                 ce_weight=0.7,
                 dice_weight=0.3,
                 dice_weight_factor=2.0,
                 class_weights=None):
        
        set_seed(seed)
        
        self.data_root = data_root
        self.image_size = image_size
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.save_dir = save_dir
        self.seed = seed
        self.model_size = model_size
        self.local_model_path = local_model_path
        self.visualize_freq = visualize_freq
        self.boundary_weight = boundary_weight
        self.use_boundary_branch = use_boundary_branch
        self.deep_supervision = deep_supervision
        self.num_classes = 9
        
        # 动态权重参数
        self.use_dynamic_weights = use_dynamic_weights
        self.dynamic_k = dynamic_k
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.dice_weight_factor = dice_weight_factor
        self.class_weights = class_weights
        
        print("=" * 80)
        print("Multimodal Vision Mamba Trainer with Corrected Dynamic Weights")
        print("=" * 80)
        
        if not os.path.exists(data_root):
            print(f" Dataset does not exist: {data_root}")
            sys.exit(1)
        
        print(f" Dataset: {data_root}")
        print(f"  Image size: {image_size}")
        print(f" Batch size: {batch_size}")
        print(f" Learning rate: {lr}")
        print(f" Epochs: {epochs}")
        print(f" Model: {model_size}")
        print(f" Number of classes: {self.num_classes}")
        print(f" Local BERT: {local_model_path}")
        print(f" Boundary enhancement: {use_boundary_branch}")
        print(f" Deep supervision: {deep_supervision}")
        print(f"\n Corrected Dynamic Weights Configuration:")
        print(f"   Use dynamic weights: {use_dynamic_weights}")
        print(f"   Dynamic K: {dynamic_k}")
        print(f"   CE weight: {ce_weight}")
        print(f"   Dice weight: {dice_weight}")
        print(f"   Dice weight factor: {dice_weight_factor}")
        print(f" Visualization frequency: Every {visualize_freq} epochs")
        print(f" Metrics calculation: Based ONLY on classes present in each image")
        
        # 创建目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_name = f"multimodal_vmamba_corrected_{model_size}_{timestamp}"
        self.result_dir = f"./results/{self.exp_name}"
        
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(os.path.join(self.result_dir, "visualizations", "train"), exist_ok=True)
        os.makedirs(os.path.join(self.result_dir, "visualizations", "val"), exist_ok=True)
        
        # 设备
        self.device = self._setup_device(use_cuda)
        
        # 初始化
        self.model = self._init_model()
        self.train_loader, self.val_loader = self._init_dataloaders()
        self.optimizer = self._init_optimizer()
        
        #  使用修正版的损失函数
        self.criterion = CorrectDynamicWeightsLoss(
            num_classes=self.num_classes,
            device=self.device,
            boundary_weight=boundary_weight,
            deep_supervision_weights=[0.1, 0.2, 0.3],
            use_focal_loss=False,  # 先不使用Focal Loss
            dice_weight_factor=dice_weight_factor,
            use_dynamic_weights=use_dynamic_weights,
            dynamic_k=dynamic_k,
            ce_weight=ce_weight,
            dice_weight=dice_weight,
            class_weights=class_weights
        )
        
        self.metrics_calc = MetricsCalculatorExcludingBackground(num_classes=self.num_classes)
        self.vis_tool = VisualizationToolWithBoundary()
        
        # 训练状态
        self.epoch = 0
        self.best_miou_exclude_bg = 0.0
        self.best_disease_miou = 0.0
        self.best_epoch = 0
        self.history = []
        
        print(" Corrected multimodal trainer initialized successfully")
    
    def _setup_device(self, use_cuda):
        if use_cuda and torch.cuda.is_available():
            device = torch.device('cuda:0')
            torch.cuda.empty_cache()
            print(f" GPU: {torch.cuda.get_device_name(device)}")
            print(f" GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")
        else:
            device = torch.device('cpu')
            print(" CPU")
        return device
    
    def _init_model(self):
        # 模型配置
        configs = {
            'tiny': {'embed_dim': 64, 'depths': [1, 2, 4, 1], 'd_state': 8},
            'small': {'embed_dim': 96, 'depths': [2, 2, 9, 2], 'd_state': 16},
            'base': {'embed_dim': 128, 'depths': [2, 2, 18, 2], 'd_state': 32}
        }
        
        if self.model_size not in configs:
            print(f" Unknown model size {self.model_size}, using base")
            self.model_size = 'base'
        
        cfg = configs[self.model_size]
        
        print(f"Initializing multimodal {self.model_size} model with corrected loss...")
        model = MultimodalVMambaSegLocalWithBoundary(
            num_classes=self.num_classes,
            embed_dim=cfg['embed_dim'],
            depths=cfg['depths'],
            d_state=cfg['d_state'],
            use_aux=True,
            text_dim=256,
            local_model_path=self.local_model_path,
            use_boundary_branch=self.use_boundary_branch,
            boundary_weight=self.boundary_weight,
            deep_supervision=self.deep_supervision
        )
        
        model = model.to(self.device)
        
        params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f" Model parameters: {params:,} (Trainable: {trainable:,})")
        
        return model
    
    def _init_dataloaders(self):
        print(f"\nLoading multimodal dataset ({self.num_classes} classes)...")
        
        try:
            train_ds = OilTeaVOCAdaptedDataset(
                data_root=self.data_root,
                split='train',
                image_size=self.image_size,
                use_main_split=True,
                num_classes=self.num_classes
            )
            
            val_ds = OilTeaVOCAdaptedDataset(
                data_root=self.data_root,
                split='val',
                image_size=self.image_size,
                use_main_split=True,
                num_classes=self.num_classes
            )
            
            print(f" Multimodal dataset: Train {len(train_ds)} | Val {len(val_ds)}")
            print(f" Number of classes: {self.num_classes}")
            
            # 检查数据集中的标签范围
            if len(train_ds) > 0:
                sample = train_ds[0]
                print(f" Sample check:")
                print(f"  Image shape: {sample['image'].shape}")
                print(f"  Mask shape: {sample['mask'].shape}")
                
                if torch.is_tensor(sample['mask']):
                    mask_np = sample['mask'].numpy()
                else:
                    mask_np = np.array(sample['mask'])
                
                print(f"  Mask value range: [{mask_np.min()}, {mask_np.max()}]")
                unique_vals = np.unique(mask_np)
                print(f"  Unique mask values: {unique_vals.tolist()}")
                
                # 检查是否有无效值
                invalid_vals = [v for v in unique_vals if v < 0 or v >= self.num_classes]
                if invalid_vals:
                    print(f"  Warning: Found invalid mask values: {invalid_vals}")
                    print(f"    These will be clamped to [0, {self.num_classes-1}] during training")
                
                print(f" Text: {sample['text_prompt'][:100]}...")
            
        except Exception as e:
            print(f" Dataset loading failed: {e}")
            sys.exit(1)
        
        # 创建安全的数据集包装器
        class SafeDataset(torch.utils.data.Dataset):
            def __init__(self, dataset, num_classes):
                self.dataset = dataset
                self.num_classes = num_classes
            
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self, idx):
                item = self.dataset[idx]
                
                # 修复mask
                mask = item['mask']
                if not torch.is_tensor(mask):
                    mask = torch.tensor(mask)
                
                # 确保mask值在有效范围内
                mask = torch.clamp(mask, 0, self.num_classes - 1)
                mask = mask.long()  # 确保是长整型
                
                item['mask'] = mask
                return item
        
        train_ds = SafeDataset(train_ds, self.num_classes)
        val_ds = SafeDataset(val_ds, self.num_classes)
        
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def _init_optimizer(self):
        # 对文本编码器使用较小的学习率
        text_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'text_encoder' in name:
                text_params.append(param)
            else:
                other_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {'params': text_params, 'lr': self.lr * 0.1},
            {'params': other_params, 'lr': self.lr}
        ], weight_decay=0.05, betas=(0.9, 0.999))
        
        # 调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.epochs,
            eta_min=self.lr * 0.01
        )
        
        print(f" Optimizer: AdamW (Text encoder lr={self.lr*0.1:.6f}, Other lr={self.lr:.6f})")
        return optimizer
    
    def train_epoch(self, collect_samples=False):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_samples = 0
        all_metrics = []
        sample_batches = []
        sample_texts = []
        
        pbar = tqdm(self.train_loader, desc=f"Training Epoch {self.epoch+1}/{self.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device, non_blocking=True)
            masks = batch['mask'].to(self.device, non_blocking=True)
            text_descriptions = batch['text_prompt']
            
            #  安全检查：确保mask值有效
            if masks.max() >= self.num_classes or masks.min() < 0:
                print(f" Batch {batch_idx}: Invalid mask range [{masks.min().item()}, {masks.max().item()}]")
                masks = torch.clamp(masks, 0, self.num_classes - 1)
            
            # 确保数据类型正确
            if masks.dtype != torch.long:
                masks = masks.long()
            
            # 统计小目标信息
            small_targets_info = []
            for c in range(2, self.num_classes):  # 病斑类别
                area = (masks == c).float().sum().item()
                if 0 < area < 100:  # 小于100像素
                    small_targets_info.append((c, area))
            
            # 前向传播
            outputs = self.model(images, text_descriptions, return_boundary=self.use_boundary_branch)
            
            # 检查输出形状
            if outputs['main'].shape[1] != self.num_classes:
                print(f" Error: main output channels {outputs['main'].shape[1]} != {self.num_classes}")
                continue
            
            # 计算损失
            loss, loss_dict = self.criterion(outputs, masks)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 优化
            self.optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            
            # 收集样本用于可视化
            if collect_samples and batch_idx == 0:
                with torch.no_grad():
                    pred_labels = torch.argmax(outputs['main'], dim=1)
                    sample_batches.append({
                        'images': images[:4].cpu(),
                        'masks': masks[:4].cpu(),
                        'preds': pred_labels[:4].cpu(),
                        'boundaries': outputs.get('boundary_maps', [])[-1][:4].cpu() if outputs.get('boundary_maps') else None
                    })
                    sample_texts.extend(text_descriptions[:4])
            
            # 计算指标
            with torch.no_grad():
                pred_np = torch.argmax(outputs['main'], dim=1).cpu().numpy()
                mask_np = masks.cpu().numpy()
                
                batch_metrics = []
                for i in range(images.size(0)):
                    metrics = self.metrics_calc.calculate_metrics(pred_np[i], mask_np[i])
                    batch_metrics.append(metrics)
                
                all_metrics.extend(batch_metrics)
            
            # 进度条显示
            batch_miou = np.mean([m['mIoU_exclude_bg'] for m in batch_metrics]) if batch_metrics else 0
            
            # 构建进度条信息
            postfix_dict = {
                'loss': f"{loss.item():.4f}",
                'mIoU': f"{batch_miou:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.6f}"
            }
            
            # 显示动态权重信息
            if self.use_dynamic_weights and 'avg_disease_weight' in loss_dict:
                postfix_dict['dyn_w'] = f"{loss_dict['avg_disease_weight']:.2f}"
            
            # 显示小目标信息
            if small_targets_info:
                postfix_dict['small'] = f"{len(small_targets_info)}"
            
            # 显示边界损失信息
            if self.use_boundary_branch and 'boundary' in loss_dict:
                postfix_dict['bound'] = f"{loss_dict['boundary']:.4f}"
            
            pbar.set_postfix(postfix_dict)
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        
        # 计算平均指标
        avg_metrics = {}
        if all_metrics:
            all_keys = set()
            for m in all_metrics:
                all_keys.update(m.keys())
            
            for key in all_keys:
                values = []
                for m in all_metrics:
                    if key in m:
                        val = m[key]
                        if isinstance(val, (int, float)):
                            values.append(float(val))
                        elif isinstance(val, (np.integer, np.floating)):
                            values.append(float(val))
                        elif isinstance(val, np.ndarray) and val.ndim == 0:
                            values.append(float(val))
                
                if values:
                    avg_metrics[key] = np.mean(values)
        
        return avg_loss, avg_metrics, sample_batches, sample_texts
    
    def validate(self, collect_samples=False):
        """验证"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        all_metrics = []
        sample_batches = []
        sample_texts = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            
            for batch_idx, batch in enumerate(pbar):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                text_descriptions = batch['text_prompt']
                
                # 安全检查
                if masks.max() >= self.num_classes or masks.min() < 0:
                    masks = torch.clamp(masks, 0, self.num_classes - 1)
                
                if masks.dtype != torch.long:
                    masks = masks.long()
                
                # 统计小目标
                small_targets_info = []
                for c in range(2, self.num_classes):
                    area = (masks == c).float().sum().item()
                    if 0 < area < 100:
                        small_targets_info.append((c, area))
                
                # 前向传播
                outputs = self.model(images, text_descriptions, return_boundary=self.use_boundary_branch)
                
                if outputs['main'].shape[1] != self.num_classes:
                    print(f" Error: main output channels {outputs['main'].shape[1]} != {self.num_classes}")
                    continue
                
                # 计算损失
                loss, loss_dict = self.criterion(outputs, masks)
                
                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)
                
                # 收集样本
                if collect_samples and batch_idx == 0:
                    pred_labels = torch.argmax(outputs['main'], dim=1)
                    sample_batches.append({
                        'images': images[:4].cpu(),
                        'masks': masks[:4].cpu(),
                        'preds': pred_labels[:4].cpu(),
                        'boundaries': outputs.get('boundary_maps', [])[-1][:4].cpu() if outputs.get('boundary_maps') else None
                    })
                    sample_texts.extend(text_descriptions[:4])
                
                # 计算指标
                pred_np = torch.argmax(outputs['main'], dim=1).cpu().numpy()
                target_np = masks.cpu().numpy()
                
                batch_metrics = []
                for i in range(images.size(0)):
                    metrics = self.metrics_calc.calculate_metrics(pred_np[i], target_np[i])
                    batch_metrics.append(metrics)
                
                all_metrics.extend(batch_metrics)
                
                # 进度条显示
                batch_miou = np.mean([m['mIoU_exclude_bg'] for m in batch_metrics]) if batch_metrics else 0
                
                postfix_dict = {
                    'loss': f"{loss.item():.4f}",
                    'mIoU': f"{batch_miou:.4f}"
                }
                
                if self.use_dynamic_weights and 'avg_disease_weight' in loss_dict:
                    postfix_dict['dyn_w'] = f"{loss_dict['avg_disease_weight']:.2f}"
                
                if small_targets_info:
                    postfix_dict['small'] = f"{len(small_targets_info)}"
                
                pbar.set_postfix(postfix_dict)
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        
        # 计算平均指标
        avg_metrics = {}
        if all_metrics:
            all_keys = set()
            for m in all_metrics:
                all_keys.update(m.keys())
            
            for key in all_keys:
                values = []
                for m in all_metrics:
                    if key in m:
                        val = m[key]
                        if isinstance(val, (int, float)):
                            values.append(float(val))
                        elif isinstance(val, (np.integer, np.floating)):
                            values.append(float(val))
                        elif isinstance(val, np.ndarray) and val.ndim == 0:
                            values.append(float(val))
                
                if values:
                    avg_metrics[key] = np.mean(values)
        
        return avg_loss, avg_metrics, sample_batches, sample_texts
    
    def visualize_results(self, sample_batches, epoch, mode='train', text_descriptions=None):
        """可视化结果"""
        if not sample_batches:
            return
        
        save_dir = os.path.join(self.result_dir, 'visualizations', mode)
        os.makedirs(save_dir, exist_ok=True)
        
        for batch_idx, batch_data in enumerate(sample_batches):
            images = batch_data['images']
            masks = batch_data['masks']
            preds = batch_data['preds']
            boundaries = batch_data.get('boundaries', None)
            
            num_samples = min(4, images.shape[0])
            batch_texts = []
            if text_descriptions is not None and batch_idx == 0:
                batch_texts = text_descriptions[:num_samples]
            
            fig, axes = plt.subplots(num_samples, 5, figsize=(25, 5*num_samples))
            
            if num_samples == 1:
                axes = axes.reshape(1, -1)
            
            for i in range(num_samples):
                img = self.vis_tool.denormalize_image(images[i])[0]
                img_np = img.permute(1, 2, 0).numpy()
                
                axes[i, 0].imshow(img_np)
                title = f'{mode} Image {i+1}'
                if i < len(batch_texts):
                    short_text = batch_texts[i][:40] + ('...' if len(batch_texts[i]) > 40 else '')
                    title += f'\nText: {short_text}'
                axes[i, 0].set_title(title, fontsize=9)
                axes[i, 0].axis('off')
                
                gt_mask = masks[i].numpy()
                gt_vis = self.vis_tool.create_mask_visualization(gt_mask, num_classes=self.num_classes)
                axes[i, 1].imshow(gt_vis)
                axes[i, 1].set_title('Ground Truth', fontsize=10)
                axes[i, 1].axis('off')
                
                pred_mask = preds[i].numpy()
                pred_vis = self.vis_tool.create_mask_visualization(pred_mask, num_classes=self.num_classes)
                axes[i, 2].imshow(pred_vis)
                axes[i, 2].set_title('Prediction', fontsize=10)
                axes[i, 2].axis('off')
                
                diff_map = self.vis_tool.create_difference_map(pred_mask, gt_mask, num_classes=self.num_classes)
                axes[i, 3].imshow(diff_map)
                axes[i, 3].set_title('Difference Map', fontsize=8)
                axes[i, 3].axis('off')
                
                if boundaries is not None and i < boundaries.shape[0]:
                    boundary_map = boundaries[i].numpy()
                    # 处理边界图
                    if boundary_map.ndim == 3 and boundary_map.shape[0] == 1:
                        boundary_map = boundary_map[0]
                    elif boundary_map.ndim == 4:
                        boundary_map = boundary_map.squeeze()
                    
                    if boundary_map.ndim == 2:
                        boundary_vis = np.stack([boundary_map] * 3, axis=-1)
                        boundary_vis = (boundary_vis * 255).astype(np.uint8)
                    else:
                        h, w = diff_map.shape[:2]
                        boundary_vis = np.zeros((h, w, 3), dtype=np.uint8)
                    
                    axes[i, 4].imshow(boundary_vis)
                    axes[i, 4].set_title('Boundary', fontsize=10)
                    axes[i, 4].axis('off')
                else:
                    h, w = diff_map.shape[:2]
                    boundary_vis = np.zeros((h, w, 3), dtype=np.uint8)
                    axes[i, 4].imshow(boundary_vis)
                    axes[i, 4].set_title('No Boundary', fontsize=10)
                    axes[i, 4].axis('off')
            
            plt.suptitle(f'{mode} Epoch {epoch+1} - Dynamic Weights (K={self.dynamic_k})', fontsize=14, y=1.02)
            plt.tight_layout()
            save_path = os.path.join(save_dir, f'epoch_{epoch+1:03d}_{mode}_batch{batch_idx}.png')
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
            plt.close()
        
        print(f" {mode} visualizations saved to: {save_dir}")
    
    def _save_best_model(self):
        """保存最佳模型"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'best_miou_exclude_bg': self.best_miou_exclude_bg,
            'best_disease_miou': self.best_disease_miou,
            'best_epoch': self.best_epoch,
            'history': self.history,
            'config': {
                'data_root': self.data_root,
                'image_size': self.image_size,
                'batch_size': self.batch_size,
                'lr': self.lr,
                'epochs': self.epochs,
                'model_size': self.model_size,
                'exp_name': self.exp_name,
                'local_model_path': self.local_model_path,
                'use_boundary_branch': self.use_boundary_branch,
                'boundary_weight': self.boundary_weight,
                'deep_supervision': self.deep_supervision,
                'use_dynamic_weights': self.use_dynamic_weights,
                'dynamic_k': self.dynamic_k,
                'ce_weight': self.ce_weight,
                'dice_weight': self.dice_weight,
                'dice_weight_factor': self.dice_weight_factor,
                'num_classes': self.num_classes
            }
        }
        
        # 清理旧的检查点
        if os.path.exists(self.save_dir):
            for file in os.listdir(self.save_dir):
                if file.endswith('.pth') and file != 'best_model.pth':
                    try:
                        os.remove(os.path.join(self.save_dir, file))
                    except:
                        pass
        
        # 保存最佳模型
        path = os.path.join(self.save_dir, "best_model.pth")
        torch.save(checkpoint, path)
        print(f" Saved best model: best_model.pth")
        print(f"   - mIoU (exclude background): {self.best_miou_exclude_bg:.4f}")
        print(f"   - Disease mIoU: {self.best_disease_miou:.4f}")
        print(f"   - At epoch: {self.best_epoch}")
        
        return path
    
    def _save_results(self):
        """保存训练结果"""
        if not self.history:
            return
        
        df = pd.DataFrame(self.history)
        csv_path = os.path.join(self.result_dir, 'training_metrics.csv')
        df.to_csv(csv_path, index=False)
        print(f" Saved metrics to: {csv_path}")
    
    def _plot_history(self):
        """绘制训练历史"""
        if not self.history:
            return
        
        epochs = [h['epoch'] for h in self.history]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 损失曲线
        train_loss = [h['train_loss'] for h in self.history]
        val_loss = [h['val_loss'] for h in self.history]
        axes[0,0].plot(epochs, train_loss, 'b-', label='Train', linewidth=2)
        axes[0,0].plot(epochs, val_loss, 'r-', label='Validation', linewidth=2)
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].set_title('Training and Validation Loss')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # mIoU曲线（排除背景）
        val_miou = [h['val_mIoU_exclude_bg'] for h in self.history]
        axes[0,1].plot(epochs, val_miou, 'g-', label='mIoU (excl. bg)', linewidth=2)
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('mIoU')
        axes[0,1].set_title('Validation mIoU (Exclude Background)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 病斑mIoU
        disease_miou = [h.get('val_disease_mIoU', 0) for h in self.history]
        axes[1,0].plot(epochs, disease_miou, 'r-', label='Disease mIoU', linewidth=2)
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('IoU')
        axes[1,0].set_title('Disease mIoU (Small Target Focus)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 动态权重信息
        if 'val_avg_disease_weight' in self.history[0]:
            disease_weight = [h.get('val_avg_disease_weight', 0) for h in self.history]
            axes[1,1].plot(epochs, disease_weight, 'orange', linewidth=2)
            axes[1,1].set_xlabel('Epoch')
            axes[1,1].set_ylabel('Weight')
            axes[1,1].set_title('Average Disease Weight (Dynamic)')
            axes[1,1].grid(True, alpha=0.3)
        else:
            # 学习率曲线
            lrs = [h.get('lr', self.lr) for h in self.history]
            axes[1,1].plot(epochs, lrs, 'orange', linewidth=2)
            axes[1,1].set_xlabel('Epoch')
            axes[1,1].set_ylabel('Learning Rate')
            axes[1,1].set_title('Learning Rate Schedule')
            axes[1,1].grid(True, alpha=0.3)
        
        # 标记最佳epoch
        if self.best_epoch > 0:
            axes[0,1].axvline(x=self.best_epoch, color='gold', linestyle='--', alpha=0.5)
            axes[0,1].text(self.best_epoch, max(val_miou)*0.9, f'Best\nepoch {self.best_epoch}', 
                         ha='center', fontsize=8, color='gold')
        
        plt.tight_layout()
        history_path = os.path.join(self.result_dir, 'training_history.png')
        plt.savefig(history_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f" Saved training history plot: {history_path}")
    
    def train(self):
        print("\n" + "="*80)
        print("Starting Training with Corrected Dynamic Weights")
        print("="*80)
        print(f" Small Target Enhancement: K={self.dynamic_k}")
        print(f" Loss weights: CE={self.ce_weight}, Dice={self.dice_weight}")
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            self.epoch = epoch
            
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.epochs}")
            print(f"{'='*60}")
            
            # 决定是否收集样本用于可视化
            collect_vis_samples = (epoch % self.visualize_freq == 0) or (epoch == self.epochs - 1)
            
            # 训练一个epoch
            train_start = time.time()
            train_loss, train_metrics, train_samples, train_texts = self.train_epoch(collect_samples=collect_vis_samples)
            train_time = time.time() - train_start
            
            # 验证
            val_start = time.time()
            val_loss, val_metrics, val_samples, val_texts = self.validate(collect_samples=collect_vis_samples)
            val_time = time.time() - val_start
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录结果
            record = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_mIoU_exclude_bg': train_metrics.get('mIoU_exclude_bg', 0),
                'val_mIoU_exclude_bg': val_metrics.get('mIoU_exclude_bg', 0),
                'val_disease_mIoU': val_metrics.get('disease_mIoU', 0),
                'val_leaf_iou': val_metrics.get('leaf_iou', 0),
                'lr': self.scheduler.get_last_lr()[0],
                'train_time': train_time,
                'val_time': val_time
            }
            
            # 添加更多指标
            for key in ['mPA_exclude_bg', 'mPrecision_exclude_bg', 'mRecall_exclude_bg', 'mDice_exclude_bg', 'mF1_exclude_bg']:
                if key in val_metrics:
                    record[f'val_{key}'] = val_metrics[key]
            
            # 统计小目标数量
            if 'present_classes_count' in val_metrics:
                record['val_present_classes_count'] = val_metrics['present_classes_count']
            
            self.history.append(record)
            
            # 检查是否是最佳模型
            current_miou = val_metrics.get('mIoU_exclude_bg', 0)
            current_disease_miou = val_metrics.get('disease_mIoU', 0)
            
            is_best = False
            if current_miou > self.best_miou_exclude_bg:
                self.best_miou_exclude_bg = current_miou
                self.best_disease_miou = current_disease_miou
                self.best_epoch = epoch + 1
                is_best = True
                print(f" New best model! mIoU: {current_miou:.4f} (epoch {self.best_epoch})")
                self._save_best_model()
            elif current_miou == self.best_miou_exclude_bg and current_disease_miou > self.best_disease_miou:
                self.best_miou_exclude_bg = current_miou
                self.best_disease_miou = current_disease_miou
                self.best_epoch = epoch + 1
                is_best = True
                print(f" New best model! (better disease mIoU)")
                self._save_best_model()
            
            # 可视化
            if collect_vis_samples:
                if train_samples and train_texts:
                    self.visualize_results(train_samples, epoch, mode='train', text_descriptions=train_texts)
                if val_samples and val_texts:
                    self.visualize_results(val_samples, epoch, mode='val', text_descriptions=val_texts)
            
            # 打印详细结果
            print(f"\n Epoch {epoch+1} Results:")
            print(f"  Training | Loss: {train_loss:.4f} | Time: {train_time:.1f}s")
            print(f"            mIoU (exclude background): {train_metrics.get('mIoU_exclude_bg', 0):.4f}")
            
            print(f"  Validation | Loss: {val_loss:.4f} | Time: {val_time:.1f}s")
            print(f"              mIoU (exclude background): {current_miou:.4f}")
            print(f"              Disease mIoU: {current_disease_miou:.4f}")
            print(f"              Leaf IoU: {val_metrics.get('leaf_iou', 0):.4f}")
            print(f"              Learning rate: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # 显示小目标增强效果
            print(f"  Dynamic Weights:")
            print(f"    - Enabled: {self.use_dynamic_weights}")
            if self.use_dynamic_weights:
                print(f"    - K value: {self.dynamic_k}")
                print(f"    - CE weight: {self.ce_weight}, Dice weight: {self.dice_weight}")
            
            if is_best:
                print(f" Saved as best model!")
            
            print(f"  Best mIoU: {self.best_miou_exclude_bg:.4f} (epoch {self.best_epoch})")
            
            # 保存结果
            self._save_results()
            
            # 定期绘制图表
            if (epoch + 1) % 10 == 0 or epoch == self.epochs - 1:
                self._plot_history()
            
            # 打印进度
            progress = (epoch + 1) / self.epochs * 100
            elapsed_time = time.time() - start_time
            avg_time_per_epoch = elapsed_time / (epoch + 1)
            remaining_time = avg_time_per_epoch * (self.epochs - epoch - 1)
            
            print(f"\n Progress: {progress:.1f}% | Elapsed: {elapsed_time/60:.1f} minutes")
            print(f" Estimated remaining: {remaining_time/60:.1f} minutes")
        
        # 最终绘制图表
        self._plot_history()
        
        # 生成最终报告
        total_time = time.time() - start_time
        print(f"\n  Total training time: {total_time/3600:.2f} hours")
        print(f"   Average per epoch: {total_time/self.epochs/60:.2f} minutes")
        
        self._save_results()
        
        print("\n" + "="*80)
        print(" Training with Corrected Dynamic Weights Completed!")
        print(f" Best model saved as: best_model.pth")
        print(f"   Best mIoU (exclude background): {self.best_miou_exclude_bg:.4f}")
        print(f"   Best Disease mIoU: {self.best_disease_miou:.4f}")
        print(f"   At epoch: {self.best_epoch}")
        print("="*80)

# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description='Train Multimodal Vision Mamba with Corrected Dynamic Weights')
    
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to dataset')
    parser.add_argument('--image_size', type=int, default=512,
                       help='Image size')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_corrected',
                       help='Directory to save checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--model_size', type=str, default='base',
                       choices=['tiny', 'small', 'base'],
                       help='Model size')
    parser.add_argument('--local_model_path', type=str, 
                       default='./models/bert-base-uncased',
                       help='Path to local BERT model')
    parser.add_argument('--visualize_freq', type=int, default=5,
                       help='Visualization frequency (every N epochs)')
    parser.add_argument('--boundary_weight', type=float, default=0.3,
                       help='Weight for boundary loss')
    parser.add_argument('--no_boundary', action='store_true',
                       help='Disable boundary enhancement')
    parser.add_argument('--no_deep_supervision', action='store_true',
                       help='Disable deep supervision')
    parser.add_argument('--no_cuda', action='store_true',
                       help='Disable CUDA')
    
    # 动态权重参数
    parser.add_argument('--use_dynamic_weights', action='store_true', default=True,
                       help='Enable corrected dynamic weights')
    parser.add_argument('--no_dynamic_weights', action='store_false', dest='use_dynamic_weights',
                       help='Disable dynamic weights')
    parser.add_argument('--dynamic_k', type=float, default=1000.0,
                       help='Dynamic enhancement coefficient K (default: 1000)')
    parser.add_argument('--ce_weight', type=float, default=0.7,
                       help='Weight for CE loss (default: 0.7)')
    parser.add_argument('--dice_weight', type=float, default=0.3,
                       help='Weight for Dice loss (default: 0.3)')
    parser.add_argument('--dice_weight_factor', type=float, default=2.0,
                       help='Dice weight enhancement factor for disease classes (default: 2.0)')
    parser.add_argument('--class_weights', type=str, default=None,
                       help='Custom class weights as comma-separated list')
    
    args = parser.parse_args()
    
    # 解析自定义类别权重
    class_weights = None
    if args.class_weights:
        try:
            class_weights = [float(w.strip()) for w in args.class_weights.split(',')]
            print(f" Using custom class weights: {class_weights}")
        except:
            print(f" Failed to parse class_weights: {args.class_weights}")
            class_weights = None
    
    print("="*80)
    print("Multimodal Vision Mamba Training with Corrected Dynamic Weights")
    print("="*80)
    
    # 创建训练器
    trainer = MultimodalVMambaTrainerCorrected(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        save_dir=args.save_dir,
        seed=args.seed,
        use_cuda=not args.no_cuda,
        model_size=args.model_size,
        local_model_path=args.local_model_path,
        visualize_freq=args.visualize_freq,
        boundary_weight=args.boundary_weight,
        use_boundary_branch=not args.no_boundary,
        deep_supervision=not args.no_deep_supervision,
        # 动态权重参数
        use_dynamic_weights=args.use_dynamic_weights,
        dynamic_k=args.dynamic_k,
        ce_weight=args.ce_weight,
        dice_weight=args.dice_weight,
        dice_weight_factor=args.dice_weight_factor,
        class_weights=class_weights
    )
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()
