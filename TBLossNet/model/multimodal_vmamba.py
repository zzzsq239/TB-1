# /root/lanyun-tmp/oiltea_multimodal_segmentation/models/multimodal_vmamba_leaf_with_boundary_corrected.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import sys
from transformers import AutoTokenizer, AutoModel, BertConfig


class BoundaryLoss(nn.Module):
    """边界损失函数"""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        
    def forward(self, pred, target):
        # 从分割标签生成边界标签
        boundary_target = self.generate_boundary(target)  # [B, 1, H, W]
        
        # 计算Dice损失
        intersection = (pred * boundary_target).sum()
        dice_loss = 1 - (2. * intersection + self.eps) / (pred.sum() + boundary_target.sum() + self.eps)
        
        return dice_loss
    
    def generate_boundary(self, mask, kernel_size=3):
        """从分割掩码生成边界标签"""
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # [B, 1, H, W]
        
        B, C, H, W = mask.shape
        
        # 使用卷积提取边界
        device = mask.device
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device) / (kernel_size * kernel_size)
        
        # 对每个类别单独处理
        boundary_maps = []
        for c in range(C):
            mask_c = mask[:, c:c+1]
            
            # 膨胀
            dilated = F.conv2d(mask_c.float(), kernel, padding=kernel_size//2)
            dilated = (dilated > 0.5).float()
            
            # 腐蚀
            eroded = F.conv2d(mask_c.float(), kernel, padding=kernel_size//2)
            eroded = (eroded > (kernel_size*kernel_size-0.5)/kernel_size**2).float()
            
            # 边界 = 膨胀 - 腐蚀
            boundary = torch.abs(dilated - eroded)
            boundary_maps.append(boundary)
        
        boundary = torch.cat(boundary_maps, dim=1)
        
        # 合并所有类别的边界
        if C > 1:
            boundary = boundary.sum(dim=1, keepdim=True).clamp(0, 1)
        
        return boundary

class CorrectDynamicWeightsLoss(nn.Module):

    def __init__(self, num_classes=9, class_weights=None, device='cpu',
                 boundary_weight=0.3, deep_supervision_weights=[0.1, 0.2, 0.3],
                 use_focal_loss=False,
                 dice_weight_factor=1.0,
                 use_dynamic_weights=True,
                 dynamic_k=1000.0,  # 动态增强系数
                 ce_weight=0.7,     # CE权重
                 dice_weight=0.3):  # Dice权重
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.boundary_weight = boundary_weight
        self.deep_supervision_weights = deep_supervision_weights
        
        # 小目标增强参数
        self.use_focal_loss = use_focal_loss
        self.dice_weight_factor = dice_weight_factor
        self.use_dynamic_weights = use_dynamic_weights
        self.dynamic_k = dynamic_k  # 动态增强系数
        

        self.base_weights = torch.tensor([
            0.1,   # 0: 背景
            1.0,   # 1: 叶片
            2.0,   # 2: 病斑1
            2.0,   # 3: 病斑2
            2.0,   # 4: 病斑3
            2.0,   # 5: 病斑4
            2.0,   # 6: 病斑5
            2.0,   # 7: 病斑6
            2.0,   # 8: 病斑7
        ]).float().to(device)
        
        if class_weights is not None:
            if isinstance(class_weights, list):
                class_weights = torch.tensor(class_weights).float()
            if len(class_weights) == num_classes:
                self.base_weights = class_weights.to(device)
        
        # 病斑类别索引
        self.disease_classes = list(range(2, num_classes))
        
        #修正3：调整损失权重比例
        self.loss_weights = {
            'ce': ce_weight,      # CE权重提高
            'dice': dice_weight,  # Dice权重降低
            'boundary': boundary_weight,
            'aux': 0.2,           # 辅助权重降低
            'deep': 0.1
        }
        
        # 动态权重参数
        self.min_area = 10      # 最小面积阈值（像素）
        self.max_area = 10000   # 最大面积阈值
        
        print(f" 修正版损失函数初始化:")
        print(f"  类别数: {num_classes}")
        print(f"  使用动态权重: {use_dynamic_weights}")
        print(f"  基础权重: {self.base_weights.cpu().numpy()}")
        print(f"  损失权重: CE={self.loss_weights['ce']}, Dice={self.loss_weights['dice']}")
        print(f"  动态增强系数 K={self.dynamic_k}")
        print(f"  Dice增强因子: {dice_weight_factor}")
        
        # 边界损失
        self.boundary_loss = BoundaryLoss()
    
    def compute_correct_dynamic_weights(self, target):
        B, H, W = target.shape
        
        # 从基础权重开始
        weights = self.base_weights.clone()
        
        # 找出实际出现的类别
        present_mask = target >= 0
        present_classes = torch.unique(target[present_mask])
        present_classes = present_classes.tolist()
        
        # 特别处理病斑类别的动态增强
        for c in present_classes:
            if c in self.disease_classes:  # 只增强病斑类别
                # 计算该类别在当前batch中的总像素数
                area = (target == c).float().sum().item()
                
                # 限制面积范围
                area = max(min(area, self.max_area), self.min_area)
                
                # 修正2：正确的动态增强公式
                # dynamic_factor = 1 + (K / (area + 1))
                # 面积越小 → factor越大 → 权重越高
                dynamic_factor = 1.0 + (self.dynamic_k / (area + 1.0))
                
                # 限制增强倍数
                dynamic_factor = min(dynamic_factor, 10.0)  # 最多增强10倍
                dynamic_factor = max(dynamic_factor, 1.0)   # 最少保持原权重
                
                # 应用动态增强
                weights[c] = self.base_weights[c] * dynamic_factor
                
                # 调试输出（只在小目标时输出）
                if area < 100:  # 小目标
                    print(f"  Disease {c}: area={area:.1f}, factor={dynamic_factor:.2f}, "
                          f"weight={weights[c].item():.2f}")
        
        # 修正3：不进行整体归一化，避免冲淡病斑权重
        # 只对病斑类别进行内部归一化
        if len(self.disease_classes) > 0:
            disease_weights = weights[self.disease_classes]
            # 保持病斑权重的相对比例
            if disease_weights.mean() > 0:
                disease_weights = disease_weights / disease_weights.mean()
                weights[self.disease_classes] = disease_weights
        
        # 安全限制：防止权重过大或过小
        weights = torch.clamp(weights, 0.05, 10.0)
        
        # 最终安全检查
        weights = torch.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
        
        return weights
    
    def dice_loss(self, pred, target):
        """Dice损失"""
        pred_soft = F.softmax(pred, dim=1)
        loss = 0
        
        for class_idx in range(self.num_classes):
            pred_ch = pred_soft[:, class_idx]
            target_ch = (target == class_idx).float()
            
            intersection = (pred_ch * target_ch).sum()
            dice = (2. * intersection + 1e-6) / (pred_ch.sum() + target_ch.sum() + 1e-6)
            
            # 病斑类别使用更高的Dice权重
            dice_weight = self.dice_weight_factor if class_idx in self.disease_classes else 1.0
            loss += dice_weight * (1 - dice)
        
        return loss / self.num_classes
    
    def focal_loss(self, pred, target, gamma=2, alpha=0.75):
        """Focal Loss"""
        ce = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce)
        loss = alpha * (1 - pt) ** gamma * ce
        return loss.mean()
    
    def forward(self, preds_dict, target):
        """前向传播"""
        # 创建干净的target
        if target.max() >= self.num_classes or target.min() < 0:
            clean_target = torch.clamp(target, 0, self.num_classes - 1)
        else:
            clean_target = target
        
        if clean_target.dtype != torch.long:
            clean_target = clean_target.long()
        
        # 获取主分割预测
        main_pred = preds_dict['main']
        
        # 计算损失
        if self.use_dynamic_weights:
            # 使用动态权重
            dynamic_weights = self.compute_correct_dynamic_weights(clean_target)
            ce_loss_fn = nn.CrossEntropyLoss(weight=dynamic_weights)
            ce_loss = ce_loss_fn(main_pred, clean_target)
            
            # 记录动态权重信息
            disease_weights = dynamic_weights[self.disease_classes]
            avg_disease_weight = disease_weights.mean().item()
            max_disease_weight = disease_weights.max().item()
            min_disease_weight = disease_weights.min().item()
            
        elif self.use_focal_loss:
            # 使用Focal Loss
            ce_loss = self.focal_loss(main_pred, clean_target)
        else:
            # 使用静态权重
            ce_loss_fn = nn.CrossEntropyLoss(weight=self.base_weights)
            ce_loss = ce_loss_fn(main_pred, clean_target)
        
        # Dice损失
        dice_loss = self.dice_loss(main_pred, clean_target)
        
        # 主损失（使用修正后的权重比例）
        main_loss = (
            self.loss_weights['ce'] * ce_loss + 
            self.loss_weights['dice'] * dice_loss
        )
        
        # 辅助损失（权重降低）
        aux_loss = 0
        aux_preds = preds_dict.get('aux', [])
        if aux_preds and len(aux_preds) > 0:
            for aux in aux_preds:
                if aux is not None:
                    # 上采样
                    if aux.shape[2:] != clean_target.shape[1:]:
                        aux = F.interpolate(aux, size=clean_target.shape[1:], 
                                           mode='bilinear', align_corners=True)
                    
                    # 辅助输出使用静态权重
                    aux_loss_fn = nn.CrossEntropyLoss(weight=self.base_weights)
                    aux_loss += aux_loss_fn(aux, clean_target)
            
            if len(aux_preds) > 0:
                aux_loss = aux_loss / len(aux_preds)
                main_loss = main_loss + self.loss_weights['aux'] * aux_loss
        
        # 边界损失
        boundary_loss = 0
        boundary_maps = preds_dict.get('boundary_maps', [])
        if boundary_maps and len(boundary_maps) > 0:
            # 使用最后一层的边界图
            boundary_pred = boundary_maps[-1]
            if boundary_pred.shape[2:] != clean_target.shape[1:]:
                boundary_pred = F.interpolate(boundary_pred, size=clean_target.shape[1:],
                                            mode='bilinear', align_corners=True)
            boundary_loss = self.boundary_loss(boundary_pred, clean_target)
            main_loss = main_loss + self.loss_weights['boundary'] * boundary_loss
        
        # 深度监督损失
        deep_loss = 0
        deep_features = preds_dict.get('deep_features', [])
        if deep_features and len(deep_features) > 0:
            for i, deep_pred in enumerate(deep_features):
                weight = self.deep_supervision_weights[i] if i < len(self.deep_supervision_weights) else 0.1
                
                if deep_pred.shape[2:] != clean_target.shape[1:]:
                    deep_pred = F.interpolate(deep_pred, size=clean_target.shape[1:],
                                            mode='bilinear', align_corners=True)
                
                # 深度监督使用静态权重
                deep_loss_fn = nn.CrossEntropyLoss(weight=self.base_weights)
                deep_loss += weight * deep_loss_fn(deep_pred, clean_target)
            
            if len(deep_features) > 0:
                deep_loss = deep_loss / len(deep_features)
                main_loss = main_loss + self.loss_weights['deep'] * deep_loss
        
        total_loss = main_loss
        
        # 损失详情
        loss_dict = {
            'total': total_loss.item(),
            'main': main_loss.item(),
            'ce': ce_loss.item(),
            'dice': dice_loss.item(),
            'aux': aux_loss if isinstance(aux_loss, (int, float)) else aux_loss.item(),
            'boundary': boundary_loss.item() if isinstance(boundary_loss, torch.Tensor) else boundary_loss,
            'deep': deep_loss.item() if isinstance(deep_loss, torch.Tensor) else deep_loss,
        }
        
        # 添加动态权重信息
        if self.use_dynamic_weights:
            loss_dict['avg_disease_weight'] = avg_disease_weight
            loss_dict['max_disease_weight'] = max_disease_weight
            loss_dict['min_disease_weight'] = min_disease_weight
            
            # 统计小目标数量
            small_target_count = 0
            for c in self.disease_classes:
                area = (clean_target == c).float().sum().item()
                if 0 < area < 100:  # 小于100像素
                    small_target_count += 1
            
            loss_dict['small_targets'] = small_target_count
        
        return total_loss, loss_dict


class EdgeDetectionBranch(nn.Module):
    """边界检测分支"""
    def __init__(self, in_channels, upsample_factor=4):
        super().__init__()
        
        # 简单的边缘检测头
        self.conv1 = nn.Conv2d(in_channels, max(32, in_channels // 2), 1)
        self.gn1 = nn.GroupNorm(8, max(32, in_channels // 2))
        self.conv2 = nn.Conv2d(max(32, in_channels // 2), 1, 1)
        
        self.upsample_factor = upsample_factor
        
        print(f"初始化边界检测分支: in_channels={in_channels}")
    
    def forward(self, x, target_size=None):

        # 特征处理
        x = F.relu(self.gn1(self.conv1(x)))
        edge_map = torch.sigmoid(self.conv2(x))  # [B, 1, H, W]
        
        # 上采样到目标尺寸
        if target_size is not None:
            edge_map = F.interpolate(edge_map, size=target_size, 
                                    mode='bilinear', align_corners=True)
        elif self.upsample_factor > 1:
            B, C, H, W = edge_map.shape
            edge_map = F.interpolate(edge_map, 
                                    size=(H * self.upsample_factor, W * self.upsample_factor),
                                    mode='bilinear', align_corners=True)
        
        return edge_map


class TextEncoderLocal(nn.Module):
    def __init__(self, local_model_path, text_dim=768, proj_dim=256):
        super().__init__()
        
        # 检查本地模型路径
        self.local_model_path = local_model_path
        
        if not os.path.exists(local_model_path):
            print(f"本地BERT模型路径不存在: {local_model_path}")
            print("请确保有以下文件:")
            print("  - config.json")
            print("  - pytorch_model.bin (或 model.safetensors)")
            print("  - vocab.txt")
            print("  - tokenizer_config.json")
            sys.exit(1)
        
        print(f"使用本地BERT模型: {local_model_path}")
        
        try:
            # 从本地加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                local_model_path, 
                local_files_only=True
            )
            
            # 从本地加载模型
            self.text_model = AutoModel.from_pretrained(
                local_model_path,
                local_files_only=True
            )
            
            print("成功从本地加载BERT模型和tokenizer")
            
        except Exception as e:
            print(f"加载本地BERT模型失败: {e}")
            print("请检查文件是否存在且格式正确")
            sys.exit(1)
        
        # 冻结BERT的大部分参数（可选）
        for param in self.text_model.parameters():
            param.requires_grad = False
        
        # 投影层：将文本特征映射到视觉特征空间
        self.projection = nn.Sequential(
            nn.Linear(text_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim)
        )
        
        self.text_dim = text_dim
        self.proj_dim = proj_dim
        
        print(f"文本编码器初始化完成: 输入维度={text_dim}, 投影维度={proj_dim}")
    
    def forward(self, text_list):
        
        try:
            # 编码文本
            inputs = self.tokenizer(
                text_list, 
                padding=True, 
                truncation=True, 
                max_length=128,
                return_tensors="pt"
            ).to(next(self.text_model.parameters()).device)
            
            with torch.no_grad():
                text_outputs = self.text_model(**inputs)
            
            # 使用[CLS] token的特征
            cls_features = text_outputs.last_hidden_state[:, 0, :]
            
            # 投影到视觉特征空间
            text_features = self.projection(cls_features)
            
            return text_features
            
        except Exception as e:
            print(f"文本编码失败: {e}")
            # 返回零特征作为备选
            batch_size = len(text_list)
            return torch.zeros(batch_size, self.proj_dim, 
                             device=next(self.text_model.parameters()).device)


class MultimodalFusion(nn.Module):
    def __init__(self, visual_dim, text_dim, fusion_dim=256):
        super().__init__()
        
        # 视觉特征处理
        self.visual_proj = nn.Sequential(
            nn.Conv2d(visual_dim, fusion_dim, 1),
            nn.GroupNorm(32, fusion_dim),
            nn.GELU()
        )
        
        # 文本特征处理（扩展到空间维度）
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU()
        )
        
        # 简单的注意力融合
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(fusion_dim * 2, fusion_dim, 3, padding=1),
            nn.GroupNorm(32, fusion_dim),
            nn.GELU(),
            nn.Conv2d(fusion_dim, fusion_dim, 3, padding=1),
            nn.GroupNorm(32, fusion_dim),
            nn.GELU()
        )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Conv2d(fusion_dim, visual_dim, 1),
            nn.GroupNorm(32, visual_dim),
            nn.GELU()
        )
        
        self.fusion_dim = fusion_dim
        
        print(f"初始化融合模块: visual_dim={visual_dim}, text_dim={text_dim}, fusion_dim={fusion_dim}")
    
    def forward(self, visual_features, text_features):
        
        B, C, H, W = visual_features.shape
        
        # 投影视觉特征
        visual_proj = self.visual_proj(visual_features)  # [B, fusion_dim, H, W]
        
        # 投影文本特征并扩展到空间维度
        text_proj = self.text_proj(text_features)  # [B, fusion_dim]
        text_spatial = text_proj.unsqueeze(-1).unsqueeze(-1)  # [B, fusion_dim, 1, 1]
        text_spatial = text_spatial.expand(-1, -1, H, W)  # [B, fusion_dim, H, W]
        
        # 拼接特征
        concat_features = torch.cat([visual_proj, text_spatial], dim=1)
        
        # 卷积融合
        fused = self.conv_fusion(concat_features)
        
        # 输出投影
        output = self.output_proj(fused)
        
        # 残差连接
        output = output + visual_features
        
        return output


class MultimodalVMambaEncoderWithBoundary(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, depths=[2,2,9,2], 
                 d_state=16, text_dim=256, drop_path_rate=0.1,
                 local_model_path='./models/bert-base-uncased',
                 use_boundary_branch=True, deep_supervision_levels=[0, 1, 2, 3, 4],
                 num_classes=9):
        super().__init__()
        
        from models.real_vision_mamba import VMambaEncoder
        
        # 单模态视觉编码器
        self.visual_encoder = VMambaEncoder(
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            d_state=d_state,
            drop_path_rate=drop_path_rate
        )
        
        # 文本编码器（使用本地文件）
        self.text_encoder = TextEncoderLocal(
            local_model_path=local_model_path,
            proj_dim=text_dim
        )
        
        # 获取特征通道数
        self.feat_channels = self.visual_encoder.feat_channels
        
        print(f"视觉编码器特征通道: {self.feat_channels}")
        
        # 多模态融合模块
        self.fusion_layers = nn.ModuleList([
            MultimodalFusion(self.feat_channels[0], text_dim),  # stage 1
            MultimodalFusion(self.feat_channels[1], text_dim),  # stage 1 output
            MultimodalFusion(self.feat_channels[2], text_dim),  # stage 2
            MultimodalFusion(self.feat_channels[3], text_dim),  # stage 3
            MultimodalFusion(self.feat_channels[4], text_dim)   # stage 4
        ])
        
        # 边界检测分支
        self.use_boundary_branch = use_boundary_branch
        if use_boundary_branch:
            self.boundary_branches = nn.ModuleList([
                EdgeDetectionBranch(self.feat_channels[i]) 
                for i in range(len(self.feat_channels))
            ])
            print(f"添加边界检测分支: {len(self.boundary_branches)}个层级")
        
        # 深度监督层级
        self.deep_supervision_levels = deep_supervision_levels
        self.num_classes = num_classes
        self.deep_supervision_heads = nn.ModuleList()
        
        for level in deep_supervision_levels:
            if level < len(self.feat_channels):
                head = nn.Sequential(
                    nn.Conv2d(self.feat_channels[level], max(64, self.feat_channels[level] // 4), 1),
                    nn.GroupNorm(8, max(64, self.feat_channels[level] // 4)),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(max(64, self.feat_channels[level] // 4), num_classes, 1)
                )
                self.deep_supervision_heads.append(head)
        
        print(f"深度监督层级: {deep_supervision_levels}")
        print(f"多模态编码器初始化完成（带边界增强），{num_classes}个类别")
    
    def forward(self, x, text_descriptions, return_boundary=True, return_deep_features=False):
        # 编码文本
        text_features = self.text_encoder(text_descriptions)  # [B, text_dim]
        
        # 获取视觉特征
        feats = self.visual_encoder.forward_features(x)
        
        # 检查特征数量是否匹配
        if len(feats) != len(self.fusion_layers):
            print(f" 特征数量不匹配: feats={len(feats)}, fusion_layers={len(self.fusion_layers)}")
            print(f"  只处理前{min(len(feats), len(self.fusion_layers))}个特征")
        
        # 在多个层级融合文本信息
        fused_feats = []
        boundary_maps = []
        deep_features = []
        
        min_len = min(len(feats), len(self.fusion_layers))
        
        for i in range(min_len):
            feat = feats[i]
            fusion_layer = self.fusion_layers[i]
            
            # 检查特征形状
            B, C, H, W = feat.shape
            expected_channels = self.feat_channels[i] if i < len(self.feat_channels) else C
            
            if C != expected_channels:
                print(f" 第{i+1}层特征通道不匹配: 期望{expected_channels}, 实际{C}")
                # 调整通道数
                adjust_conv = nn.Conv2d(C, expected_channels, 1).to(feat.device)
                feat = adjust_conv(feat)
            
            fused_feat = fusion_layer(feat, text_features)
            fused_feats.append(fused_feat)
            
            # 生成边界图
            if self.use_boundary_branch and return_boundary:
                boundary_map = self.boundary_branches[i](fused_feat)
                boundary_maps.append(boundary_map)
            
            # 深度监督特征
            if return_deep_features and i in self.deep_supervision_levels:
                level_idx = self.deep_supervision_levels.index(i)
                deep_feat = self.deep_supervision_heads[level_idx](fused_feat)
                deep_features.append(deep_feat)
        
        # 如果还有未处理的特征，直接添加
        if len(feats) > len(self.fusion_layers):
            for i in range(len(self.fusion_layers), len(feats)):
                fused_feats.append(feats[i])
        
        # 返回结果
        outputs = [fused_feats]
        
        if return_boundary and self.use_boundary_branch:
            outputs.append(boundary_maps)
        
        if return_deep_features:
            outputs.append(deep_features)
        
        return outputs[0] if len(outputs) == 1 else tuple(outputs)


class MultimodalVMambaSegLocalWithBoundary(nn.Module):
    def __init__(self, num_classes=9, in_chans=3, embed_dim=96,
                 depths=[2,2,9,2], d_state=16, use_aux=True,
                 text_dim=256, local_model_path='./models/bert-base-uncased',
                 use_boundary_branch=True, boundary_weight=0.3,
                 deep_supervision=True, deep_supervision_weights=[0.1, 0.2, 0.3]):
        super().__init__()
        
        self.use_aux = use_aux
        self.use_boundary_branch = use_boundary_branch
        self.boundary_weight = boundary_weight
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes
        
        # 多模态编码器（带边界分支）
        self.encoder = MultimodalVMambaEncoderWithBoundary(
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            d_state=d_state,
            text_dim=text_dim,
            local_model_path=local_model_path,
            use_boundary_branch=use_boundary_branch,
            deep_supervision_levels=[2, 3, 4] if deep_supervision else [],
            num_classes=num_classes
        )
        
        ch = self.encoder.feat_channels
        
        
        # 解码器
        from models.real_vision_mamba import DecoderBlock, SegmentationHead, AuxSegmentationHead
        
        self.decoder4 = DecoderBlock(ch[4], ch[3], embed_dim*4)
        self.decoder3 = DecoderBlock(embed_dim*4, ch[2], embed_dim*2)
        self.decoder2 = DecoderBlock(embed_dim*2, ch[1], embed_dim)
        self.decoder1 = DecoderBlock(embed_dim, ch[0], embed_dim)
        
        self.seg_head = SegmentationHead(embed_dim, num_classes)
        
        if use_aux:
            self.aux_head4 = AuxSegmentationHead(embed_dim*4, num_classes)
            self.aux_head3 = AuxSegmentationHead(embed_dim*2, num_classes)
        
        print(f" 多模态VMambaSeg模型初始化完成（带边界增强）")
        print(f"   Use auxiliary heads: {use_aux}")
        print(f"   Use boundary branch: {use_boundary_branch}")
        print(f"   Boundary weight: {boundary_weight}")
        print(f"   Deep supervision: {deep_supervision}")
        print(f"   Deep supervision weights: {deep_supervision_weights}")
        print(f"   Number of classes: {num_classes}")
    
    def forward(self, x, text_descriptions, return_boundary=True):
        input_size = x.shape[2:]
        
        # 获取多模态特征
        encoder_outputs = self.encoder(x, text_descriptions, 
                                      return_boundary=return_boundary,
                                      return_deep_features=self.deep_supervision)
        
        if self.use_boundary_branch and self.deep_supervision:
            feats, boundary_maps, deep_features = encoder_outputs
        elif self.use_boundary_branch:
            feats, boundary_maps = encoder_outputs
            deep_features = []
        elif self.deep_supervision:
            feats, deep_features = encoder_outputs
            boundary_maps = []
        else:
            feats = encoder_outputs
            boundary_maps = []
            deep_features = []
        
        # 检查获取的特征数量
        if len(feats) < 5:
            print(f" 只获取到 {len(feats)} 个特征，但需要5个")
            while len(feats) < 5:
                feats.append(feats[-1])
        
        aux = []
        
        # 解码过程
        x = self.decoder4(feats[4], feats[3])
        if self.use_aux:
            aux.append(self.aux_head4(x))
        
        x = self.decoder3(x, feats[2])
        if self.use_aux:
            aux.append(self.aux_head3(x))
        
        x = self.decoder2(x, feats[1])
        x = self.decoder1(x, feats[0])
        
        main = self.seg_head(x)
        main = F.interpolate(main, size=input_size, mode='bilinear', align_corners=True)
        
        # 上采样辅助输出
        if self.use_aux:
            for i in range(len(aux)):
                if aux[i].shape[2:] != input_size:
                    aux[i] = F.interpolate(aux[i], size=input_size,
                                          mode='bilinear', align_corners=True)
        
        # 上采样深度监督特征
        if self.deep_supervision and deep_features:
            for i in range(len(deep_features)):
                if deep_features[i].shape[2:] != input_size:
                    deep_features[i] = F.interpolate(deep_features[i], size=input_size,
                                                    mode='bilinear', align_corners=True)
        
        # 上采样边界图
        if self.use_boundary_branch and boundary_maps:
            for i in range(len(boundary_maps)):
                if boundary_maps[i].shape[2:] != input_size:
                    boundary_maps[i] = F.interpolate(boundary_maps[i], size=input_size,
                                                    mode='bilinear', align_corners=True)
        
        # 组织返回结果
        outputs = {
            'main': main,
            'aux': aux if self.use_aux else [],
            'boundary_maps': boundary_maps if self.use_boundary_branch else [],
            'deep_features': deep_features if self.deep_supervision else []
        }
        
        return outputs

def test_multimodal_vmamba_with_corrected_loss():
    print("测试带修正动态权重的多模态VMamba模型...")
    
    # 本地BERT模型路径
    local_bert_path = "./models/bert-base-uncased"
    
    # 检查文件是否存在
    required_files = ['config.json', 'vocab.txt', 'tokenizer_config.json']
    model_files = ['pytorch_model.bin', 'model.safetensors']
    
    print(f"检查本地BERT模型文件...")
    for file in required_files:
        file_path = os.path.join(local_bert_path, file)
        if os.path.exists(file_path):
            print(f"   {file}")
        else:
            print(f"   {file} 不存在")
    
    has_model = False
    for model_file in model_files:
        model_path = os.path.join(local_bert_path, model_file)
        if os.path.exists(model_path):
            has_model = True
            print(f"   {model_file}")
            break
    
    if not has_model:
        print(f"   没有找到模型文件")
        return False
    
    # 创建模型
    model = MultimodalVMambaSegLocalWithBoundary(
        num_classes=9,
        embed_dim=96,
        depths=[2,2,9,2],
        d_state=16,
        use_aux=True,
        local_model_path=local_bert_path,
        use_boundary_branch=True,
        boundary_weight=0.3,
        deep_supervision=True
    )
    
    # 测试数据
    x = torch.randn(2, 3, 512, 512)
    text_descriptions = [
        "One leaf, upper middle, white.",
        "Two leaves, lower left and upper right, white."
    ]
    
    print(f"\n测试前向传播...")
    print(f"输入图像形状: {x.shape}")
    
    # 前向传播
    try:
        outputs = model(x, text_descriptions)
        
        print(f" 主输出: {outputs['main'].shape} (应为 [2, 9, 512, 512])")
        
        if outputs['aux']:
            for i, a in enumerate(outputs['aux']):
                print(f" 辅助输出{i+1}: {a.shape}")
        
        if outputs['boundary_maps']:
            for i, b in enumerate(outputs['boundary_maps']):
                print(f" 边界图{i+1}: {b.shape}")
        
        if outputs['deep_features']:
            for i, d in enumerate(outputs['deep_features']):
                print(f" 深度监督特征{i+1}: {d.shape}")
        
       
        target = torch.randint(0, 9, (2, 512, 512))  # 0-8
        # 添加小目标
        target[0, 250:260, 250:260] = 2  # 小病斑区域
        
        # 测试不同配置
        configs = [
            ("静态权重", {'use_dynamic_weights': False, 'dynamic_k': 1000}),
            ("动态权重(K=500)", {'use_dynamic_weights': True, 'dynamic_k': 500}),
            ("动态权重(K=1000)", {'use_dynamic_weights': True, 'dynamic_k': 1000}),
            ("动态权重(K=2000)", {'use_dynamic_weights': True, 'dynamic_k': 2000}),
        ]
        
        for name, params in configs:
            print(f"\n=== {name} ===")
            criterion = CorrectDynamicWeightsLoss(
                num_classes=9,
                device='cpu',
                use_dynamic_weights=params['use_dynamic_weights'],
                dynamic_k=params['dynamic_k']
            )
            
            loss, loss_dict = criterion(outputs, target)
            print(f"  总损失: {loss_dict['total']:.4f}")
            print(f"  CE损失: {loss_dict['ce']:.4f}")
            print(f"  Dice损失: {loss_dict['dice']:.4f}")
            
            if 'avg_disease_weight' in loss_dict:
                print(f"  病斑平均权重: {loss_dict['avg_disease_weight']:.2f}")
                print(f"  病斑最大权重: {loss_dict['max_disease_weight']:.2f}")
                print(f"  病斑最小权重: {loss_dict['min_disease_weight']:.2f}")
                print(f"  小目标数量: {loss_dict.get('small_targets', 0)}")
        
        # 参数统计
        params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n 模型统计:")
        print(f"  总参数: {params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        
        return True
        
    except Exception as e:
        print(f" 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_multimodal_vmamba_with_corrected_loss()
