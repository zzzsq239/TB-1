import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import json

class OilTeaVOCAdaptedDataset(Dataset):
    def __init__(self, data_root, split='train', image_size=352, use_main_split=True, num_classes=9):
        """
        参数:
            data_root: 数据集根目录
            split: 数据集分割 (train/val/test)
            image_size: 图像尺寸
            use_main_split: 是否使用Main分割
            num_classes: 总类别数 = 1背景 + 1叶片 + 7病斑 = 9
        """
        self.data_root = data_root
        self.split = split
        self.num_classes = num_classes
        
        # 处理image_size参数
        if isinstance(image_size, list):
            self.image_size = tuple(image_size)
        elif isinstance(image_size, tuple):
            self.image_size = image_size
        else:
            self.image_size = (image_size, image_size)
        
        # 图像路径
        self.image_dir = os.path.join(data_root, 'JPEGImages')
        if not os.path.exists(self.image_dir):
            self.image_dir = os.path.join(data_root, 'img')
        
        # mask路径
        self.mask_dir = os.path.join(data_root, 'SegmentationClass')
        if not os.path.exists(self.mask_dir):
            self.mask_dir = os.path.join(data_root, 'labelcol')
        
        print(f"图像目录: {self.image_dir}")
        print(f"Mask目录: {self.mask_dir}")
        print(f"图像尺寸: {self.image_size}")
        print(f"类别数: {num_classes}")
        
        # 加载图像列表
        if use_main_split:
            split_file = os.path.join(data_root, 'ImageSets', 'Main', f'{split}.txt')
        else:
            split_file = os.path.join(data_root, 'ImageSets', 'Segmentation', f'{split}.txt')
        
        print(f"分割文件: {split_file}")
        
        with open(split_file, 'r') as f:
            self.image_ids = [self._normalize_image_id(line.strip()) for line in f.readlines()]
        
        # Step 1: 自动提取mask中实际出现的颜色并建立映射
        if split == 'train':
            self.color_mapping = self._build_color_mapping_from_data()
        else:
            # 验证集/测试集：从训练集保存的映射文件加载
            self.color_mapping = self._load_color_mapping()
        
        print(f"颜色映射大小: {len(self.color_mapping)} 种颜色")
        
        # 图像预处理
        self.image_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Mask预处理
        self.mask_transform = transforms.Compose([
            transforms.Resize(self.image_size,
                            interpolation=transforms.InterpolationMode.NEAREST),
        ])
        
        # 加载文本描述
        self.text_descriptions = self._load_text_descriptions_new(split)
        
        # 验证文本描述
        self.verify_text_descriptions()
        
        print(f"加载了 {len(self.image_ids)} 张图片 for {split} 分割")
        print(f"加载了 {len(self.text_descriptions)} 个文本描述")
    
    def _normalize_image_id(self, image_id):
        """标准化图像ID"""
        if image_id.startswith("IMG") and len(image_id) > 3:
            remaining = image_id[3:]
            if remaining.startswith("IMG"):
                return remaining
        return image_id
    
    def _extract_all_colors(self):
        """从所有训练集mask中提取所有颜色"""
        all_colors = Counter()
        
        print("从训练集mask中提取颜色...")
        for idx, img_id in enumerate(self.image_ids[:50]):  # 只分析前50个样本，加快速度
            mask_path = self._find_mask_path(img_id)
            if mask_path and os.path.exists(mask_path):
                try:
                    mask = Image.open(mask_path).convert('RGB')
                    mask_array = np.array(mask)
                    
                    # 提取所有唯一颜色
                    unique_colors = np.unique(mask_array.reshape(-1, 3), axis=0)
                    for color in unique_colors:
                        color_tuple = tuple(color)
                        all_colors[color_tuple] += 1
                        
                    if idx < 3:  # 显示前3个样本的信息
                        print(f"  样本 {idx}: {img_id} - {len(unique_colors)} 种颜色")
                except Exception as e:
                    print(f"处理 {img_id} 时出错: {e}")
        
        return all_colors
    
    def _build_color_mapping_from_data(self):
        """
        Step 1 & 2: 自动提取颜色并建立颜色→类别ID的映射
        
        规则:
        1. 黑色或接近黑色 → 类别0 (背景)
        2. 绿色系 → 类别1 (叶片)
        3. 其他颜色 → 类别2-8 (7种病斑)
        """
        all_colors = self._extract_all_colors()
        
        if not all_colors:
            print("警告: 没有提取到颜色，使用默认映射")
            return self._create_default_mapping()
        
        print(f"从数据中提取到 {len(all_colors)} 种颜色")
        
        # 按出现频率排序
        sorted_colors = sorted(all_colors.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n最常见的颜色 (前20个):")
        for i, (color, count) in enumerate(sorted_colors[:20]):
            r, g, b = color
            print(f"  {i+1:2d}. RGB({r:3d}, {g:3d}, {b:3d}): {count:6d}次出现")
        
        # 分类颜色
        background_colors = []
        leaf_colors = []
        disease_colors = []
        
        for color, count in sorted_colors:
            r, g, b = color
            
            # 1. 背景：黑色或接近黑色
            if self._is_background_color(color):
                background_colors.append((color, count))
            
            # 2. 叶片：绿色系
            elif self._is_leaf_color(color):
                leaf_colors.append((color, count))
            
            # 3. 其他：病斑颜色
            else:
                disease_colors.append((color, count))
        
        print(f"\n分类结果:")
        print(f"  背景颜色: {len(background_colors)} 种")
        print(f"  叶片颜色: {len(leaf_colors)} 种")
        print(f"  病斑颜色: {len(disease_colors)} 种")
        
        # 构建颜色到类别的映射
        color_mapping = {}
        
        # 1. 背景 → 类别0 (取最常见的3种背景颜色)
        for color, _ in background_colors[:3]:
            color_mapping[color] = 0
        
        # 2. 叶片 → 类别1 (取最常见的5种叶片颜色)
        for color, _ in leaf_colors[:5]:
            color_mapping[color] = 1
        
        # 3. 病斑 → 类别2-8 (7种病斑类别)
        # 将病斑颜色按颜色空间分组
        disease_groups = self._group_disease_colors(disease_colors)
        
        class_id = 2
        for group in disease_groups:
            if group:  # 只处理非空组
                for color in group:
                    color_mapping[color] = class_id
                print(f"  将 {len(group)} 种颜色映射到类别 {class_id}")
                class_id += 1
            if class_id > 8:  # 只有7种病斑类别 (2-8)
                break
        
        print(f"\n最终颜色映射统计:")
        for class_id in range(self.num_classes):
            colors_in_class = [color for color, cid in color_mapping.items() if cid == class_id]
            class_name = self._get_class_name(class_id)
            print(f"  类别 {class_id} ({class_name}): {len(colors_in_class)} 种颜色")
            if colors_in_class:
                for i, color in enumerate(colors_in_class[:3]):  # 显示前3个颜色
                    r, g, b = color
                    print(f"    {i+1}. RGB({r:3d}, {g:3d}, {b:3d})")
        
        # 保存映射到文件
        self._save_color_mapping(color_mapping)
        
        return color_mapping
    
    def _is_background_color(self, color):
        """判断是否为背景颜色（黑色或接近黑色）"""
        r, g, b = color
        
        # 纯黑色
        if r == 0 and g == 0 and b == 0:
            return True
        
        # 非常暗的颜色
        if r < 20 and g < 20 and b < 20:
            return True
        
        return False
    
    def _is_leaf_color(self, color):
        """判断是否为叶片颜色（绿色系）"""
        r, g, b = color
        
        # 排除背景
        if self._is_background_color(color):
            return False
        
        # 绿色分量显著高于红色和蓝色
        if g > r * 1.5 and g > b * 1.5:
            return True
        
        # 或者纯绿色
        if r < 100 and g > 150 and b < 100:
            return True
        
        return False
    
    def _group_disease_colors(self, disease_colors, n_groups=7):
        """
        将病斑颜色分组为7个类别
        使用简单的颜色空间聚类
        """
        if len(disease_colors) == 0:
            return [[] for _ in range(n_groups)]
        
        # 提取颜色和计数
        colors = [color for color, count in disease_colors]
        counts = [count for color, count in disease_colors]
        
        print(f"\n病斑颜色分组前:")
        for i, (color, count) in enumerate(disease_colors[:10]):  # 显示前10个
            r, g, b = color
            print(f"  {i+1}. RGB({r:3d}, {g:3d}, {b:3d}): {count}次出现")
        
        # 如果颜色太少，直接分配
        if len(colors) <= n_groups:
            print(f"病斑颜色数量 ({len(colors)}) <= 分组数 ({n_groups})，直接分配")
            groups = [[] for _ in range(n_groups)]
            for i, color in enumerate(colors):
                groups[i].append(color)
            return groups
        
        # 按颜色值排序并分组
        # 使用颜色值总和进行排序
        color_sums = [(color, sum(color)) for color in colors]
        sorted_color_sums = sorted(color_sums, key=lambda x: x[1])
        
        # 计算每个分组的大小
        group_size = len(sorted_color_sums) // n_groups
        
        groups = []
        for i in range(n_groups):
            start_idx = i * group_size
            if i == n_groups - 1:  # 最后一组包含剩余所有
                end_idx = len(sorted_color_sums)
            else:
                end_idx = (i + 1) * group_size
            
            group_colors = [color for color, _ in sorted_color_sums[start_idx:end_idx]]
            groups.append(group_colors)
        
        print(f"\n病斑颜色分组后:")
        for i, group in enumerate(groups):
            print(f"  组 {i+1} (将映射到类别 {i+2}): {len(group)} 种颜色")
            for j, color in enumerate(group[:2]):  # 显示前2个颜色
                r, g, b = color
                print(f"    {j+1}. RGB({r:3d}, {g:3d}, {b:3d})")
        
        return groups
    
    def _create_default_mapping(self):
        """创建默认的颜色映射"""
        print("使用默认颜色映射...")
        
        # 预定义一些常见的颜色
        default_mapping = {
            # 背景 (类别0)
            (0, 0, 0): 0,      # 纯黑色
            (10, 10, 10): 0,   # 深灰色
            
            # 叶片 (类别1)
            (0, 255, 0): 1,    # 纯绿色
            (34, 139, 34): 1,  # 森林绿
            (50, 205, 50): 1,  # 酸橙绿
            
            # 病斑类型1 (类别2) - 蓝色系
            (0, 0, 255): 2,    # 纯蓝色
            (0, 0, 200): 2,    # 中蓝色
            
            # 病斑类型2 (类别3) - 棕色系
            (139, 69, 19): 3,  # 鞍棕色
            (160, 82, 45): 3,  # 赭色
            
            # 病斑类型3 (类别4) - 红棕色系
            (165, 42, 42): 4,  # 棕色
            
            # 病斑类型4 (类别5) - 橙色系
            (255, 165, 0): 5,  # 橙色
            
            # 病斑类型5 (类别6) - 紫色系
            (138, 43, 226): 6, # 蓝紫色
            
            # 病斑类型6 (类别7) - 黄色系
            (255, 255, 0): 7,  # 黄色
            
            # 病斑类型7 (类别8) - 粉色系
            (255, 0, 255): 8,  # 品红色
        }
        
        return default_mapping
    
    def _save_color_mapping(self, color_mapping):
        """保存颜色映射到文件"""
        save_path = os.path.join(self.data_root, 'color_mapping.json')
        
        # 转换颜色元组为字符串以便JSON序列化
        mapping_dict = {f"{color[0]},{color[1]},{color[2]}": class_id 
                       for color, class_id in color_mapping.items()}
        
        with open(save_path, 'w') as f:
            json.dump(mapping_dict, f, indent=2)
        
        print(f"颜色映射已保存到: {save_path}")
        
        # 同时保存类别名称
        class_names = {
            '0': 'background',
            '1': 'healthy_leaf',
            '2': 'disease_type_1',
            '3': 'disease_type_2',
            '4': 'disease_type_3',
            '5': 'disease_type_4',
            '6': 'disease_type_5',
            '7': 'disease_type_6',
            '8': 'disease_type_7'
        }
        
        names_path = os.path.join(self.data_root, 'class_names.json')
        with open(names_path, 'w') as f:
            json.dump(class_names, f, indent=2)
        
        print(f"类别名称已保存到: {names_path}")
    
    def _load_color_mapping(self):
        """从文件加载颜色映射"""
        load_path = os.path.join(self.data_root, 'color_mapping.json')
        
        if os.path.exists(load_path):
            print(f"从文件加载颜色映射: {load_path}")
            with open(load_path, 'r') as f:
                mapping_dict = json.load(f)
            
            # 转换字符串键回颜色元组
            color_mapping = {}
            for color_str, class_id in mapping_dict.items():
                r, g, b = map(int, color_str.split(','))
                color_mapping[(r, g, b)] = class_id
            
            print(f"加载了 {len(color_mapping)} 种颜色的映射")
            return color_mapping
        else:
            print(f"警告: 颜色映射文件不存在: {load_path}")
            print("使用默认颜色映射")
            return self._create_default_mapping()
    
    def rgb_to_class_mask(self, rgb_tensor):
        """
        Step 3: 将颜色mask转换为类别ID mask
        
        参数:
            rgb_tensor: [3, H, W] 范围0-1
        返回:
            class_mask: [H, W] 类别ID (0-8)
        """
        # 转换为0-255整数
        rgb_np = (rgb_tensor * 255).byte().numpy()  # [3, H, W]
        rgb_np = np.transpose(rgb_np, (1, 2, 0))    # [H, W, 3]
        
        h, w = rgb_np.shape[:2]
        class_mask = np.zeros((h, w), dtype=np.int64)
        
        # 首先处理精确匹配的颜色
        processed_mask = np.zeros((h, w), dtype=bool)
        
        for color, class_id in self.color_mapping.items():
            color_array = np.array(color, dtype=np.uint8)
            mask = np.all(rgb_np == color_array, axis=-1)
            class_mask[mask] = class_id
            processed_mask[mask] = True
        
        # 处理未匹配的像素：找到最接近的颜色
        unmatched_mask = ~processed_mask
        
        if unmatched_mask.any():
            unmatched_pixels = rgb_np[unmatched_mask]
            
            # 为每个未匹配像素找到最接近的颜色
            for i in range(unmatched_pixels.shape[0]):
                pixel = unmatched_pixels[i]
                min_distance = float('inf')
                best_class_id = 0  # 默认背景
                
                for color, class_id in self.color_mapping.items():
                    distance = np.sum((pixel - np.array(color)) ** 2)
                    if distance < min_distance:
                        min_distance = distance
                        best_class_id = class_id
                
                # 应用找到的类别
                positions = np.where(unmatched_mask)
                y, x = positions[0][i], positions[1][i]
                class_mask[y, x] = best_class_id
        
        return torch.from_numpy(class_mask).long()
    
    def class_mask_to_rgb(self, class_mask):
        """
        Step 4: 将类别ID mask转换回RGB mask
        
        参数:
            class_mask: [H, W] 类别ID (0-8)
        返回:
            rgb_mask: [3, H, W] RGB颜色
        """
        if torch.is_tensor(class_mask):
            class_mask = class_mask.cpu().numpy()
        
        h, w = class_mask.shape
        rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 反向查找：类别ID → 颜色
        # 对于每个类别，找到映射中的一个颜色
        class_to_color = {}
        for color, class_id in self.color_mapping.items():
            if class_id not in class_to_color:
                class_to_color[class_id] = color
        
        # 为每个类别填充颜色
        for class_id in range(self.num_classes):
            if class_id in class_to_color:
                mask = class_mask == class_id
                color = class_to_color[class_id]
                rgb_mask[mask] = color
        
        # 转置为 [3, H, W]
        rgb_mask = np.transpose(rgb_mask, (2, 0, 1))
        
        return torch.from_numpy(rgb_mask).float() / 255.0
    
    def analyze_color_distribution(self, mask_path):
        """分析mask的颜色分布"""
        mask = Image.open(mask_path).convert('RGB')
        mask_array = np.array(mask)
        
        print(f"Mask形状: {mask_array.shape}")
        
        # 分析唯一颜色
        unique_colors, counts = np.unique(mask_array.reshape(-1, 3), axis=0, return_counts=True)
        
        print(f"唯一颜色数量: {len(unique_colors)}")
        print("\n颜色分布:")
        
        # 统计每个类别的像素数
        class_counts = defaultdict(int)
        
        for color, count in zip(unique_colors, counts):
            color_tuple = tuple(color)
            class_id = self.color_mapping.get(color_tuple)
            
            if class_id is None:
                # 找到最接近的颜色
                min_distance = float('inf')
                for map_color, map_class in self.color_mapping.items():
                    distance = np.sum((color - np.array(map_color)) ** 2)
                    if distance < min_distance:
                        min_distance = distance
                        class_id = map_class
            
            class_counts[class_id] += count
        
        total_pixels = mask_array.shape[0] * mask_array.shape[1]
        
        for class_id in sorted(class_counts.keys()):
            count = class_counts[class_id]
            percentage = count / total_pixels * 100
            class_name = self._get_class_name(class_id)
            print(f"  类别 {class_id} ({class_name}): {count}像素 ({percentage:.2f}%)")
        
        return mask_array
    
    def _get_class_name(self, class_id):
        """获取类别名称"""
        class_names = {
            0: '背景',
            1: '健康叶片',
            2: '病斑类型1',
            3: '病斑类型2',
            4: '病斑类型3',
            5: '病斑类型4',
            6: '病斑类型5',
            7: '病斑类型6',
            8: '病斑类型7'
        }
        return class_names.get(class_id, f'未知类别{class_id}')
    
    # ========== 以下方法保持不变 ==========
    
    def _load_text_descriptions_new(self, split):
        """加载新的文本描述文件"""
        text_descriptions = {}
        all_texts = {}
        
        print(f"\n=== 开始加载文本描述 for {split} split ===")
        print(f"数据集中的图片数量: {len(self.image_ids)}")
        
        try:
            all_text_file = os.path.join(self.data_root, 'ImageSets', 'Segmentation', 'Train_text.xlsx')
            print(f"加载主要文本文件: {all_text_file}")
            
            if os.path.exists(all_text_file):
                all_df = pd.read_excel(all_text_file)
                print(f"成功加载文本文件，形状: {all_df.shape}")
                print(f"列名: {all_df.columns.tolist()}")
                
                id_column = None
                for col in all_df.columns:
                    if 'image' in col.lower() or 'id' in col.lower() or col.lower() == 'image':
                        id_column = col
                        break
                
                if not id_column and len(all_df.columns) > 0:
                    id_column = all_df.columns[0]
                
                desc_column = None
                for col in all_df.columns:
                    if 'desc' in col.lower() or 'text' in col.lower() or col.lower() == 'description':
                        desc_column = col
                        break
                
                if not desc_column and len(all_df.columns) > 1:
                    desc_column = all_df.columns[1]
                
                print(f"使用ID列: '{id_column}'")
                print(f"使用描述列: '{desc_column}'")
                
                for index, row in all_df.iterrows():
                    image_id_full = str(row[id_column]).strip()
                    
                    if '.' in image_id_full:
                        image_id_base = os.path.splitext(image_id_full)[0]
                    else:
                        image_id_base = image_id_full
                    
                    image_id_base = self._normalize_image_id(image_id_base)
                    
                    if desc_column:
                        description = str(row[desc_column]).strip()
                    else:
                        description = ""
                    
                    if pd.isna(description) or description.lower() in ['nan', 'none', '']:
                        description = "oil tea leaf with disease spots"
                    
                    all_texts[image_id_base] = description
                
                print(f"从Train_text.xlsx加载了 {len(all_texts)} 个文本描述")
                
        except Exception as e:
            print(f"加载Excel文件时出错: {e}")
            import traceback
            traceback.print_exc()
        
        matched_count = 0
        not_found_count = 0
        
        for img_id in self.image_ids:
            normalized_img_id = self._normalize_image_id(img_id)
            
            if normalized_img_id in all_texts:
                text_descriptions[img_id] = all_texts[normalized_img_id]
                matched_count += 1
            else:
                found = False
                possible_ids = [
                    normalized_img_id,
                    f"IMG{normalized_img_id}" if not normalized_img_id.startswith("IMG") else normalized_img_id[3:],
                    normalized_img_id.lstrip('0'),
                    normalized_img_id.zfill(4),
                ]
                
                for test_id in possible_ids:
                    if test_id in all_texts:
                        text_descriptions[img_id] = all_texts[test_id]
                        matched_count += 1
                        found = True
                        break
                
                if not found:
                    text_descriptions[img_id] = "oil tea leaf with disease spots"
                    not_found_count += 1
        
        print(f"\n=== 匹配结果 ===")
        print(f"总图片数: {len(self.image_ids)}")
        print(f"成功匹配: {matched_count}")
        print(f"匹配失败: {not_found_count}")
        
        return text_descriptions
    
    def verify_text_descriptions(self):
        """验证文本描述"""
        print("\n=== 文本描述验证 ===")
        
        text_counts = {}
        for img_id, text in self.text_descriptions.items():
            if text in text_counts:
                text_counts[text].append(img_id)
            else:
                text_counts[text] = [img_id]
        
        duplicate_texts = {text: imgs for text, imgs in text_counts.items() if len(imgs) > 1}
        
        if duplicate_texts:
            print(f"发现 {len(duplicate_texts)} 个重复的文本描述:")
            for text, imgs in list(duplicate_texts.items())[:3]:
                print(f"  '{text[:50]}...' 出现在 {len(imgs)} 张图片")
        else:
            print("✓ 所有图片都有独特的文本描述")
        
        default_text = "oil tea leaf with disease spots"
        default_count = sum(1 for text in self.text_descriptions.values() if text == default_text)
        if default_count > 0:
            print(f"警告: 有 {default_count} 张图片使用默认文本描述")
        
        text_lengths = [len(text) for text in self.text_descriptions.values()]
        print(f"文本长度统计: 平均={np.mean(text_lengths):.1f}, "
              f"最小={min(text_lengths)}, 最大={max(text_lengths)}")
        
        unique_texts = set(self.text_descriptions.values())
        print(f"不同文本描述的数量: {len(unique_texts)}")
        
        print("=== 验证结束 ===\n")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # 加载图像
        image_path = self._find_image_path(image_id)
        if image_path is None:
            raise FileNotFoundError(f"未找到图像: {image_id}")
        
        image = Image.open(image_path).convert('RGB')
        image = self.image_transform(image)
        
        # 加载mask
        mask_path = self._find_mask_path(image_id)
        if mask_path and os.path.exists(mask_path):
            # 分析前几个样本
            if idx < 3:
                print(f"\n=== 分析mask: {image_id} ===")
                mask_array = self.analyze_color_distribution(mask_path)
            
            # 加载RGB mask
            mask_rgb = Image.open(mask_path).convert('RGB')
            mask_rgb_resized = self.mask_transform(mask_rgb)
            
            # 转换为tensor
            mask_rgb_tensor = transforms.ToTensor()(mask_rgb_resized)
            
            # 转换为类别mask
            mask = self.rgb_to_class_mask(mask_rgb_tensor)
            
            # 调试信息
            if idx < 3:
                unique_vals, counts = torch.unique(mask, return_counts=True)
                total_pixels = mask.shape[0] * mask.shape[1]
                print(f"最终类别分布:")
                for val, count in zip(unique_vals, counts):
                    class_name = self._get_class_name(val.item())
                    percentage = count.item() / total_pixels * 100
                    print(f"  类别 {val.item()} ({class_name}): {count.item()}像素 ({percentage:.2f}%)")
        
        else:
            print(f"警告: 未找到 {image_id} 的mask")
            mask = torch.zeros((self.image_size[0], self.image_size[1]), dtype=torch.long)
        
        # 获取文本描述
        text_prompt = self.text_descriptions.get(image_id, "oil tea leaf with disease spots")
        
        sample = {
            'image': image,
            'mask': mask,  # 单通道类别mask [H, W], 值范围 0-8
            'text_prompt': text_prompt,
            'image_id': image_id
        }
        
        return sample
    
    def _find_image_path(self, image_id):
        """查找图像文件路径"""
        possible_extensions = ['.jpg', '.png', '.jpeg', '.JPG', '.JPEG', '.PNG']
        possible_prefixes = ['', 'IMG']
        
        for prefix in possible_prefixes:
            for ext in possible_extensions:
                filename = f"{prefix}{image_id}{ext}"
                test_path = os.path.join(self.image_dir, filename)
                if os.path.exists(test_path):
                    return test_path
        
        return None
    
    def _find_mask_path(self, image_id):
        """查找mask文件路径"""
        possible_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
        possible_prefixes = ['', 'IMG']
        
        for prefix in possible_prefixes:
            for ext in possible_extensions:
                filename = f"{prefix}{image_id}{ext}"
                test_path = os.path.join(self.mask_dir, filename)
                if os.path.exists(test_path):
                    return test_path
        
        return None


# 测试函数
def test_color_mapping():
    """测试颜色映射功能"""
    print("测试颜色映射...")
    
    data_root = '/root/lanyun-tmp/oiltea_multimodal_segmentation/datasets/VOC2007 - 文本'
    
    # 创建训练数据集
    train_dataset = OilTeaVOCAdaptedDataset(
        data_root=data_root,
        split='train',
        image_size=512,
        num_classes=9
    )
    
    print(f"\n训练集大小: {len(train_dataset)}")
    
    # 获取一个样本
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print(f"\n样本信息:")
        print(f"  图像ID: {sample['image_id']}")
        print(f"  图像形状: {sample['image'].shape}")
        print(f"  掩码形状: {sample['mask'].shape}")
        print(f"  掩码值范围: [{sample['mask'].min()}, {sample['mask'].max()}]")
        print(f"  文本描述: {sample['text_prompt'][:50]}...")
        
        # 检查类别分布
        unique_vals, counts = torch.unique(sample['mask'], return_counts=True)
        total_pixels = sample['mask'].numel()
        
        print(f"\n类别分布:")
        for val, count in zip(unique_vals, counts):
            class_name = train_dataset._get_class_name(val.item())
            percentage = count.item() / total_pixels * 100
            print(f"  类别 {val.item()} ({class_name}): {count.item()}像素 ({percentage:.2f}%)")
        
        # 测试转换回RGB
        rgb_mask = train_dataset.class_mask_to_rgb(sample['mask'])
        print(f"\n转换回RGB的形状: {rgb_mask.shape}")
        print(f"RGB值范围: [{rgb_mask.min():.3f}, {rgb_mask.max():.3f}]")
    
    # 创建验证数据集
    print(f"\n{'='*60}")
    print("创建验证数据集...")
    
    val_dataset = OilTeaVOCAdaptedDataset(
        data_root=data_root,
        split='val',
        image_size=512,
        num_classes=9
    )
    
    print(f"验证集大小: {len(val_dataset)}")


if __name__ == "__main__":
    test_color_mapping()