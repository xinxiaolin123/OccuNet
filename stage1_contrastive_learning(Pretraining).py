import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from collections import defaultdict
import random
import glob
import cv2
import yaml
import re
import time

from ultralytics.models.yolo.model import YOLO
from ultralytics.utils import LOGGER

warnings.filterwarnings("ignore")


def parse_filename_for_pairing(filename):
    """解析文件名：原图(数字.png) vs 增强图(数字_pos_aug.png)"""
    name = Path(filename).stem
    base_match = re.search(r'^(\d+)', name)
    if not base_match:
        return None, False
    
    base_id = base_match.group(1)
    is_augmented = bool(re.search(r'_pos_aug\d*', name.lower()))
    return base_id, is_augmented


def load_simple_config(cfg_path):
    """加载简化的对比学习配置"""
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except:
        config = {}

    # 简化配置 - 稳定且有效
    simple_config = {
        'stage1_epochs': config.get('stage1_epochs', 100),               
        'simclr_temperature': config.get('simclr_temperature', 0.1),     # 适中温度
        'simclr_lr': config.get('simclr_lr', 0.0005),                    
        'simclr_weight_decay': config.get('simclr_weight_decay', 1e-4),  
        'simclr_hidden_dim': config.get('simclr_hidden_dim', 512),       # 简化维度
        'simclr_output_dim': config.get('simclr_output_dim', 256),       
        'batch_size': config.get('batch_size', 64),                      
        'batches_per_epoch': config.get('batches_per_epoch', 60),        
        'min_pairs_per_batch': config.get('min_pairs_per_batch', 6),     
        'max_pairs_per_batch': config.get('max_pairs_per_batch', 24),    
        'print_freq': config.get('print_freq', 10),                      
        'warmup_epochs': config.get('warmup_epochs', 10),                
        'cosine_annealing': config.get('cosine_annealing', True),
        'mixed_precision': config.get('mixed_precision', True),          
        'feature_extraction_layer': config.get('feature_extraction_layer', 'AIFI'),  
        'max_groups_per_batch': config.get('max_groups_per_batch', 20),   
        'samples_per_group': config.get('samples_per_group', 3),          
    }

    LOGGER.info("Simple Contrastive Learning Configuration:")
    for key, value in simple_config.items():
        LOGGER.info(f"  - {key}: {value}")
    return simple_config


class SimpleAugmentation:
    """简化的数据增强"""
    
    def __init__(self, img_size=640):
        self.img_size = img_size
        
    def augment_batch(self, images):
        """简单批量增强"""
        batch_size = images.shape[0]
        augmented = []
        
        for i in range(batch_size):
            img = images[i]
            
            # 简单的随机增强
            if random.random() > 0.5:
                img = torch.flip(img, dims=[2])  # 水平翻转
            
            if random.random() > 0.7:
                brightness_factor = 0.8 + random.random() * 0.4
                img = img * brightness_factor
                img = torch.clamp(img, 0, 1)
            
            augmented.append(img)
        
        return torch.stack(augmented)


class SimpleContrastiveLoss(nn.Module):
    """简化的对比学习损失函数"""
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, features, labels):
        batch_size = features.shape[0]
        device = features.device
        
        if batch_size < 2:
            return torch.tensor(0.0, device=device, requires_grad=True), 0
        
        # L2归一化
        features = F.normalize(features, dim=1, eps=1e-8)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 简化的标签处理
        unique_labels = list(set(labels))
        if len(unique_labels) < 2:
            return torch.tensor(0.0, device=device, requires_grad=True), 0
            
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        numeric_labels = torch.tensor([label_to_idx[label] for label in labels], device=device)
        
        # 构建正样本mask
        labels_eq = numeric_labels.unsqueeze(0) == numeric_labels.unsqueeze(1)
        
        # 排除对角线
        mask = torch.eye(batch_size, device=device).bool()
        labels_eq = labels_eq & ~mask
        
        # 计算正样本对数量
        positive_pairs = 0
        processed = set()
        
        for i in range(batch_size):
            if i in processed:
                continue
            positive_mask = labels_eq[i]
            if positive_mask.any():
                # 找到第一个正样本
                pos_indices = positive_mask.nonzero(as_tuple=True)[0]
                for pos_idx in pos_indices:
                    pos_idx_val = pos_idx.item()
                    if pos_idx_val not in processed:
                        positive_pairs += 1
                        processed.add(i)
                        processed.add(pos_idx_val)
                        break
        
        if positive_pairs == 0:
            return torch.tensor(0.0, device=device, requires_grad=True), 0
        
        # 简化的InfoNCE损失
        losses = []
        
        for i in range(batch_size):
            positive_mask = labels_eq[i]
            if not positive_mask.any():
                continue
            
            # 选择第一个正样本
            pos_indices = positive_mask.nonzero(as_tuple=True)[0]
            if len(pos_indices) == 0:
                continue
            pos_idx = pos_indices[0]
            
            # 负样本mask
            negative_mask = ~positive_mask & ~mask[i]
            if not negative_mask.any():
                continue
            
            # 构建logits
            positive_sim = similarity_matrix[i, pos_idx]
            negative_sims = similarity_matrix[i, negative_mask]
            
            logits = torch.cat([positive_sim.unsqueeze(0), negative_sims])
            target = torch.zeros(1, dtype=torch.long, device=device)
            loss = self.criterion(logits.unsqueeze(0), target)
            losses.append(loss)
        
        if losses:
            final_loss = torch.stack(losses).mean()
            return final_loss, positive_pairs
        else:
            return torch.tensor(0.0, device=device, requires_grad=True), 0


class SimpleProjectionHead(nn.Module):
    """简化的投影头"""

    def __init__(self, input_dim, hidden_dim=512, output_dim=256):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        
        projected = self.projection(x)
        return F.normalize(projected, dim=1, eps=1e-8)


class SimpleDataset:
    """简化的数据集"""

    def __init__(self, img_path, img_size=640):
        self.img_size = img_size
        self.augmentation = SimpleAugmentation(img_size)
        
        # 扫描图片
        self.img_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.img_files.extend(glob.glob(f"{img_path}/**/{ext}", recursive=True))
        
        # 分组处理
        self.original_images = defaultdict(list)
        self.augmented_images = defaultdict(list)
        
        for img_file in self.img_files:
            base_id, is_aug = parse_filename_for_pairing(Path(img_file).name)
            if base_id:
                if is_aug:
                    self.augmented_images[base_id].append(img_file)
                else:
                    self.original_images[base_id].append(img_file)
        
        # 创建配对组
        self.paired_groups = {}
        total_potential_pairs = 0
        
        for base_id in set(self.original_images.keys()) & set(self.augmented_images.keys()):
            originals = self.original_images[base_id]
            augmented = self.augmented_images[base_id]
            if originals and augmented:
                self.paired_groups[base_id] = {
                    'originals': originals,
                    'augmented': augmented
                }
                total_potential_pairs += len(originals) * len(augmented)
        
        LOGGER.info(f"Simple Dataset Statistics:")
        LOGGER.info(f"  - Total images: {len(self.img_files)}")
        LOGGER.info(f"  - Paired groups: {len(self.paired_groups)}")
        LOGGER.info(f"  - Total potential pairs: {total_potential_pairs}")

    def create_simple_batch(self, batch_size, max_groups=20):
        """创建简单batch"""
        batch_files = []
        batch_labels = []
        
        if not self.paired_groups:
            return [], []
        
        available_groups = list(self.paired_groups.keys())
        random.shuffle(available_groups)
        
        # 简单的分组策略
        target_groups = min(max_groups, len(available_groups), batch_size // 2)
        selected_groups = available_groups[:target_groups]
        
        for group_id in selected_groups:
            if len(batch_files) >= batch_size:
                break
                
            group_data = self.paired_groups[group_id]
            if not group_data['originals'] or not group_data['augmented']:
                continue
            
            # 每组添加2-3个样本
            remaining = batch_size - len(batch_files)
            max_for_group = min(3, remaining)
            
            # 添加原图
            if max_for_group > 0 and group_data['originals']:
                orig = random.choice(group_data['originals'])
                batch_files.append(orig)
                batch_labels.append(group_id)
                max_for_group -= 1
            
            # 添加增强图
            if max_for_group > 0 and group_data['augmented']:
                num_augs = min(max_for_group, len(group_data['augmented']))
                aug_samples = random.sample(group_data['augmented'], num_augs)
                
                for aug in aug_samples:
                    if len(batch_files) < batch_size:
                        batch_files.append(aug)
                        batch_labels.append(group_id)
        
        return batch_files, batch_labels

    def load_batch_images(self, batch_files):
        """加载batch图片"""
        images = []
        for img_file in batch_files:
            try:
                img = cv2.imread(img_file)
                if img is not None:
                    img = cv2.resize(img, (self.img_size, self.img_size))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                    images.append(img_tensor)
                else:
                    # 备用图片
                    img = np.random.randint(0, 255, (self.img_size, self.img_size, 3), dtype=np.uint8)
                    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                    images.append(img_tensor)
                    
            except Exception as e:
                img = np.random.randint(0, 255, (self.img_size, self.img_size, 3), dtype=np.uint8)
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                images.append(img_tensor)
        
        batch_tensor = torch.stack(images)
        augmented_batch = self.augmentation.augment_batch(batch_tensor)
        return augmented_batch

    def get_batch_statistics(self, batch_labels):
        """获取batch统计信息"""
        label_counts = defaultdict(int)
        for label in batch_labels:
            label_counts[label] += 1
        
        valid_pairs = sum(count * (count - 1) // 2 for count in label_counts.values() if count >= 2)
        return {
            'total_pairs': valid_pairs,
            'unique_groups': len(label_counts),
            'avg_group_size': sum(label_counts.values()) / len(label_counts) if label_counts else 0,
        }


class SimpleMultiScaleExtractor(nn.Module):
    """简化的特征提取器"""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.features = None
        self.hook_handle = None
        self._register_hook()
    
    def _register_hook(self):
        def hook_fn(module, input, output):
            self.features = output
        
        if len(self.model.model) > 9:
            target_layer = self.model.model[9]  # AIFI layer
            self.hook_handle = target_layer.register_forward_hook(hook_fn)
    
    def forward(self, x):
        self.features = None
        _ = self.model(x)
        return self.features
    
    def __del__(self):
        if self.hook_handle:
            self.hook_handle.remove()


class SimpleContrastiveYOLO(YOLO):
    """简化的对比学习YOLO"""

    def __init__(self, model="yolo12s.yaml", task=None, verbose=True):
        super().__init__(model, task, verbose)
        LOGGER.info("=== Simple Contrastive Learning YOLO initialized ===")

    def train_simple_contrastive_stage1(self, **kwargs):
        """简化的对比学习Stage1训练"""
        LOGGER.info("=== Starting Simple Contrastive Learning Stage1 ===")
        LOGGER.info("Strategy: Simplified and stable training")
        
        config = load_simple_config(kwargs.get('cfg', '/path/to/default.yaml'))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 输出目录
        output_dir = Path('./simple_contrastive_output2')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        LOGGER.info(f"Simple Training Configuration:")
        LOGGER.info(f"  - Device: {device}")
        LOGGER.info(f"  - Total epochs: {config['stage1_epochs']}")
        LOGGER.info(f"  - Temperature: {config['simclr_temperature']}")
        LOGGER.info(f"  - Batch size: {config['batch_size']}")
        
        # 数据集
        try:
            with open(kwargs.get('data'), 'r', encoding='utf-8') as f:
                data_config = yaml.safe_load(f)
            
            dataset = SimpleDataset(data_config['train'])
            
        except Exception as e:
            LOGGER.error(f"Failed to create dataset: {e}")
            return None
        
        # 模型设置
        model = self.model.to(device)
        model.eval()
        
        feature_extractor = SimpleMultiScaleExtractor(model).to(device)
        
        # 获取特征维度
        dummy_input = torch.randn(1, 3, 640, 640).to(device)
        with torch.no_grad():
            dummy_features = feature_extractor(dummy_input)
            if dummy_features is not None and len(dummy_features.shape) == 4:
                feature_dim = dummy_features.shape[1]
            else:
                feature_dim = 1024
        
        LOGGER.info(f"Feature dimension: {feature_dim}")
        
        # 创建组件
        projection_head = SimpleProjectionHead(
            input_dim=feature_dim,
            hidden_dim=config['simclr_hidden_dim'],
            output_dim=config['simclr_output_dim']
        ).to(device)
        
        # 优化器
        optimizer = torch.optim.AdamW(
            projection_head.parameters(),
            lr=config['simclr_lr'],
            weight_decay=config['simclr_weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器
        if config['cosine_annealing']:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config['stage1_epochs'], eta_min=config['simclr_lr'] * 0.1
            )
        else:
            scheduler = None
        
        # 损失函数
        contrastive_criterion = SimpleContrastiveLoss(config['simclr_temperature'])
        
        # 混合精度训练
        scaler = torch.cuda.amp.GradScaler() if config['mixed_precision'] else None
        
        # 训练循环
        projection_head.train()
        
        losses = []
        best_loss = float('inf')
        best_epoch = 0
        
        LOGGER.info("Starting simple training loop...")
        
        for epoch in range(config['stage1_epochs']):
            epoch_start_time = time.time()
            epoch_losses = []
            epoch_pairs = []
            
            # 学习率预热
            if epoch < config['warmup_epochs']:
                warmup_lr = config['simclr_lr'] * (epoch + 1) / config['warmup_epochs']
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            
            for batch_idx in range(config['batches_per_epoch']):
                try:
                    # 创建batch
                    batch_files, batch_labels = dataset.create_simple_batch(
                        config['batch_size'], 
                        config['max_groups_per_batch']
                    )
                    
                    if len(batch_files) < 2:
                        continue
                    
                    images = dataset.load_batch_images(batch_files).to(device)
                    batch_stats = dataset.get_batch_statistics(batch_labels)
                    
                    # 特征提取
                    with torch.no_grad():
                        features = feature_extractor(images)
                    
                    if features is None:
                        continue
                    
                    # 前向传播
                    if scaler:
                        with torch.cuda.amp.autocast():
                            projected = projection_head(features)
                            contrastive_loss, pairs = contrastive_criterion(projected, batch_labels)
                    else:
                        projected = projection_head(features)
                        contrastive_loss, pairs = contrastive_criterion(projected, batch_labels)
                    
                    # 反向传播
                    optimizer.zero_grad()
                    
                    if scaler:
                        scaler.scale(contrastive_loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(projection_head.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        contrastive_loss.backward()
                        torch.nn.utils.clip_grad_norm_(projection_head.parameters(), max_norm=1.0)
                        optimizer.step()
                    
                    epoch_losses.append(contrastive_loss.item())
                    epoch_pairs.append(pairs)
                    
                    # 进度报告
                    if batch_idx % config['print_freq'] == 0:
                        current_lr = optimizer.param_groups[0]['lr']
                        
                        LOGGER.info(f"  Epoch {epoch+1:3d} Batch {batch_idx:3d}: "
                                   f"Loss={contrastive_loss.item():.5f}, "
                                   f"Pairs={pairs:2d}, Groups={batch_stats['unique_groups']:2d}, "
                                   f"LR={current_lr:.6f}")
                
                except Exception as e:
                    LOGGER.error(f"Batch {batch_idx} failed: {e}")
                    continue
            
            # Epoch结束处理
            if scheduler and epoch >= config['warmup_epochs']:
                scheduler.step()
            
            if epoch_losses:
                avg_loss = sum(epoch_losses) / len(epoch_losses)
                avg_pairs = sum(epoch_pairs) / len(epoch_pairs) if epoch_pairs else 0
                
                losses.append(avg_loss)
                
                # 保存最佳模型
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_epoch = epoch + 1
                    
                    best_checkpoint = {
                        'epoch': best_epoch,
                        'projection_head_state_dict': projection_head.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_loss,
                        'config': config,
                        'feature_dim': feature_dim,
                    }
                    
                    torch.save(best_checkpoint, output_dir / 'best_simple_model.pt')
                
                epoch_time = time.time() - epoch_start_time
                current_lr = optimizer.param_groups[0]['lr']
                
                LOGGER.info(f"Epoch {epoch+1:3d}: Loss={avg_loss:.5f}, "
                           f"Pairs={avg_pairs:.1f}, Best={best_loss:.5f}@E{best_epoch}, "
                           f"LR={current_lr:.6f}, Time={epoch_time:.1f}s")
                
                if avg_loss == best_loss:
                    LOGGER.info(f"  *** NEW BEST: {best_loss:.5f} ***")
        
        # 保存最终权重
        final_checkpoint = {
            'projection_head_state_dict': projection_head.state_dict(),
            'config': config,
            'feature_dim': feature_dim,
            'final_loss': best_loss,
        }
        
        weights_path = output_dir / 'simple_contrastive_weights.pt'
        torch.save(final_checkpoint, weights_path)
        
        # 绘制训练曲线
        self._plot_simple_curves(losses, config, output_dir)
        
        # 清理
        if feature_extractor.hook_handle:
            feature_extractor.hook_handle.remove()
        
        LOGGER.info("=== Simple Contrastive Learning Completed ===")
        LOGGER.info(f"Weights saved to: {weights_path}")
        LOGGER.info(f"Best loss: {best_loss:.5f} @ epoch {best_epoch}")
        LOGGER.info("Strategy: Simple and stable training achieved")
        
        return {
            'losses': losses,
            'best_loss': best_loss,
            'best_epoch': best_epoch,
            'weights_path': str(weights_path),
        }

    def _plot_simple_curves(self, losses, config, output_dir):
        """绘制简单训练曲线"""
        try:
            plt.figure(figsize=(8, 6))
            
            epochs = range(1, len(losses) + 1)
            plt.plot(epochs, losses, 'b-', linewidth=2, label='Contrastive Loss')
            
            if losses:
                min_loss = min(losses)
                min_epoch = losses.index(min_loss) + 1
                plt.plot(min_epoch, min_loss, 'ro', markersize=8, label=f'Best: {min_loss:.4f}@E{min_epoch}')
            
            plt.title('Simple Contrastive Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'simple_contrastive_curves.png', dpi=200, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            LOGGER.error(f"Failed to plot curves: {e}")


def main():
    """主函数"""
    try:
        LOGGER.info("Creating Simple Contrastive Learning YOLO model...")
        LOGGER.info("Strategy: Simplified and stable approach")
        LOGGER.info("Key features:")
        LOGGER.info("- Simplified contrastive loss without complex negative mining")
        LOGGER.info("- Basic data augmentation")
        LOGGER.info("- Stable tensor operations")
        LOGGER.info("- Batch size 64 for good contrastive learning")
        LOGGER.info("- Mixed precision training for efficiency")
        
        model = SimpleContrastiveYOLO("/home/waas/simclr/Proj/ultralytics/cfg/models/12/yolo12-ours.yaml")
        
        results = model.train_simple_contrastive_stage1(
            cfg='/home/waas/simclr/Proj/ultralytics/cfg/default.yaml',
            data='/home/waas/simclr/Proj/ultralytics/cfg/datasets/Fracture.yaml',
            name='simple_contrastive_stage1'
        )
        
        if results:
            print("\n" + "="*70)
            print("Simple Contrastive Learning Completed")
            print("="*70)
            print(f"Best loss: {results['best_loss']:.5f} @ epoch {results['best_epoch']}")
            print(f"Weights: {results['weights_path']}")
            print("\nSimple Features:")
            print("- Stable and reliable training")
            print("- No complex negative mining issues")
            print("- Efficient batch processing")
            print("- Good contrastive learning with batch size 64")
        
    except Exception as e:
        LOGGER.error(f"Simple training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()