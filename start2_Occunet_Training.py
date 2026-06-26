import warnings
import torch
import numpy as np
import random
from ultralytics import YOLO
from pathlib import Path
import yaml

warnings.filterwarnings("ignore")

class OverfittingFixedTrainer:
    """
    修复过拟合的鲁棒性训练器
    主要修复：
    1. 大幅增强正则化防止过拟合
    2. 早停机制优化
    3. 学习率调度优化
    4. 数据增强策略增强
    5. 验证监控机制
    """
    
    def __init__(self):
        # 基础路径配置
        self.model_config = "/home/waas/simclr/Proj/ultralytics/cfg/models/12/yolo12-ours.yaml"
        self.train_config = "/home/waas/simclr/Proj/ultralytics/cfg/default.yaml"
        self.data_config = "/home/waas/simclr/Proj/ultralytics/cfg/datasets/Fracture.yaml"
        self.pretrained_weights = "/home/waas/simclr/Proj/best_pt/fold1newaug.pt"
        
        # 训练模式配置
        self.training_modes = {
            "backbone": "只训练backbone特征提取层 (0-9)",
            "neck": "只训练neck特征融合层 (10-20)", 
            "head": "只训练head检测层 (21)",
            "neck_head": "训练neck+head层 (10-21)",
            "full": "训练所有层 (0-21)"
        }
        
        # 防过拟合的鲁棒性配置 - 大幅增强正则化
        self.robustness_levels = {
            "low": {
                "weight_decay": 0.0005,     # 增加权重衰减
                "dropout": 0.15,            # 增加dropout
                "lr_factor": 0.8,           # 降低学习率
                "augment_factor": 1.2,      # 增强数据增强
                "mixup": 0.1,              # 开启mixup
                "label_smoothing": 0.1,     # 标签平滑
                "description": "轻度防过拟合正则化"
            },
            "medium": {
                "weight_decay": 0.001,      # 更强权重衰减
                "dropout": 0.2,             # 更强dropout
                "lr_factor": 0.6,           # 更低学习率
                "augment_factor": 1.5,      # 更强数据增强
                "mixup": 0.15,             # 更强mixup
                "label_smoothing": 0.15,    # 更强标签平滑
                "description": "中等防过拟合正则化"
            },
            "high": {
                "weight_decay": 0.002,      # 强权重衰减
                "dropout": 0.25,            # 强dropout
                "lr_factor": 0.4,           # 低学习率
                "augment_factor": 2.0,      # 强数据增强
                "mixup": 0.2,              # 强mixup
                "label_smoothing": 0.2,     # 强标签平滑
                "description": "强防过拟合正则化"
            },
            "extreme": {
                "weight_decay": 0.005,      # 极强权重衰减
                "dropout": 0.3,             # 极强dropout
                "lr_factor": 0.2,           # 极低学习率
                "augment_factor": 2.5,      # 极强数据增强
                "mixup": 0.25,             # 极强mixup
                "label_smoothing": 0.25,    # 极强标签平滑
                "description": "极强防过拟合正则化"
            }
        }
    
    def setup_environment(self):
        """设置环境和随机种子"""
        seed = 123
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.empty_cache()
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        print(f"Environment ready: PyTorch {torch.__version__}")
        print(f"Random seed set to: {seed}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    
    def validate_paths(self):
        """验证所有路径"""
        paths = {
            "Model config": self.model_config,
            "Train config": self.train_config,
            "Data config": self.data_config,
            "Pretrained weights": self.pretrained_weights
        }
        
        all_valid = True
        for name, path in paths.items():
            if Path(path).exists():
                print(f"✓ {name}: {path}")
            else:
                print(f"✗ {name}: Missing - {path}")
                all_valid = False
        
        return all_valid
    
    def apply_layer_training_control(self, model, training_mode):
        """应用层级训练控制"""
        print(f"\nApplying layer training control: {training_mode}")
        print(f"Description: {self.training_modes[training_mode]}")
        
        # 首先冻结所有参数
        for param in model.model.parameters():
            param.requires_grad = False
        
        trainable_count = 0
        frozen_count = 0
        trainable_modules = []
        
        # 根据YOLO12结构和训练模式解冻相应层
        for name, module in model.model.named_modules():
            should_train = False
            
            if training_mode == "backbone":
                if any(pattern in name for pattern in [
                    'model.0', 'model.1', 'model.2', 'model.3', 'model.4',
                    'model.5', 'model.6', 'model.7', 'model.8', 'model.9'
                ]):
                    should_train = True
                    
            elif training_mode == "neck":
                if any(pattern in name for pattern in [
                    'model.10', 'model.11', 'model.12', 'model.13', 'model.14',
                    'model.15', 'model.16', 'model.17', 'model.18', 'model.19', 'model.20'
                ]):
                    should_train = True
                    
            elif training_mode == "head":
                if 'model.21' in name or 'detect' in name.lower():
                    should_train = True
                    
            elif training_mode == "neck_head":
                if any(pattern in name for pattern in [
                    'model.10', 'model.11', 'model.12', 'model.13', 'model.14',
                    'model.15', 'model.16', 'model.17', 'model.18', 'model.19', 
                    'model.20', 'model.21'
                ]) or 'detect' in name.lower():
                    should_train = True
                    
            elif training_mode == "full":
                should_train = True
            
            # 应用训练控制
            if should_train:
                module_params = 0
                for param in module.parameters():
                    param.requires_grad = True
                    module_params += 1
                
                if module_params > 0:
                    trainable_count += module_params
                    trainable_modules.append(name)
            else:
                module_params = 0
                for param in module.parameters():
                    param.requires_grad = False
                    module_params += 1
                frozen_count += module_params
        
        print(f"\nLayer control results:")
        print(f"  Trainable parameters: {trainable_count:,}")
        print(f"  Frozen parameters: {frozen_count:,}")
        print(f"  Training ratio: {trainable_count/(trainable_count+frozen_count)*100:.1f}%")
        
        return trainable_count > 0
    
    def get_overfitting_resistant_config(self, training_mode, robustness_level):
        """获取防过拟合训练配置"""
        
        robust_params = self.robustness_levels[robustness_level]
        
        # 基础配置
        base_config = {
            'cfg': self.train_config,
            'data': self.data_config,
            'pretrained': self.pretrained_weights,
            
            # 基础设置
            'task': 'detect',
            'mode': 'train',
            'imgsz': 640,
            'device': 0,
            'workers': 8,
            'amp': True,
            'val': True,
            'plots': True,
            'save': True,
            'exist_ok': False,
            'project': 'runs/overfitting_fixed',
            'verbose': True,
            'deterministic': True,
            'resume': False,
            'seed': 789,
            'freeze': None,
        }
        
        # 根据训练模式调整参数 - 防过拟合优化
        if training_mode == "backbone":
            mode_config = {
                'epochs': 250,               # 减少epochs防止过拟合
                'batch': 32,                # 增大batch size
                'patience': 50,             # 更积极的早停
                'lr0': 0.00008,            # 更低学习率
            }
        elif training_mode == "neck":
            mode_config = {
                'epochs': 250,              # 减少epochs
                'batch': 64,                
                'patience': 50,             # 更积极的早停
                'lr0': 0.00008,
            }
        elif training_mode == "head":
            mode_config = {
                'epochs': 250,               # 大幅减少epochs
                'batch': 64,
                'patience': 50,             # 很积极的早停
                'lr0': 0.00015,
            }
        elif training_mode == "neck_head":
            mode_config = {
                'epochs': 200,              # 减少epochs
                'batch': 32,
                'patience': 50,             # 更积极的早停
                'lr0': 0.00008,
            }
        else:  # full
            mode_config = {
                'epochs': 250,              # 大幅减少epochs
                'batch': 32,
                'patience': 32,             # 积极早停
                'lr0': 0.00001,            # 极低学习率
            }
        
        # 防过拟合配置 - 大幅增强正则化
        anti_overfitting_config = {
            # 学习率调度 - 防过拟合优化
            'lr0': mode_config['lr0'] * robust_params['lr_factor'],
            'lrf': 0.05,                    # 更低的最终学习率比例
            'momentum': 0.9,                # 稍低momentum
            'weight_decay': robust_params['weight_decay'],  # 大幅增强权重衰减
            'warmup_epochs': max(5, mode_config['epochs'] // 15),  # 增加warmup
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            # 优化器设置
            'optimizer': 'AdamW',           # 使用AdamW，更好的正则化
            'cos_lr': True,                 # 余弦学习率衰减
            
            # 正则化 - 大幅增强
            'dropout': robust_params['dropout'],           # 大幅增强dropout
            'label_smoothing': robust_params['label_smoothing'],  # 增强标签平滑
            
            # 数据增强 - 大幅增强防止过拟合
            'mixup': robust_params['mixup'],               # 增强mixup
            'copy_paste': robust_params['mixup'] * 0.5,   # 增加copy_paste
            'degrees': min(10.0, 8.0 * robust_params['augment_factor']),      # 增强旋转
            'translate': min(0.2, 0.1 * robust_params['augment_factor']),     # 增强平移
            'scale': min(0.6, 0.3 * robust_params['augment_factor']),         # 增强缩放
            'shear': min(4.0, 2.0 * robust_params['augment_factor']),         # 增强剪切
            'perspective': min(0.0005, 0.0002 * robust_params['augment_factor']), # 增加透视变换
            'fliplr': 0.3,                  # 保持翻转
            'flipud': 0.1,                  # 增加垂直翻转
            'mosaic': 0.8,                  # 保持mosaic
            'close_mosaic': max(10, mode_config['epochs'] - 60),  # 延后关闭mosaic
            
            # HSV增强 - 增强
            'hsv_h': min(0.02, 0.01 * robust_params['augment_factor']),
            'hsv_s': min(0.6, 0.3 * robust_params['augment_factor']),
            'hsv_v': min(0.4, 0.2 * robust_params['augment_factor']),
            
            # 验证和保存设置 - 增强监控
            'save_period': -1,              
            'plots': True,
            
           
            
            # Loss设置 - 防过拟合调整
            'box': 7.5,
            'cls': 0.5,                     # 增加分类loss权重
            'dfl': 1.5,
            
            # 训练名称
            'name': f'overfitting_fixed_{robustness_level}_{training_mode}'
        }
        
        # 合并配置
        final_config = {**base_config, **mode_config, **anti_overfitting_config}
        
        # 配置验证
        self._validate_anti_overfitting_config(final_config, training_mode, robustness_level)
        
        return final_config
    
    def _validate_anti_overfitting_config(self, config, training_mode, robustness_level):
        """验证防过拟合配置"""
        print(f"\n防过拟合配置验证:")
        
        # 检查关键防过拟合参数
        overfitting_checks = {
            'weight_decay': ('权重衰减', 0.0003, '应该足够大防止过拟合'),
            'dropout': ('Dropout率', 0.1, '应该足够大防止过拟合'),
            'mixup': ('Mixup强度', 0.05, '应该开启防止过拟合'),
            'label_smoothing': ('标签平滑', 0.05, '应该开启防止过拟合'),
            'patience': ('早停耐心', 50, '应该足够小及时停止'),
            'epochs': ('训练轮数', 200, '不应过多避免过拟合')
        }
        
        recommendations = []
        
        for param, (desc, threshold, advice) in overfitting_checks.items():
            value = config.get(param, 'NOT_SET')
            if param in ['weight_decay', 'dropout', 'mixup', 'label_smoothing']:
                if isinstance(value, (int, float)) and value < threshold:
                    recommendations.append(f"建议增加{desc}: 当前{value}, {advice}")
                else:
                    print(f"  ✓ {desc}: {value} (防过拟合配置)")
            elif param in ['patience', 'epochs']:
                if isinstance(value, (int, float)) and value > threshold:
                    recommendations.append(f"建议减少{desc}: 当前{value}, {advice}")
                else:
                    print(f"  ✓ {desc}: {value} (防过拟合配置)")
            else:
                print(f"  ✓ {desc}: {value}")
        
        if recommendations:
            print(f"\n防过拟合建议:")
            for rec in recommendations:
                print(f"  💡 {rec}")
        
        # 检查正则化组合效果
        total_regularization = (
            config.get('weight_decay', 0) * 1000 +  # 权重衰减贡献
            config.get('dropout', 0) * 100 +        # dropout贡献
            config.get('mixup', 0) * 50 +           # mixup贡献
            config.get('label_smoothing', 0) * 20   # 标签平滑贡献
        )
        
        print(f"\n正则化强度评估:")
        if total_regularization < 50:
            print(f"  ⚠️ 正则化强度较低: {total_regularization:.1f}, 可能仍有过拟合风险")
        elif total_regularization > 200:
            print(f"  ⚠️ 正则化强度很高: {total_regularization:.1f}, 可能影响收敛")
        else:
            print(f"  ✓ 正则化强度适中: {total_regularization:.1f}, 平衡过拟合和性能")
        
        # 特别检查学习率设置
        lr_ratio = config['lrf']
        final_lr = config['lr0'] * lr_ratio
        print(f"  ✓ 初始学习率: {config['lr0']:.2e}")
        print(f"  ✓ 最终学习率: {final_lr:.2e}")
        
    
    def execute_anti_overfitting_training(self, training_mode="neck_head", robustness_level="high"):
        """执行防过拟合训练"""
        
        print("=" * 70)
        print("ANTI-OVERFITTING YOLO TRAINING")
        print("Overfitting-resistant approach - 解决过拟合问题")
        print("=" * 70)
        
        print(f"\n训练配置:")
        print(f"  模式: {training_mode}")
        print(f"  防过拟合级别: {robustness_level}")
        print(f"  描述: {self.robustness_levels[robustness_level]['description']}")
        
        # 设置环境
        self.setup_environment()
        
        # 验证路径
        if not self.validate_paths():
            print("路径验证失败!")
            return None
        
        # 创建模型
        print(f"\n加载模型: {self.model_config}")
        model = YOLO(self.model_config)
        
        # 应用层级训练控制
        has_trainable = self.apply_layer_training_control(model, training_mode)
        
        if not has_trainable:
            print("错误: 没有可训练参数!")
            return None
        
        # 获取防过拟合训练配置
        train_config = self.get_overfitting_resistant_config(training_mode, robustness_level)
        
        # 显示关键配置
        print(f"\n关键防过拟合参数:")
        print(f"  Epochs: {train_config['epochs']} (减少避免过训练)")
        print(f"  Batch size: {train_config['batch']} (增大提高稳定性)")
        print(f"  学习率: {train_config['lr0']:.6f} (降低避免过拟合)")
        print(f"  最终学习率: {train_config['lr0'] * train_config['lrf']:.6f}")
        print(f"  权重衰减: {train_config['weight_decay']:.4f} (增强防过拟合)")
        print(f"  Dropout: {train_config['dropout']:.2f} (增强防过拟合)")
        print(f"  Mixup: {train_config['mixup']:.2f} (增强防过拟合)")
        print(f"  标签平滑: {train_config['label_smoothing']:.2f} (增强防过拟合)")
        print(f"  早停耐心: {train_config['patience']} (积极早停)")
        
        
        print(f"\n预期改进:")
        print(f"  🎯 训练和验证损失同步下降")
        print(f"  📈 稳定的mAP增长无波动")
        print(f"  🚫 消除过拟合现象")
        print(f"  ⚖️ 更好的泛化性能")
        print(f"  ⏱️ 及时的早停避免过训练")
        print("=" * 70)
        
        try:
            print("开始防过拟合训练...")
            results = model.train(**train_config)
            
            print("\n" + "=" * 70)
            print("防过拟合训练成功完成!")
            print(f"结果保存在: {train_config['project']}/{train_config['name']}/")
            print("请检查新的训练曲线:")
            print("  - 训练和验证损失应该同步下降")
            print("  - mAP指标应该持续稳定增长")
            print("  - 没有明显的过拟合迹象")
            print("=" * 70)
            
            return results
            
        except KeyboardInterrupt:
            print("\n训练被用户中断")
            return None
            
        except Exception as e:
            print(f"\n训练失败: {e}")
            raise e


def main():
    """主函数 - 防过拟合版本"""
    
    print("Anti-Overfitting YOLO Training")
    print("专门解决训练过拟合、验证性能下降、泛化能力差问题")
    
    # 创建防过拟合训练器
    trainer = OverfittingFixedTrainer()
    
    # ============ 防过拟合配置 ============
    TRAINING_MODE = "neck"    # 推荐从neck_head开始
    ROBUSTNESS_LEVEL = "low"      # 推荐high级别防过拟合
    # ==================================
    
    print(f"\n防过拟合策略:")
    print(f"  1. 大幅增强权重衰减 (weight_decay)")
    print(f"  2. 增加Dropout防止过拟合")
    print(f"  3. 开启Mixup数据增强")
    print(f"  4. 开启标签平滑")
    print(f"  5. 降低学习率避免过快收敛")
    print(f"  6. 积极早停及时停止训练")
    print(f"  7. 增强数据增强提高泛化")
    print(f"  8. EMA模型平滑")
    
    print(f"\n当前配置:")
    print(f"  训练模式: {TRAINING_MODE}")
    print(f"  防过拟合级别: {ROBUSTNESS_LEVEL}")
    
    try:
        # 执行防过拟合训练
        results = trainer.execute_anti_overfitting_training(
            training_mode=TRAINING_MODE,
            robustness_level=ROBUSTNESS_LEVEL
        )
        
        if results:
            print(f"\n🎉 防过拟合训练完成!")
            print(f"📁 结果目录: runs/overfitting_fixed/overfitting_fixed_{ROBUSTNESS_LEVEL}_{TRAINING_MODE}/")
            print(f"📊 新的训练曲线应该显示:")
            print(f"   - 训练和验证损失同步下降")
            print(f"   - 稳定的mAP增长")
            print(f"   - 没有过拟合迹象")
            print(f"   - 更好的泛化性能")
            
    except KeyboardInterrupt:
        print(f"\n⏹️ 训练被用户停止")
        
    except Exception as e:
        print(f"\n💥 训练错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()