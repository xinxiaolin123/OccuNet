import warnings
import random
from pathlib import Path

import torch
import numpy as np
from ultralytics import YOLO

warnings.filterwarnings("ignore")


class OccuNetDetectorTrainer:
    def __init__(self):
        self.model_config = "/home/waas/simclr/Proj/ultralytics/cfg/models/12/yolo12-ours.yaml"
        self.train_config = "/home/waas/simclr/Proj/ultralytics/cfg/default.yaml"
        self.data_config = "/home/waas/simclr/Proj/ultralytics/cfg/datasets/Fracture.yaml"
        self.pretrained_weights = "/home/waas/simclr/Proj/best_pt/fold1newaug.pt"

        self.training_modes = {
            "backbone": "Backbone layers only",
            "neck": "Neck layers only",
            "head": "Detection head only",
            "neck_head": "Neck and detection head",
            "full": "All layers",
        }

        self.regularization_levels = {
            "low": {
                "weight_decay": 5e-4,
                "dropout": 0.15,
                "lr_factor": 0.8,
                "augment_factor": 1.2,
                "mixup": 0.10,
                "label_smoothing": 0.10,
            },
            "medium": {
                "weight_decay": 1e-3,
                "dropout": 0.20,
                "lr_factor": 0.6,
                "augment_factor": 1.5,
                "mixup": 0.15,
                "label_smoothing": 0.15,
            },
            "high": {
                "weight_decay": 2e-3,
                "dropout": 0.25,
                "lr_factor": 0.4,
                "augment_factor": 2.0,
                "mixup": 0.20,
                "label_smoothing": 0.20,
            },
            "extreme": {
                "weight_decay": 5e-3,
                "dropout": 0.30,
                "lr_factor": 0.2,
                "augment_factor": 2.5,
                "mixup": 0.25,
                "label_smoothing": 0.25,
            },
        }

    @staticmethod
    def setup_environment(seed=123):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.empty_cache()
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        print(f"PyTorch version: {torch.__version__}")
        print(f"Random seed: {seed}")
        print(f"CUDA available: {torch.cuda.is_available()}")

    def validate_paths(self):
        paths = {
            "Model configuration": self.model_config,
            "Training configuration": self.train_config,
            "Data configuration": self.data_config,
            "Pretrained weights": self.pretrained_weights,
        }

        all_valid = True
        for name, path in paths.items():
            if Path(path).exists():
                print(f"{name}: {path}")
            else:
                print(f"{name} missing: {path}")
                all_valid = False

        return all_valid

    def apply_layer_training_control(self, model, training_mode):
        print(f"\nTraining mode: {training_mode}")
        print(f"Trainable module group: {self.training_modes[training_mode]}")

        for param in model.model.parameters():
            param.requires_grad = False

        trainable_count = 0
        frozen_count = 0

        for name, module in model.model.named_modules():
            should_train = False

            if training_mode == "backbone":
                should_train = any(
                    pattern in name
                    for pattern in [
                        "model.0",
                        "model.1",
                        "model.2",
                        "model.3",
                        "model.4",
                        "model.5",
                        "model.6",
                        "model.7",
                        "model.8",
                        "model.9",
                    ]
                )
            elif training_mode == "neck":
                should_train = any(
                    pattern in name
                    for pattern in [
                        "model.10",
                        "model.11",
                        "model.12",
                        "model.13",
                        "model.14",
                        "model.15",
                        "model.16",
                        "model.17",
                        "model.18",
                        "model.19",
                        "model.20",
                    ]
                )
            elif training_mode == "head":
                should_train = "model.21" in name or "detect" in name.lower()
            elif training_mode == "neck_head":
                should_train = any(
                    pattern in name
                    for pattern in [
                        "model.10",
                        "model.11",
                        "model.12",
                        "model.13",
                        "model.14",
                        "model.15",
                        "model.16",
                        "model.17",
                        "model.18",
                        "model.19",
                        "model.20",
                        "model.21",
                    ]
                ) or "detect" in name.lower()
            elif training_mode == "full":
                should_train = True

            module_parameter_count = sum(param.numel() for param in module.parameters())

            if should_train:
                for param in module.parameters():
                    param.requires_grad = True
                trainable_count += module_parameter_count
            else:
                for param in module.parameters():
                    param.requires_grad = False
                frozen_count += module_parameter_count

        total_count = trainable_count + frozen_count
        training_ratio = 100 * trainable_count / total_count if total_count else 0

        print(f"Trainable parameters: {trainable_count:,}")
        print(f"Frozen parameters: {frozen_count:,}")
        print(f"Training ratio: {training_ratio:.1f}%")

        return trainable_count > 0

    def get_training_config(self, training_mode, regularization_level):
        regularization = self.regularization_levels[regularization_level]

        base_config = {
            "cfg": self.train_config,
            "data": self.data_config,
            "pretrained": self.pretrained_weights,
            "task": "detect",
            "mode": "train",
            "imgsz": 640,
            "device": 0,
            "workers": 8,
            "amp": True,
            "val": True,
            "plots": True,
            "save": True,
            "exist_ok": False,
            "project": "runs/occunet_stage2",
            "verbose": True,
            "deterministic": True,
            "resume": False,
            "seed": 789,
            "freeze": None,
        }

        if training_mode == "backbone":
            mode_config = {
                "epochs": 250,
                "batch": 32,
                "patience": 50,
                "lr0": 8e-5,
            }
        elif training_mode == "neck":
            mode_config = {
                "epochs": 250,
                "batch": 64,
                "patience": 50,
                "lr0": 8e-5,
            }
        elif training_mode == "head":
            mode_config = {
                "epochs": 250,
                "batch": 64,
                "patience": 50,
                "lr0": 1.5e-4,
            }
        elif training_mode == "neck_head":
            mode_config = {
                "epochs": 200,
                "batch": 32,
                "patience": 50,
                "lr0": 8e-5,
            }
        else:
            mode_config = {
                "epochs": 250,
                "batch": 32,
                "patience": 32,
                "lr0": 1e-5,
            }

        training_config = {
            "lr0": mode_config["lr0"] * regularization["lr_factor"],
            "lrf": 0.05,
            "momentum": 0.9,
            "weight_decay": regularization["weight_decay"],
            "warmup_epochs": max(5, mode_config["epochs"] // 15),
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            "optimizer": "AdamW",
            "cos_lr": True,
            "dropout": regularization["dropout"],
            "label_smoothing": regularization["label_smoothing"],
            "mixup": regularization["mixup"],
            "copy_paste": regularization["mixup"] * 0.5,
            "degrees": min(10.0, 8.0 * regularization["augment_factor"]),
            "translate": min(0.2, 0.1 * regularization["augment_factor"]),
            "scale": min(0.6, 0.3 * regularization["augment_factor"]),
            "shear": min(4.0, 2.0 * regularization["augment_factor"]),
            "perspective": min(0.0005, 0.0002 * regularization["augment_factor"]),
            "fliplr": 0.3,
            "flipud": 0.1,
            "mosaic": 0.8,
            "close_mosaic": max(10, mode_config["epochs"] - 60),
            "hsv_h": min(0.02, 0.01 * regularization["augment_factor"]),
            "hsv_s": min(0.6, 0.3 * regularization["augment_factor"]),
            "hsv_v": min(0.4, 0.2 * regularization["augment_factor"]),
            "save_period": -1,
            "plots": True,
            "box": 7.5,
            "cls": 0.5,
            "dfl": 1.5,
            "name": f"occunet_{regularization_level}_{training_mode}",
        }

        final_config = {**base_config, **mode_config, **training_config}
        self.report_training_config(final_config, training_mode, regularization_level)
        return final_config

    @staticmethod
    def report_training_config(config, training_mode, regularization_level):
        print("\nTraining configuration:")
        print(f"Mode: {training_mode}")
        print(f"Regularization level: {regularization_level}")
        print(f"Epochs: {config['epochs']}")
        print(f"Batch size: {config['batch']}")
        print(f"Initial learning rate: {config['lr0']:.2e}")
        print(f"Final learning rate: {config['lr0'] * config['lrf']:.2e}")
        print(f"Weight decay: {config['weight_decay']:.4f}")
        print(f"Dropout: {config['dropout']:.2f}")
        print(f"Mixup: {config['mixup']:.2f}")
        print(f"Label smoothing: {config['label_smoothing']:.2f}")
        print(f"Early stopping patience: {config['patience']}")

    def train(self, training_mode="neck_head", regularization_level="high"):
        print("=" * 70)
        print("OccuNet Stage 2 detector fine-tuning")
        print("=" * 70)

        self.setup_environment()

        if not self.validate_paths():
            print("Path validation failed.")
            return None

        print(f"\nLoading model: {self.model_config}")
        model = YOLO(self.model_config)

        has_trainable_parameters = self.apply_layer_training_control(model, training_mode)
        if not has_trainable_parameters:
            print("No trainable parameters were selected.")
            return None

        train_config = self.get_training_config(training_mode, regularization_level)

        try:
            print("\nTraining started.")
            results = model.train(**train_config)

            print("\nTraining completed.")
            print(f"Output directory: {train_config['project']}/{train_config['name']}/")
            return results

        except KeyboardInterrupt:
            print("\nTraining interrupted.")
            return None

        except Exception as exc:
            print(f"\nTraining failed: {exc}")
            raise exc


def main():
    trainer = OccuNetDetectorTrainer()

    training_mode = "neck"
    regularization_level = "low"

    try:
        results = trainer.train(
            training_mode=training_mode,
            regularization_level=regularization_level,
        )

        if results:
            print("\nStage 2 fine-tuning completed.")
            print(f"Output directory: runs/occunet_stage2/occunet_{regularization_level}_{training_mode}/")

    except KeyboardInterrupt:
        print("\nTraining stopped by user.")

    except Exception as exc:
        print(f"\nTraining error: {exc}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
