import warnings
import time
import re
import random
import glob
from pathlib import Path
from collections import defaultdict

import cv2
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from ultralytics.models.yolo.model import YOLO
from ultralytics.utils import LOGGER

warnings.filterwarnings("ignore")


def parse_filename_for_pairing(filename):
    name = Path(filename).stem
    base_match = re.search(r"^(\d+)", name)
    if not base_match:
        return None, False

    base_id = base_match.group(1)
    is_augmented = bool(re.search(r"_pos_aug\d*", name.lower()))
    return base_id, is_augmented


def load_stage1_config(cfg_path):
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f) or {}
    except Exception:
        raw_config = {}

    config = {
        "stage1_epochs": raw_config.get("stage1_epochs", 100),
        "simclr_temperature": raw_config.get("simclr_temperature", 0.1),
        "simclr_lr": raw_config.get("simclr_lr", 5e-4),
        "simclr_weight_decay": raw_config.get("simclr_weight_decay", 1e-4),
        "simclr_hidden_dim": raw_config.get("simclr_hidden_dim", 512),
        "simclr_output_dim": raw_config.get("simclr_output_dim", 256),
        "batch_size": raw_config.get("batch_size", 64),
        "batches_per_epoch": raw_config.get("batches_per_epoch", 60),
        "min_pairs_per_batch": raw_config.get("min_pairs_per_batch", 6),
        "max_pairs_per_batch": raw_config.get("max_pairs_per_batch", 24),
        "print_freq": raw_config.get("print_freq", 10),
        "warmup_epochs": raw_config.get("warmup_epochs", 10),
        "cosine_annealing": raw_config.get("cosine_annealing", True),
        "mixed_precision": raw_config.get("mixed_precision", True),
        "feature_extraction_layer": raw_config.get("feature_extraction_layer", "AIFI"),
        "max_groups_per_batch": raw_config.get("max_groups_per_batch", 20),
        "samples_per_group": raw_config.get("samples_per_group", 3),
    }

    LOGGER.info("Stage 1 contrastive pretraining configuration:")
    for key, value in config.items():
        LOGGER.info(f"  {key}: {value}")
    return config


class RadiographAugmentation:
    def __init__(self, img_size=640):
        self.img_size = img_size

    def augment_batch(self, images):
        augmented = []

        for img in images:
            if random.random() > 0.5:
                img = torch.flip(img, dims=[2])

            if random.random() > 0.7:
                brightness_factor = 0.8 + random.random() * 0.4
                img = torch.clamp(img * brightness_factor, 0, 1)

            augmented.append(img)

        return torch.stack(augmented)


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, features, labels):
        batch_size = features.shape[0]
        device = features.device

        if batch_size < 2:
            return torch.tensor(0.0, device=device, requires_grad=True), 0

        features = F.normalize(features, dim=1, eps=1e-8)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        unique_labels = list(set(labels))
        if len(unique_labels) < 2:
            return torch.tensor(0.0, device=device, requires_grad=True), 0

        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        numeric_labels = torch.tensor([label_to_idx[label] for label in labels], device=device)

        labels_eq = numeric_labels.unsqueeze(0) == numeric_labels.unsqueeze(1)
        diagonal_mask = torch.eye(batch_size, device=device).bool()
        labels_eq = labels_eq & ~diagonal_mask

        positive_pairs = 0
        processed = set()

        for i in range(batch_size):
            if i in processed:
                continue

            positive_mask = labels_eq[i]
            if positive_mask.any():
                positive_indices = positive_mask.nonzero(as_tuple=True)[0]
                for positive_index in positive_indices:
                    positive_value = positive_index.item()
                    if positive_value not in processed:
                        positive_pairs += 1
                        processed.add(i)
                        processed.add(positive_value)
                        break

        if positive_pairs == 0:
            return torch.tensor(0.0, device=device, requires_grad=True), 0

        losses = []

        for i in range(batch_size):
            positive_mask = labels_eq[i]
            if not positive_mask.any():
                continue

            positive_indices = positive_mask.nonzero(as_tuple=True)[0]
            if len(positive_indices) == 0:
                continue

            positive_index = positive_indices[0]
            negative_mask = ~positive_mask & ~diagonal_mask[i]

            if not negative_mask.any():
                continue

            positive_similarity = similarity_matrix[i, positive_index]
            negative_similarities = similarity_matrix[i, negative_mask]

            logits = torch.cat([positive_similarity.unsqueeze(0), negative_similarities])
            target = torch.zeros(1, dtype=torch.long, device=device)
            loss = self.criterion(logits.unsqueeze(0), target)
            losses.append(loss)

        if not losses:
            return torch.tensor(0.0, device=device, requires_grad=True), 0

        return torch.stack(losses).mean(), positive_pairs


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=256):
        super().__init__()
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)

        projected = self.projection(x)
        return F.normalize(projected, dim=1, eps=1e-8)


class PairedRadiographDataset:
    def __init__(self, img_path, img_size=640):
        self.img_size = img_size
        self.augmentation = RadiographAugmentation(img_size)

        self.img_files = []
        for extension in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            self.img_files.extend(glob.glob(f"{img_path}/**/{extension}", recursive=True))

        self.original_images = defaultdict(list)
        self.augmented_images = defaultdict(list)

        for img_file in self.img_files:
            base_id, is_augmented = parse_filename_for_pairing(Path(img_file).name)
            if not base_id:
                continue

            if is_augmented:
                self.augmented_images[base_id].append(img_file)
            else:
                self.original_images[base_id].append(img_file)

        self.paired_groups = {}
        total_potential_pairs = 0

        paired_ids = set(self.original_images.keys()) & set(self.augmented_images.keys())
        for base_id in paired_ids:
            originals = self.original_images[base_id]
            augmented = self.augmented_images[base_id]

            if originals and augmented:
                self.paired_groups[base_id] = {
                    "originals": originals,
                    "augmented": augmented,
                }
                total_potential_pairs += len(originals) * len(augmented)

        LOGGER.info("Dataset summary:")
        LOGGER.info(f"  total images: {len(self.img_files)}")
        LOGGER.info(f"  paired groups: {len(self.paired_groups)}")
        LOGGER.info(f"  total potential pairs: {total_potential_pairs}")

    def create_batch(self, batch_size, max_groups=20):
        batch_files = []
        batch_labels = []

        if not self.paired_groups:
            return [], []

        available_groups = list(self.paired_groups.keys())
        random.shuffle(available_groups)

        target_groups = min(max_groups, len(available_groups), batch_size // 2)
        selected_groups = available_groups[:target_groups]

        for group_id in selected_groups:
            if len(batch_files) >= batch_size:
                break

            group_data = self.paired_groups[group_id]
            if not group_data["originals"] or not group_data["augmented"]:
                continue

            remaining = batch_size - len(batch_files)
            max_for_group = min(3, remaining)

            if max_for_group > 0:
                original = random.choice(group_data["originals"])
                batch_files.append(original)
                batch_labels.append(group_id)
                max_for_group -= 1

            if max_for_group > 0:
                n_augmented = min(max_for_group, len(group_data["augmented"]))
                augmented_samples = random.sample(group_data["augmented"], n_augmented)

                for augmented_file in augmented_samples:
                    if len(batch_files) < batch_size:
                        batch_files.append(augmented_file)
                        batch_labels.append(group_id)

        return batch_files, batch_labels

    def load_batch_images(self, batch_files):
        images = []

        for img_file in batch_files:
            try:
                img = cv2.imread(img_file)
                if img is None:
                    raise ValueError("Image read failed")

                img = cv2.resize(img, (self.img_size, self.img_size))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception:
                img = np.random.randint(
                    0,
                    255,
                    (self.img_size, self.img_size, 3),
                    dtype=np.uint8,
                )

            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            images.append(img_tensor)

        batch_tensor = torch.stack(images)
        return self.augmentation.augment_batch(batch_tensor)

    @staticmethod
    def get_batch_statistics(batch_labels):
        label_counts = defaultdict(int)
        for label in batch_labels:
            label_counts[label] += 1

        valid_pairs = sum(
            count * (count - 1) // 2
            for count in label_counts.values()
            if count >= 2
        )

        return {
            "total_pairs": valid_pairs,
            "unique_groups": len(label_counts),
            "avg_group_size": (
                sum(label_counts.values()) / len(label_counts)
                if label_counts
                else 0
            ),
        }


class FeatureHookExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.features = None
        self.hook_handle = None
        self._register_hook()

    def _register_hook(self):
        def hook_fn(module, inputs, output):
            self.features = output

        if len(self.model.model) > 9:
            target_layer = self.model.model[9]
            self.hook_handle = target_layer.register_forward_hook(hook_fn)

    def forward(self, x):
        self.features = None
        _ = self.model(x)
        return self.features

    def close(self):
        if self.hook_handle:
            self.hook_handle.remove()
            self.hook_handle = None


class ContrastivePretrainingYOLO(YOLO):
    def __init__(self, model="yolo12s.yaml", task=None, verbose=True):
        super().__init__(model, task, verbose)
        LOGGER.info("Contrastive pretraining model initialized.")

    def train_contrastive_stage1(self, **kwargs):
        LOGGER.info("Starting Stage 1 contrastive pretraining.")

        config = load_stage1_config(kwargs.get("cfg", "/path/to/default.yaml"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_amp = bool(config["mixed_precision"] and device.type == "cuda")

        output_dir = Path("./simple_contrastive_output2")
        output_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info(f"Device: {device}")
        LOGGER.info(f"Epochs: {config['stage1_epochs']}")
        LOGGER.info(f"Temperature: {config['simclr_temperature']}")
        LOGGER.info(f"Batch size: {config['batch_size']}")

        try:
            with open(kwargs.get("data"), "r", encoding="utf-8") as f:
                data_config = yaml.safe_load(f)

            dataset = PairedRadiographDataset(data_config["train"])
        except Exception as exc:
            LOGGER.error(f"Dataset preparation failed: {exc}")
            return None

        model = self.model.to(device)
        model.eval()
        feature_extractor = FeatureHookExtractor(model).to(device)

        dummy_input = torch.randn(1, 3, 640, 640).to(device)
        with torch.no_grad():
            dummy_features = feature_extractor(dummy_input)
            if dummy_features is not None and len(dummy_features.shape) == 4:
                feature_dim = dummy_features.shape[1]
            else:
                feature_dim = 1024

        LOGGER.info(f"Feature dimension: {feature_dim}")

        projection_head = ProjectionHead(
            input_dim=feature_dim,
            hidden_dim=config["simclr_hidden_dim"],
            output_dim=config["simclr_output_dim"],
        ).to(device)

        optimizer = torch.optim.AdamW(
            projection_head.parameters(),
            lr=config["simclr_lr"],
            weight_decay=config["simclr_weight_decay"],
            betas=(0.9, 0.999),
        )

        if config["cosine_annealing"]:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config["stage1_epochs"],
                eta_min=config["simclr_lr"] * 0.1,
            )
        else:
            scheduler = None

        contrastive_criterion = ContrastiveLoss(config["simclr_temperature"])
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        projection_head.train()
        losses = []
        best_loss = float("inf")
        best_epoch = 0

        for epoch in range(config["stage1_epochs"]):
            epoch_start_time = time.time()
            epoch_losses = []
            epoch_pairs = []

            if epoch < config["warmup_epochs"]:
                warmup_lr = config["simclr_lr"] * (epoch + 1) / config["warmup_epochs"]
                for param_group in optimizer.param_groups:
                    param_group["lr"] = warmup_lr

            for batch_idx in range(config["batches_per_epoch"]):
                try:
                    batch_files, batch_labels = dataset.create_batch(
                        config["batch_size"],
                        config["max_groups_per_batch"],
                    )

                    if len(batch_files) < 2:
                        continue

                    images = dataset.load_batch_images(batch_files).to(device)
                    batch_stats = dataset.get_batch_statistics(batch_labels)

                    with torch.no_grad():
                        features = feature_extractor(images)

                    if features is None:
                        continue

                    if scaler:
                        with torch.cuda.amp.autocast():
                            projected = projection_head(features)
                            contrastive_loss, pairs = contrastive_criterion(
                                projected,
                                batch_labels,
                            )
                    else:
                        projected = projection_head(features)
                        contrastive_loss, pairs = contrastive_criterion(
                            projected,
                            batch_labels,
                        )

                    optimizer.zero_grad()

                    if scaler:
                        scaler.scale(contrastive_loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            projection_head.parameters(),
                            max_norm=1.0,
                        )
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        contrastive_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            projection_head.parameters(),
                            max_norm=1.0,
                        )
                        optimizer.step()

                    epoch_losses.append(contrastive_loss.item())
                    epoch_pairs.append(pairs)

                    if batch_idx % config["print_freq"] == 0:
                        current_lr = optimizer.param_groups[0]["lr"]
                        LOGGER.info(
                            f"Epoch {epoch + 1:03d} Batch {batch_idx:03d}: "
                            f"loss={contrastive_loss.item():.5f}, "
                            f"pairs={pairs:02d}, "
                            f"groups={batch_stats['unique_groups']:02d}, "
                            f"lr={current_lr:.6f}"
                        )

                except Exception as exc:
                    LOGGER.error(f"Batch {batch_idx} failed: {exc}")
                    continue

            if scheduler and epoch >= config["warmup_epochs"]:
                scheduler.step()

            if not epoch_losses:
                continue

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            avg_pairs = sum(epoch_pairs) / len(epoch_pairs) if epoch_pairs else 0
            losses.append(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_epoch = epoch + 1

                best_checkpoint = {
                    "epoch": best_epoch,
                    "projection_head_state_dict": projection_head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    "config": config,
                    "feature_dim": feature_dim,
                }
                torch.save(best_checkpoint, output_dir / "best_simple_model.pt")

            epoch_time = time.time() - epoch_start_time
            current_lr = optimizer.param_groups[0]["lr"]

            LOGGER.info(
                f"Epoch {epoch + 1:03d}: loss={avg_loss:.5f}, "
                f"pairs={avg_pairs:.1f}, best={best_loss:.5f}@E{best_epoch}, "
                f"lr={current_lr:.6f}, time={epoch_time:.1f}s"
            )

        final_checkpoint = {
            "projection_head_state_dict": projection_head.state_dict(),
            "config": config,
            "feature_dim": feature_dim,
            "final_loss": best_loss,
        }

        weights_path = output_dir / "simple_contrastive_weights.pt"
        torch.save(final_checkpoint, weights_path)

        self._plot_training_curve(losses, output_dir)
        feature_extractor.close()

        LOGGER.info("Stage 1 contrastive pretraining completed.")
        LOGGER.info(f"Weights saved to: {weights_path}")
        LOGGER.info(f"Best loss: {best_loss:.5f} at epoch {best_epoch}")

        return {
            "losses": losses,
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "weights_path": str(weights_path),
        }

    @staticmethod
    def _plot_training_curve(losses, output_dir):
        try:
            plt.figure(figsize=(8, 6))

            epochs = range(1, len(losses) + 1)
            plt.plot(epochs, losses, "b-", linewidth=2, label="Contrastive loss")

            if losses:
                min_loss = min(losses)
                min_epoch = losses.index(min_loss) + 1
                plt.plot(
                    min_epoch,
                    min_loss,
                    "ro",
                    markersize=8,
                    label=f"Best: {min_loss:.4f}@E{min_epoch}",
                )

            plt.title("Stage 1 contrastive loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                output_dir / "simple_contrastive_curves.png",
                dpi=200,
                bbox_inches="tight",
            )
            plt.close()

        except Exception as exc:
            LOGGER.error(f"Failed to plot training curve: {exc}")


def main():
    try:
        LOGGER.info("Creating OccuNet Stage 1 contrastive pretraining model.")

        model = ContrastivePretrainingYOLO(
            "/home/waas/simclr/Proj/ultralytics/cfg/models/12/yolo12-ours.yaml"
        )

        results = model.train_contrastive_stage1(
            cfg="/home/waas/simclr/Proj/ultralytics/cfg/default.yaml",
            data="/home/waas/simclr/Proj/ultralytics/cfg/datasets/Fracture.yaml",
            name="stage1_contrastive_pretraining",
        )

        if results:
            print("\nStage 1 contrastive pretraining completed.")
            print(f"Best loss: {results['best_loss']:.5f} at epoch {results['best_epoch']}")
            print(f"Weights: {results['weights_path']}")

    except Exception as exc:
        LOGGER.error(f"Stage 1 training failed: {exc}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
