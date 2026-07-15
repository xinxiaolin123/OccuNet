from __future__ import annotations

import argparse
import random
from collections.abc import Callable
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = Path(__file__).resolve().parent / "Test_radiograph.png"
DEFAULT_SEED = 2026


def read_grayscale(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to decode image: {path}")
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"Unable to encode image: {path}")
    encoded.tofile(path)


def add_scattered_dead_pixels(image: np.ndarray, intensity_factor: float = 1.0) -> np.ndarray:
    h, w = image.shape[:2]
    image_float = image.astype(np.float32)
    base_min = max(1, int(1000 * intensity_factor))
    base_max = max(base_min, int(2000 * intensity_factor))
    num_dead_pixels = random.randint(base_min, base_max)

    for _ in range(num_dead_pixels):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        skip = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                xx, yy = x + dx, y + dy
                if 0 <= xx < w and 0 <= yy < h:
                    if image_float[yy, xx] < 30 or image_float[yy, xx] > 225:
                        if random.random() < 0.2:
                            skip = True
                            break
            if skip:
                break
        if not skip:
            image_float[y, x] = random.randint(0, 15) if random.random() < 0.7 else random.randint(240, 255)

    return np.clip(image_float, 0, 255).astype(np.uint8)


def add_electrical_interference(image: np.ndarray, intensity_factor: float = 1.0) -> np.ndarray:
    h, w = image.shape[:2]
    image_float = image.astype(np.float32)
    base_min = max(1, int(25 * intensity_factor))
    base_max = max(base_min, int(55 * intensity_factor))
    stripe_intensity = random.randint(base_min, base_max)
    stripe_direction = random.choice(("horizontal", "vertical", "both"))

    if stripe_direction in ("horizontal", "both"):
        for i in range(0, h, random.randint(3, 12)):
            image_float[i, :] += random.randint(-stripe_intensity, stripe_intensity)
    if stripe_direction in ("vertical", "both"):
        for i in range(0, w, random.randint(3, 12)):
            image_float[:, i] += random.randint(-stripe_intensity, stripe_intensity)

    return np.clip(image_float, 0, 255).astype(np.uint8)


def add_mechanical_scratches(image: np.ndarray, intensity_factor: float = 1.0) -> np.ndarray:
    h, w = image.shape[:2]
    image_float = image.astype(np.float32)
    base_min_count = max(1, int(6 * intensity_factor))
    base_max_count = max(base_min_count, int(12 * intensity_factor))
    num_scratches = random.randint(base_min_count, base_max_count)
    base_min_intensity = int(-60 * intensity_factor)
    base_max_intensity = int(-30 * intensity_factor)

    for _ in range(num_scratches):
        scratch_intensity = random.randint(base_min_intensity, base_max_intensity)
        width = random.randint(1, 4)
        if random.random() < 0.5:
            y_pos = random.randint(0, h - 1)
            start_x = random.randint(0, max(0, w // 4))
            end_x = random.randint(max(start_x + 1, 3 * w // 4), w)
            for dy in range(-width // 2, width // 2 + 1):
                yy = y_pos + dy
                if 0 <= yy < h:
                    image_float[yy, start_x:end_x] += scratch_intensity
        else:
            x_pos = random.randint(0, w - 1)
            start_y = random.randint(0, max(0, h // 4))
            end_y = random.randint(max(start_y + 1, 3 * h // 4), h)
            for dx in range(-width // 2, width // 2 + 1):
                xx = x_pos + dx
                if 0 <= xx < w:
                    image_float[start_y:end_y, xx] += scratch_intensity

    return np.clip(image_float, 0, 255).astype(np.uint8)


def _random_region(length: int, start_limit: int, min_size: int, max_size: int) -> tuple[int, int]:
    start = random.randint(0, max(0, min(start_limit, length - 1)))
    available = length - start
    if available <= 1:
        return start, length
    low = min(min_size, available)
    high = min(max_size, available)
    if low > high:
        low = high
    return start, start + random.randint(max(1, low), max(1, high))


def add_power_fluctuation(image: np.ndarray, intensity_factor: float = 1.0) -> np.ndarray:
    h, w = image.shape[:2]
    image_float = image.astype(np.float32)
    base_min_regions = max(1, int(8 * intensity_factor))
    base_max_regions = max(base_min_regions, int(15 * intensity_factor))
    num_regions = random.randint(base_min_regions, base_max_regions)
    base_fluctuation = max(1, int(60 * intensity_factor))

    for _ in range(num_regions):
        x1, x2 = _random_region(w, w // 2, 50, 250)
        y1, y2 = _random_region(h, h // 2, 50, 250)
        image_float[y1:y2, x1:x2] += random.randint(-base_fluctuation, base_fluctuation)

    return np.clip(image_float, 0, 255).astype(np.uint8)


def add_dust_spots(image: np.ndarray, intensity_factor: float = 1.0) -> np.ndarray:
    h, w = image.shape[:2]
    image_float = image.astype(np.float32)
    base_min_spots = max(1, int(150 * intensity_factor))
    base_max_spots = max(base_min_spots, int(300 * intensity_factor))
    num_dust_spots = random.randint(base_min_spots, base_max_spots)
    base_min_intensity = int(-80 * intensity_factor)
    base_max_intensity = int(-35 * intensity_factor)

    for _ in range(num_dust_spots):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        spot_size = random.randint(1, 6)
        spot_intensity = random.randint(base_min_intensity, base_max_intensity)
        value = float(image_float[y, x] + spot_intensity)
        cv2.circle(image_float, (x, y), spot_size, value, -1)

    return np.clip(image_float, 0, 255).astype(np.uint8)


def add_simple_noise(image: np.ndarray, intensity_factor: float = 1.0) -> np.ndarray:
    sigma = max(1, int(30 * intensity_factor))
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def add_grid_artifact(image: np.ndarray, intensity_factor: float = 1.0) -> np.ndarray:
    h, w = image.shape[:2]
    base_min = max(1, int(20 * intensity_factor))
    base_max = max(base_min, int(50 * intensity_factor))
    amplitude = random.randint(base_min, base_max)

    if random.random() < 0.5:
        axis = np.arange(h)
        pattern = np.sin(axis * random.uniform(0.02, 0.25)) * amplitude
        pattern = np.tile(pattern.reshape(-1, 1), (1, w))
    else:
        axis = np.arange(w)
        pattern = np.sin(axis * random.uniform(0.02, 0.25)) * amplitude
        pattern = np.tile(pattern.reshape(1, -1), (h, 1))

    return np.clip(image.astype(np.float32) + pattern, 0, 255).astype(np.uint8)


def add_sensor_saturation(image: np.ndarray, intensity_factor: float = 1.0) -> np.ndarray:
    h, w = image.shape[:2]
    image_float = image.astype(np.float32)
    fx1, fx2 = int(w * 0.25), int(w * 0.75)
    fy1, fy2 = int(h * 0.25), int(h * 0.75)

    def overlaps(x1: int, y1: int, x2: int, y2: int) -> bool:
        return not (x2 < fx1 or x1 > fx2 or y2 < fy1 or y1 > fy2)

    base_min = max(1, int(5 * intensity_factor))
    base_max = max(base_min, int(10 * intensity_factor))
    target = random.randint(base_min, base_max)
    created = 0

    for _ in range(target * 3):
        if created >= target:
            break
        x1, x2 = _random_region(w, w // 2, 40, 150)
        y1, y2 = _random_region(h, h // 2, 40, 150)
        if not overlaps(x1, y1, x2, y2):
            value = random.randint(230, 255) if random.random() < 0.7 else random.randint(0, 25)
            image_float[y1:y2, x1:x2] = value
            created += 1

    return np.clip(image_float, 0, 255).astype(np.uint8)


def add_cable_shadows(image: np.ndarray, intensity_factor: float = 1.0) -> np.ndarray:
    h, w = image.shape[:2]
    image_float = image.astype(np.float32)
    base_min = max(1, int(3 * intensity_factor))
    base_max = max(base_min, int(8 * intensity_factor))
    num_cables = random.randint(base_min, base_max)
    base_min_intensity = int(-55 * intensity_factor)
    base_max_intensity = int(-25 * intensity_factor)

    for _ in range(num_cables):
        cable_width = random.randint(4, 10)
        shadow_intensity = random.randint(base_min_intensity, base_max_intensity)
        if random.random() < 0.7:
            y_pos = random.randint(h // 4, max(h // 4, 3 * h // 4))
            start_x = random.randint(0, max(0, w // 3))
            end_x = random.randint(max(start_x + 1, 2 * w // 3), w)
            for dy in range(-cable_width // 2, cable_width // 2 + 1):
                yy = y_pos + dy
                if 0 <= yy < h:
                    image_float[yy, start_x:end_x] += shadow_intensity
        else:
            x_pos = random.randint(w // 4, max(w // 4, 3 * w // 4))
            start_y = random.randint(0, max(0, h // 3))
            end_y = random.randint(max(start_y + 1, 2 * h // 3), h)
            for dx in range(-cable_width // 2, cable_width // 2 + 1):
                xx = x_pos + dx
                if 0 <= xx < w:
                    image_float[start_y:end_y, xx] += shadow_intensity

    return np.clip(image_float, 0, 255).astype(np.uint8)


def add_electromagnetic_interference(image: np.ndarray, intensity_factor: float = 1.0) -> np.ndarray:
    h, w = image.shape[:2]
    y, x = np.ogrid[:h, :w]
    center_x = random.randint(w // 4, max(w // 4, 3 * w // 4))
    center_y = random.randint(h // 4, max(h // 4, 3 * h // 4))
    distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    frequency = random.uniform(0.08, 0.4)
    base_min = max(1, int(25 * intensity_factor))
    base_max = max(base_min, int(50 * intensity_factor))
    amplitude = random.randint(base_min, base_max)
    wave_pattern = amplitude * np.sin(distance * frequency)
    return np.clip(image.astype(np.float32) + wave_pattern, 0, 255).astype(np.uint8)


def add_detector_malfunction(image: np.ndarray, intensity_factor: float = 1.0) -> np.ndarray:
    h, w = image.shape[:2]
    image_float = image.astype(np.float32)
    base_min = max(1, int(2 * intensity_factor))
    base_max = max(base_min, int(6 * intensity_factor))
    num_dead_zones = random.randint(base_min, base_max)

    for _ in range(num_dead_zones):
        if random.choice(("rectangle", "line")) == "rectangle":
            x1, x2 = _random_region(w, w // 2, 30, 120)
            y1, y2 = _random_region(h, h // 2, 30, 120)
            image_float[y1:y2, x1:x2] = random.randint(0, 25)
        elif random.random() < 0.5:
            image_float[random.randint(0, h - 1), :] = random.randint(0, 25)
        else:
            image_float[:, random.randint(0, w - 1)] = random.randint(0, 25)

    return np.clip(image_float, 0, 255).astype(np.uint8)


def add_compression_artifacts(image: np.ndarray, intensity_factor: float = 1.0) -> np.ndarray:
    base_min, base_max = 15, 40
    quality = max(base_min, min(base_max, base_max - int((base_max - base_min) * intensity_factor)))
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise OSError("JPEG encoding failed during compression artifact simulation.")
    compressed = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    h, w = compressed.shape
    block_size = 8
    base_noise = max(1, int(15 * intensity_factor))

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            if random.random() < 0.4:
                end_i = min(i + block_size, h)
                end_j = min(j + block_size, w)
                block = compressed[i:end_i, j:end_j].astype(np.float32)
                block += random.randint(-base_noise, base_noise)
                compressed[i:end_i, j:end_j] = np.clip(block, 0, 255).astype(np.uint8)

    return compressed


def add_window_level_variation(image: np.ndarray, intensity_factor: float = 1.0) -> np.ndarray:
    image_float = image.astype(np.float32)
    adjustment_type = random.choice(("contrast", "brightness", "gamma", "subtle_window"))

    if adjustment_type == "contrast":
        base_range = 0.4 * intensity_factor
        factor = random.uniform(1.0 - base_range / 2, 1.0 + base_range / 2)
        mean_value = float(np.mean(image_float))
        result = (image_float - mean_value) * factor + mean_value
    elif adjustment_type == "brightness":
        offset = max(1, int(35 * intensity_factor))
        result = image_float + random.randint(-offset, offset)
    elif adjustment_type == "gamma":
        base_range = 0.5 * intensity_factor
        gamma = random.uniform(1.0 - base_range / 2, 1.0 + base_range / 2)
        result = np.power(image_float / 255.0, gamma) * 255.0
    else:
        image_min, image_max = float(np.min(image_float)), float(np.max(image_float))
        image_range = max(image_max - image_min, 1.0)
        width_factor = random.uniform(1.0 - 0.35 * intensity_factor, 1.0 + 0.35 * intensity_factor)
        new_width = max(image_range * width_factor, 1.0)
        current_center = (image_min + image_max) / 2
        center_shift = random.uniform(-0.2 * intensity_factor, 0.2 * intensity_factor) * image_range
        new_center = current_center + center_shift
        new_min = new_center - new_width / 2
        result = (image_float - new_min) / new_width * 255.0

    return np.clip(result, 0, 255).astype(np.uint8)


def add_natural_xray_blur(image: np.ndarray, intensity_factor: float = 1.0) -> np.ndarray:
    size = random.randint(3, 7)
    kernel = np.zeros((size, size), dtype=np.float32)
    if random.random() < 0.5:
        kernel[size // 2, :] = 1
    else:
        kernel[:, size // 2] = 1
    kernel /= np.sum(kernel)
    return cv2.filter2D(image, -1, kernel)


ARTIFACTS: tuple[tuple[str, str, Callable[[np.ndarray, float], np.ndarray]], ...] = (
    ("dead_pixels", "Dead pixels", add_scattered_dead_pixels),
    ("electrical_interference", "Electrical\ninterference", add_electrical_interference),
    ("mechanical_scratches", "Mechanical\nscratches", add_mechanical_scratches),
    ("power_fluctuation", "Power\nfluctuation", add_power_fluctuation),
    ("dust_spots", "Dust spots", add_dust_spots),
    ("quantum_noise", "Quantum noise", add_simple_noise),
    ("grid_artifact", "Grid artifact", add_grid_artifact),
    ("sensor_saturation", "Sensor\nsaturation", add_sensor_saturation),
    ("cable_shadows", "Cable shadows", add_cable_shadows),
    ("electromagnetic_interference", "Electromagnetic\ninterference", add_electromagnetic_interference),
    ("detector_fault", "Detector fault", add_detector_malfunction),
    ("compression_artifacts", "Compression\nartifacts", add_compression_artifacts),
    ("window_level_variation", "Window/level\nvariation", add_window_level_variation),
    ("motion_blur", "Motion blur", add_natural_xray_blur),
)


def apply_with_seed(
    function: Callable[[np.ndarray, float], np.ndarray],
    image: np.ndarray,
    intensity_factor: float,
    seed: int,
) -> np.ndarray:
    random.seed(seed)
    np.random.seed(seed)
    return function(image.copy(), intensity_factor)


def create_comparison_figure(
    panels: list[tuple[str, np.ndarray]],
    output_path: Path,
    dpi: int,
) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titleweight": "semibold",
    })
    fig, axes = plt.subplots(3, 5, figsize=(15, 9.6), facecolor="white")
    panel_letters = "ABCDEFGHIJKLMNO"

    for index, (ax, (label, image)) in enumerate(zip(axes.flat, panels, strict=True)):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(
            f"{panel_letters[index]}  {label}",
            fontsize=13.5,
            color="#24292f",
            pad=7,
            linespacing=1.05,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#d0d7de")
            spine.set_linewidth(0.8)

    fig.suptitle(
        "Physics-informed Radiographic Artifact Simulation",
        fontsize=20,
        fontweight="bold",
        color="#1f2328",
        y=0.985,
    )
    fig.text(
        0.5,
        0.952,
        "Single-effect examples generated from the same de-identified AP pelvic radiograph using fixed random seeds",
        ha="center",
        va="top",
        fontsize=13.5,
        color="#57606a",
    )
    fig.subplots_adjust(
        left=0.025,
        right=0.975,
        bottom=0.025,
        top=0.91,
        wspace=0.08,
        hspace=0.24,
    )
    fig.savefig(output_path, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def generate_artifact_examples(
    input_path: Path,
    output_dir: Path | None = None,
    intensity_factor: float = 1.0,
    base_seed: int = DEFAULT_SEED,
) -> Path:
    if intensity_factor <= 0:
        raise ValueError("intensity_factor must be greater than zero.")

    image = read_grayscale(input_path)
    destination = output_dir or input_path.parent / "artifact_simulation_output"
    destination.mkdir(parents=True, exist_ok=True)

    panels: list[tuple[str, np.ndarray]] = [("Original", image)]
    write_png(destination / "00_original.png", image)

    for index, (filename, label, function) in enumerate(ARTIFACTS, start=1):
        artifact_image = apply_with_seed(function, image, intensity_factor, base_seed + index)
        write_png(destination / f"{index:02d}_{filename}.png", artifact_image)
        panels.append((label, artifact_image))

    comparison_path = destination / "artifact_comparison_3x5.png"
    github_path = destination / "artifact_examples.png"
    create_comparison_figure(panels, comparison_path, dpi=300)
    create_comparison_figure(panels, github_path, dpi=160)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic examples of 14 radiographic artifacts.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to the source radiograph.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory; defaults next to the input image.")
    parser.add_argument("--intensity-factor", type=float, default=1.0, help="Shared artifact intensity multiplier.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Base seed for deterministic generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = generate_artifact_examples(
        input_path=args.input,
        output_dir=args.output_dir,
        intensity_factor=args.intensity_factor,
        base_seed=args.seed,
    )
    print(f"Saved original, 14 artifact images, a 300-dpi comparison, and a GitHub-ready preview to: {output_dir}")


if __name__ == "__main__":
    main()
