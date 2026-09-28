from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure

from assignment_utils import IMAGES_DIR, ensure_output_dir, load_grayscale_image, save_grayscale_image, save_text


OUTPUT_DIR = ensure_output_dir("askisi2_histogram_equalization")
INPUT_DIR = IMAGES_DIR / "Ασκηση 2"
IMAGE_PATHS = sorted(INPUT_DIR.glob("dark_road_*.jpg"))
ADAPTIVE_KERNEL_SIZE = (64, 64)
ADAPTIVE_CANDIDATES = [(32, 32), (64, 64), (128, 128)]


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert a [0, 1] float image to uint8."""
    return np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8)


def histogram_counts(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return grayscale histogram counts and bin positions."""
    counts, bins = np.histogram(image.ravel(), bins=256, range=(0, 255))
    return counts, bins[:-1]


def noise_proxy(image: np.ndarray) -> float:
    """Estimate a simple high-frequency energy proxy."""
    gx = np.diff(image, axis=1, prepend=image[:, :1])
    gy = np.diff(image, axis=0, prepend=image[:1, :])
    return float(np.std(gx) + np.std(gy))


def save_histogram_figure(original: np.ndarray, equalized: np.ndarray, adaptive: np.ndarray, path: Path) -> None:
    """Create a figure with images and their histograms."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    entries = [
        ("Original", original),
        ("Global HE", equalized),
        (f"Adaptive HE {ADAPTIVE_KERNEL_SIZE}", adaptive),
    ]

    for row, (title, image) in enumerate(entries):
        axes[row, 0].imshow(image, cmap="gray", vmin=0, vmax=255)
        axes[row, 0].set_title(title)
        axes[row, 0].axis("off")

        counts, bins = histogram_counts(image)
        axes[row, 1].plot(bins, counts, color="black", linewidth=1)
        axes[row, 1].set_title(f"{title} Histogram")
        axes[row, 1].set_xlim(0, 255)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_adaptive_candidates_figure(original: np.ndarray, variants: list[tuple[tuple[int, int], np.ndarray]], path: Path) -> None:
    """Compare several adaptive histogram equalization window sizes."""
    fig, axes = plt.subplots(1, len(variants) + 1, figsize=(5 * (len(variants) + 1), 5))

    axes[0].imshow(original, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original")
    axes[0].axis("off")

    for ax, (kernel_size, image) in zip(axes[1:], variants):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"CLAHE {kernel_size}")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def process_image(path: Path) -> str:
    """Process one image and return a metrics summary line."""
    image = load_grayscale_image(path)
    image_unit = image / 255.0

    global_he = exposure.equalize_hist(image_unit)
    adaptive_he = exposure.equalize_adapthist(image_unit, kernel_size=ADAPTIVE_KERNEL_SIZE, clip_limit=0.01)

    candidate_images = []
    candidate_lines = []
    for kernel_size in ADAPTIVE_CANDIDATES:
        adaptive_candidate = exposure.equalize_adapthist(image_unit, kernel_size=kernel_size, clip_limit=0.01)
        adaptive_candidate_uint8 = to_uint8(adaptive_candidate)
        candidate_images.append((kernel_size, adaptive_candidate_uint8))
        candidate_lines.append(
            f"kernel={kernel_size} contrast_std={np.std(adaptive_candidate_uint8):.2f} noise_proxy={noise_proxy(adaptive_candidate_uint8):.2f}"
        )

    global_he_uint8 = to_uint8(global_he)
    adaptive_he_uint8 = to_uint8(adaptive_he)

    base_name = path.stem
    save_grayscale_image(global_he_uint8, OUTPUT_DIR / f"{base_name}_global_he.png")
    save_grayscale_image(adaptive_he_uint8, OUTPUT_DIR / f"{base_name}_adaptive_he.png")
    save_histogram_figure(image, global_he_uint8, adaptive_he_uint8, OUTPUT_DIR / f"{base_name}_histograms.png")
    save_adaptive_candidates_figure(image, candidate_images, OUTPUT_DIR / f"{base_name}_adaptive_candidates.png")

    metrics_text = "\n".join(candidate_lines)
    save_text(metrics_text, OUTPUT_DIR / f"{base_name}_adaptive_metrics.txt")

    return (
        f"{base_name}: original_std={np.std(image):.2f}, "
        f"global_std={np.std(global_he_uint8):.2f}, "
        f"adaptive_std={np.std(adaptive_he_uint8):.2f}, "
        f"adaptive_noise_proxy={noise_proxy(adaptive_he_uint8):.2f}"
    )


def main() -> None:
    summary_lines = [
        f"Adaptive kernel size selected: {ADAPTIVE_KERNEL_SIZE}",
        "Candidate kernels: " + ", ".join(str(item) for item in ADAPTIVE_CANDIDATES),
        "",
    ]

    for image_path in IMAGE_PATHS:
        summary_lines.append(process_image(image_path))

    summary_text = "\n".join(summary_lines)
    save_text(summary_text, OUTPUT_DIR / "summary.txt")
    print(summary_text)
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
