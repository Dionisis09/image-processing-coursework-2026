import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import median_filter, uniform_filter

from assignment_utils import (
    IMAGES_DIR,
    add_gaussian_noise_with_exact_snr,
    ensure_output_dir,
    mse,
    psnr,
    save_grayscale_image,
    save_text,
    snr_db,
)


OUTPUT_DIR = ensure_output_dir("askisi3_denoising")
MAT_PATH = IMAGES_DIR / "Ασκηση 3" / "tiger.mat"
TARGET_SNR_DB = 15.0
IMPULSE_DENSITY = 0.20
KERNEL_SIZES = (3, 5)
RNG = np.random.default_rng(42)


def load_tiger() -> np.ndarray:
    """Load the tiger image from the MAT file."""
    data = loadmat(MAT_PATH)
    return np.asarray(data["tiger"], dtype=np.float64)


def add_gaussian_noise_for_snr(image: np.ndarray, snr_db: float, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Add zero-mean Gaussian noise to reach a target SNR."""
    return add_gaussian_noise_with_exact_snr(image, snr_db, rng, 0.0, 1.0)


def add_impulse_noise(image: np.ndarray, density: float, rng: np.random.Generator) -> np.ndarray:
    """Add salt-and-pepper noise with the requested density."""
    noisy = image.copy()
    random_map = rng.random(image.shape)
    pepper_mask = random_map < (density / 2.0)
    salt_mask = (random_map >= (density / 2.0)) & (random_map < density)
    noisy[pepper_mask] = 0.0
    noisy[salt_mask] = 1.0
    return noisy


def moving_average_filter(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply a moving average filter."""
    return uniform_filter(image, size=kernel_size, mode="reflect")


def median_denoise(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply a median filter."""
    return median_filter(image, size=kernel_size, mode="reflect")


def metrics_dict(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Compute denoising metrics."""
    return {
        "mse": mse(reference, candidate),
        "psnr_db": psnr(reference, candidate, data_range=1.0),
    }


def save_triplet_figure(
    original: np.ndarray,
    noisy: np.ndarray,
    restored: np.ndarray,
    title: str,
    path: Path,
) -> None:
    """Save a 3-panel comparison figure."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    panels = [("Original", original), ("Noisy", noisy), (title, restored)]

    for ax, (panel_title, image) in zip(axes, panels):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(panel_title)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_kernel_comparison_figure(
    noisy: np.ndarray,
    mean_results: dict[int, np.ndarray],
    median_results: dict[int, np.ndarray],
    path: Path,
) -> None:
    """Save a figure comparing kernel sizes and filter types."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes[0, 0].imshow(noisy, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Noisy Input")
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")

    for column, kernel_size in enumerate(KERNEL_SIZES, start=1):
        axes[0, column].imshow(mean_results[kernel_size], cmap="gray", vmin=0, vmax=1)
        axes[0, column].set_title(f"Mean {kernel_size}x{kernel_size}")
        axes[0, column].axis("off")

        axes[1, column].imshow(median_results[kernel_size], cmap="gray", vmin=0, vmax=1)
        axes[1, column].set_title(f"Median {kernel_size}x{kernel_size}")
        axes[1, column].axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def process_noise_case(
    case_name: str,
    original: np.ndarray,
    noisy: np.ndarray,
) -> dict[str, object]:
    """Apply both filters with multiple kernels and collect metrics."""
    mean_results = {k: moving_average_filter(noisy, k) for k in KERNEL_SIZES}
    median_results = {k: median_denoise(noisy, k) for k in KERNEL_SIZES}

    save_kernel_comparison_figure(
        noisy,
        mean_results,
        median_results,
        OUTPUT_DIR / f"{case_name}_kernel_comparison.png",
    )

    results: dict[str, object] = {
        "noisy_metrics": metrics_dict(original, noisy),
        "filters": {},
    }

    for kernel_size in KERNEL_SIZES:
        mean_image = mean_results[kernel_size]
        median_image = median_results[kernel_size]
        results["filters"][f"mean_{kernel_size}"] = metrics_dict(original, mean_image)
        results["filters"][f"median_{kernel_size}"] = metrics_dict(original, median_image)

        save_grayscale_image(mean_image * 255.0, OUTPUT_DIR / f"{case_name}_mean_{kernel_size}.png")
        save_grayscale_image(median_image * 255.0, OUTPUT_DIR / f"{case_name}_median_{kernel_size}.png")

    return results


def process_combined_case(original: np.ndarray) -> dict[str, object]:
    """Handle the mixed Gaussian + impulse noise case and compare filter order."""
    gaussian_noisy, gaussian_noise_power = add_gaussian_noise_for_snr(original, TARGET_SNR_DB, RNG)
    combined_noisy = add_impulse_noise(gaussian_noisy, IMPULSE_DENSITY, RNG)

    mean_then_median = median_denoise(moving_average_filter(combined_noisy, 3), 3)
    median_then_mean = moving_average_filter(median_denoise(combined_noisy, 3), 3)

    save_triplet_figure(
        original,
        combined_noisy,
        mean_then_median,
        "Mean -> Median",
        OUTPUT_DIR / "combined_mean_then_median.png",
    )
    save_triplet_figure(
        original,
        combined_noisy,
        median_then_mean,
        "Median -> Mean",
        OUTPUT_DIR / "combined_median_then_mean.png",
    )

    result = {
        "noise_power": gaussian_noise_power,
        "noisy_metrics": metrics_dict(original, combined_noisy),
        "mean_then_median": metrics_dict(original, mean_then_median),
        "median_then_mean": metrics_dict(original, median_then_mean),
    }

    result["recommended_order"] = (
        "median_then_mean"
        if result["median_then_mean"]["mse"] < result["mean_then_median"]["mse"]
        else "mean_then_median"
    )

    save_grayscale_image(combined_noisy * 255.0, OUTPUT_DIR / "combined_noisy.png")
    save_grayscale_image(mean_then_median * 255.0, OUTPUT_DIR / "combined_mean_then_median_restored.png")
    save_grayscale_image(median_then_mean * 255.0, OUTPUT_DIR / "combined_median_then_mean_restored.png")
    return result


def main() -> None:
    original = load_tiger()
    save_grayscale_image(original * 255.0, OUTPUT_DIR / "tiger_original.png")

    gaussian_noisy, gaussian_noise_power = add_gaussian_noise_for_snr(original, TARGET_SNR_DB, RNG)
    impulse_noisy = add_impulse_noise(original, IMPULSE_DENSITY, RNG)

    save_grayscale_image(gaussian_noisy * 255.0, OUTPUT_DIR / "tiger_gaussian_noisy.png")
    save_grayscale_image(impulse_noisy * 255.0, OUTPUT_DIR / "tiger_impulse_noisy.png")

    gaussian_results = process_noise_case("gaussian", original, gaussian_noisy)
    impulse_results = process_noise_case("impulse", original, impulse_noisy)
    combined_results = process_combined_case(original)

    summary = {
        "target_snr_db": TARGET_SNR_DB,
        "impulse_density": IMPULSE_DENSITY,
        "gaussian_noise_power": gaussian_noise_power,
        "gaussian_actual_snr_db": snr_db(original, gaussian_noisy),
        "gaussian_case": gaussian_results,
        "impulse_case": impulse_results,
        "combined_case": combined_results,
    }

    summary_text = json.dumps(summary, indent=2)
    save_text(summary_text, OUTPUT_DIR / "summary.json")
    print(summary_text)
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
