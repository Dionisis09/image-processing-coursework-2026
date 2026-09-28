from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parent
IMAGE_PATH = ROOT_DIR / "Images" / "Ασκηση 1" / "moon.jpg"
OUTPUT_DIR = ROOT_DIR / "outputs" / "askisi1_erotima1"
SELECTED_CUTOFF_RADIUS = 35
CUTOFF_CANDIDATES = [15, 25, 35, 45, 60]


def load_grayscale_image(path: Path) -> np.ndarray:
    """Load the input image as a grayscale float64 array."""
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.float64)


def linear_stretch(image: np.ndarray) -> np.ndarray:
    """Stretch the image values so they fully cover [0, 255]."""
    min_val = image.min()
    max_val = image.max()

    if max_val == min_val:
        return np.zeros_like(image)

    stretched = 255.0 * (image - min_val) / (max_val - min_val)
    return stretched


def center_dft_origin(image: np.ndarray) -> np.ndarray:
    """Move the (0,0) frequency to the center using the DFT property."""
    rows, cols = image.shape
    row_idx, col_idx = np.indices((rows, cols))
    checkerboard = (-1) ** (row_idx + col_idx)
    return image * checkerboard


def dft2_rows_columns(image: np.ndarray) -> np.ndarray:
    """Compute the 2D-DFT using consecutive 1D DFTs on rows and columns."""
    row_transformed = np.apply_along_axis(np.fft.fft, axis=1, arr=image)
    return np.apply_along_axis(np.fft.fft, axis=0, arr=row_transformed)


def idft2_rows_columns(spectrum: np.ndarray) -> np.ndarray:
    """Compute the 2D-IDFT using consecutive 1D inverse DFTs."""
    col_inverted = np.apply_along_axis(np.fft.ifft, axis=0, arr=spectrum)
    return np.apply_along_axis(np.fft.ifft, axis=1, arr=col_inverted)


def magnitude_spectrum(spectrum: np.ndarray) -> np.ndarray:
    """Return the magnitude of a complex spectrum."""
    return np.abs(spectrum)


def ideal_lowpass_filter(shape: tuple[int, int], cutoff_radius: float) -> np.ndarray:
    """Create an ideal lowpass mask centered in the frequency domain."""
    rows, cols = shape
    row_idx, col_idx = np.indices((rows, cols))
    center_row = rows // 2
    center_col = cols // 2
    distance = np.sqrt((row_idx - center_row) ** 2 + (col_idx - center_col) ** 2)
    return (distance <= cutoff_radius).astype(np.float64)


def mse(reference: np.ndarray, test: np.ndarray) -> float:
    """Compute mean squared error between two images."""
    return float(np.mean((reference - test) ** 2))


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    """Normalize any image to [0, 255] for visualization."""
    min_val = image.min()
    max_val = image.max()

    if max_val == min_val:
        return np.zeros_like(image, dtype=np.uint8)

    normalized = 255.0 * (image - min_val) / (max_val - min_val)
    return normalized.astype(np.uint8)


def save_image(image: np.ndarray, path: Path) -> None:
    """Save an array as an 8-bit grayscale image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="L").save(path)


def save_figure(
    original: np.ndarray,
    stretched: np.ndarray,
    centered_visual: np.ndarray,
    path: Path,
) -> None:
    """Create a summary figure with the main preprocessing stages."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(stretched, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Linear Stretch")
    axes[1].axis("off")

    axes[2].imshow(centered_visual, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("Centered for DFT")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_spectrum_figure(
    linear_magnitude: np.ndarray,
    log_magnitude: np.ndarray,
    path: Path,
) -> None:
    """Create a figure with linear and logarithmic magnitude displays."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(normalize_for_display(linear_magnitude), cmap="gray")
    axes[0].set_title("Magnitude - Linear Scale")
    axes[0].axis("off")

    axes[1].imshow(normalize_for_display(log_magnitude), cmap="gray")
    axes[1].set_title("Magnitude - Log Scale")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_filtering_figure(
    filter_mask: np.ndarray,
    filtered_log_magnitude: np.ndarray,
    restored_image: np.ndarray,
    path: Path,
) -> None:
    """Create a figure that summarizes the filtering stage."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(filter_mask, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Ideal Lowpass Mask (D0={SELECTED_CUTOFF_RADIUS})")
    axes[0].axis("off")

    axes[1].imshow(normalize_for_display(filtered_log_magnitude), cmap="gray")
    axes[1].set_title("Filtered Spectrum - Log Scale")
    axes[1].axis("off")

    axes[2].imshow(restored_image, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("Restored Spatial Image")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_mse_curve(cutoff_radii: list[int], mse_values: list[float], path: Path) -> None:
    """Save the reconstruction MSE for several cutoff radii."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cutoff_radii, mse_values, marker="o")
    ax.set_xlabel("Cutoff Radius")
    ax.set_ylabel("MSE")
    ax.set_title("Lowpass Cutoff Radius vs Reconstruction MSE")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    original = load_grayscale_image(IMAGE_PATH)
    stretched = linear_stretch(original)
    centered = center_dft_origin(stretched)
    centered_visual = normalize_for_display(centered)
    spectrum = dft2_rows_columns(centered)
    linear_magnitude = magnitude_spectrum(spectrum)
    log_magnitude = np.log1p(linear_magnitude)

    cutoff_mse_values = []
    selected_outputs = {}
    for cutoff_radius in CUTOFF_CANDIDATES:
        lowpass_mask = ideal_lowpass_filter(spectrum.shape, cutoff_radius)
        filtered_spectrum = spectrum * lowpass_mask
        filtered_log_magnitude = np.log1p(magnitude_spectrum(filtered_spectrum))
        restored_centered = idft2_rows_columns(filtered_spectrum)
        restored_spatial = center_dft_origin(np.real(restored_centered))
        restored_spatial = np.clip(restored_spatial, 0, 255)
        reconstruction_mse = mse(stretched, restored_spatial)
        cutoff_mse_values.append(reconstruction_mse)

        if cutoff_radius == SELECTED_CUTOFF_RADIUS:
            selected_outputs = {
                "lowpass_mask": lowpass_mask,
                "filtered_spectrum": filtered_spectrum,
                "filtered_log_magnitude": filtered_log_magnitude,
                "restored_spatial": restored_spatial,
                "restored_uint8": restored_spatial.astype(np.uint8),
                "reconstruction_mse": reconstruction_mse,
            }

    save_image(stretched, OUTPUT_DIR / "moon_stretched.png")
    save_image(centered_visual, OUTPUT_DIR / "moon_centered_visual.png")
    save_figure(original, stretched, centered_visual, OUTPUT_DIR / "preprocessing_summary.png")
    save_image(normalize_for_display(linear_magnitude), OUTPUT_DIR / "dft_magnitude_linear.png")
    save_image(normalize_for_display(log_magnitude), OUTPUT_DIR / "dft_magnitude_log.png")
    save_spectrum_figure(linear_magnitude, log_magnitude, OUTPUT_DIR / "spectrum_summary.png")
    save_image(selected_outputs["lowpass_mask"] * 255.0, OUTPUT_DIR / "lowpass_mask.png")
    save_image(normalize_for_display(selected_outputs["filtered_log_magnitude"]), OUTPUT_DIR / "filtered_spectrum_log.png")
    save_image(selected_outputs["restored_uint8"], OUTPUT_DIR / "moon_lowpass_restored.png")
    save_filtering_figure(
        selected_outputs["lowpass_mask"],
        selected_outputs["filtered_log_magnitude"],
        selected_outputs["restored_uint8"],
        OUTPUT_DIR / "filtering_summary.png",
    )
    save_mse_curve(CUTOFF_CANDIDATES, cutoff_mse_values, OUTPUT_DIR / "cutoff_mse_curve.png")

    np.save(OUTPUT_DIR / "moon_stretched.npy", stretched)
    np.save(OUTPUT_DIR / "moon_centered_for_dft.npy", centered)
    np.save(OUTPUT_DIR / "moon_dft.npy", spectrum)
    np.save(OUTPUT_DIR / "lowpass_mask.npy", selected_outputs["lowpass_mask"])
    np.save(OUTPUT_DIR / "moon_filtered_dft.npy", selected_outputs["filtered_spectrum"])
    np.save(OUTPUT_DIR / "moon_restored_spatial.npy", selected_outputs["restored_spatial"])

    print(f"Input image: {IMAGE_PATH}")
    print(f"Original shape: {original.shape}")
    print(f"Original min/max: {original.min():.2f} / {original.max():.2f}")
    print(f"Stretched min/max: {stretched.min():.2f} / {stretched.max():.2f}")
    print(f"Tested cutoff radii: {CUTOFF_CANDIDATES}")
    print(f"Selected cutoff radius: {SELECTED_CUTOFF_RADIUS}")
    print(f"Reconstruction MSE after lowpass filtering: {selected_outputs['reconstruction_mse']:.4f}")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
