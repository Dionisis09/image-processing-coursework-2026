import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dctn, idctn

from assignment_utils import IMAGES_DIR, ensure_output_dir, load_grayscale_image, mse, save_grayscale_image, save_text


OUTPUT_DIR = ensure_output_dir("askisi5_dct_compression")
IMAGE_PATH = IMAGES_DIR / "Ασκηση 5" / "board.png"
BLOCK_SIZE = 32
R_VALUES = list(range(5, 55, 5))
SHOWCASE_R_VALUES = [10, 25, 50]


def pad_to_block_size(image: np.ndarray, block_size: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Pad an image so both dimensions are divisible by block_size."""
    rows, cols = image.shape
    padded_rows = int(np.ceil(rows / block_size) * block_size)
    padded_cols = int(np.ceil(cols / block_size) * block_size)
    pad_bottom = padded_rows - rows
    pad_right = padded_cols - cols
    padded = np.pad(image, ((0, pad_bottom), (0, pad_right)), mode="edge")
    return padded, (rows, cols)


def image_to_blocks(image: np.ndarray, block_size: int) -> np.ndarray:
    """Split a padded image into non-overlapping blocks."""
    rows, cols = image.shape
    reshaped = image.reshape(rows // block_size, block_size, cols // block_size, block_size)
    return reshaped.transpose(0, 2, 1, 3)


def blocks_to_image(blocks: np.ndarray) -> np.ndarray:
    """Reconstruct an image from its block representation."""
    block_rows, block_cols, block_size, _ = blocks.shape
    return blocks.transpose(0, 2, 1, 3).reshape(block_rows * block_size, block_cols * block_size)


def zigzag_indices(size: int) -> np.ndarray:
    """Return zigzag traversal indices for a square matrix."""
    order = []
    for diag in range(2 * size - 1):
        coords = []
        row_start = max(0, diag - size + 1)
        row_end = min(diag, size - 1)
        for row in range(row_start, row_end + 1):
            col = diag - row
            coords.append((row, col))
        if diag % 2 == 0:
            coords.reverse()
        order.extend(coords)
    return np.array(order, dtype=int)


def zonal_mask(size: int, keep_count: int) -> np.ndarray:
    """Create a zigzag-based zonal coding mask."""
    mask = np.zeros((size, size), dtype=np.float64)
    coords = zigzag_indices(size)[:keep_count]
    mask[coords[:, 0], coords[:, 1]] = 1.0
    return mask


def reconstruct_from_blocks(blocks: np.ndarray) -> np.ndarray:
    """Apply inverse 2D-DCT blockwise."""
    return idctn(blocks, axes=(-2, -1), norm="ortho")


def save_comparison_figure(
    original: np.ndarray,
    reconstructions: dict[int, np.ndarray],
    method_name: str,
    path,
) -> None:
    """Save side-by-side comparisons for selected retained-information ratios."""
    fig, axes = plt.subplots(1, len(reconstructions) + 1, figsize=(5 * (len(reconstructions) + 1), 5))

    axes[0].imshow(original, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original")
    axes[0].axis("off")

    for ax, (r_value, image) in zip(axes[1:], reconstructions.items()):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{method_name} r={r_value}%")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_mse_curve(r_values: list[int], zonal_mse: list[float], threshold_mse: list[float], path) -> None:
    """Save the MSE-vs-r plot for both methods."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r_values, zonal_mse, marker="o", label="Zonal coding")
    ax.plot(r_values, threshold_mse, marker="s", label="Threshold coding")
    ax.set_xlabel("Retained Information r (%)")
    ax.set_ylabel("MSE")
    ax.set_title("MSE vs Retained Information")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    original = load_grayscale_image(IMAGE_PATH)
    padded, original_shape = pad_to_block_size(original, BLOCK_SIZE)
    blocks = image_to_blocks(padded, BLOCK_SIZE)
    dct_blocks = dctn(blocks, axes=(-2, -1), norm="ortho")
    flat_coeffs = dct_blocks.reshape(-1, BLOCK_SIZE * BLOCK_SIZE)
    abs_flat_coeffs = np.abs(flat_coeffs)

    metrics_summary: dict[str, dict[int, dict[str, float]]] = {"zonal": {}, "threshold": {}}
    zonal_mse_values: list[float] = []
    threshold_mse_values: list[float] = []
    zonal_showcase: dict[int, np.ndarray] = {}
    threshold_showcase: dict[int, np.ndarray] = {}

    for r_value in R_VALUES:
        keep_count = max(1, int(round((r_value / 100.0) * (BLOCK_SIZE * BLOCK_SIZE))))

        zonal = dct_blocks * zonal_mask(BLOCK_SIZE, keep_count)[None, None, :, :]
        zonal_reconstructed = blocks_to_image(reconstruct_from_blocks(zonal))[: original_shape[0], : original_shape[1]]
        zonal_reconstructed = np.clip(zonal_reconstructed, 0, 255)
        zonal_error = mse(original, zonal_reconstructed)
        zonal_mse_values.append(zonal_error)
        metrics_summary["zonal"][r_value] = {"mse": zonal_error}

        threshold_mask = np.zeros_like(flat_coeffs, dtype=np.float64)
        top_indices = np.argpartition(abs_flat_coeffs, -keep_count, axis=1)[:, -keep_count:]
        row_indices = np.arange(flat_coeffs.shape[0])[:, None]
        threshold_mask[row_indices, top_indices] = 1.0
        threshold_blocks = (flat_coeffs * threshold_mask).reshape(dct_blocks.shape)
        threshold_reconstructed = blocks_to_image(reconstruct_from_blocks(threshold_blocks))
        threshold_reconstructed = np.clip(threshold_reconstructed[: original_shape[0], : original_shape[1]], 0, 255)
        threshold_error = mse(original, threshold_reconstructed)
        threshold_mse_values.append(threshold_error)
        metrics_summary["threshold"][r_value] = {"mse": threshold_error}

        save_grayscale_image(zonal_reconstructed, OUTPUT_DIR / f"zonal_r_{r_value}.png")
        save_grayscale_image(threshold_reconstructed, OUTPUT_DIR / f"threshold_r_{r_value}.png")

        if r_value in SHOWCASE_R_VALUES:
            zonal_showcase[r_value] = zonal_reconstructed
            threshold_showcase[r_value] = threshold_reconstructed

    save_comparison_figure(original, zonal_showcase, "Zonal", OUTPUT_DIR / "zonal_showcase.png")
    save_comparison_figure(original, threshold_showcase, "Threshold", OUTPUT_DIR / "threshold_showcase.png")
    save_mse_curve(R_VALUES, zonal_mse_values, threshold_mse_values, OUTPUT_DIR / "mse_curve.png")

    summary_text = json.dumps(
        {
            "block_size": BLOCK_SIZE,
            "r_values": R_VALUES,
            "original_shape": original_shape,
            "padded_shape": padded.shape,
            "metrics": metrics_summary,
        },
        indent=2,
    )
    save_text(summary_text, OUTPUT_DIR / "summary.json")
    print(summary_text)
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
