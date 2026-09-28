import json

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import convolve
from skimage.filters import threshold_otsu
from skimage.transform import probabilistic_hough_line

from assignment_utils import IMAGES_DIR, ensure_output_dir, load_grayscale_image, normalize_to_uint8, save_grayscale_image, save_text


OUTPUT_DIR = ensure_output_dir("askisi6_edges")
IMAGE_PATH = IMAGES_DIR / "Ασκηση 6" / "hallway.png"
LINE_COLOR = (255, 0, 0)
LINE_WIDTH = 3

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)


def save_panel(images: list[tuple[str, np.ndarray]], path) -> None:
    """Save a row of image panels."""
    fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 5))
    if len(images) == 1:
        axes = [axes]

    for ax, (title, image) in zip(axes, images):
        if image.ndim == 2:
            ax.imshow(image, cmap="gray")
        else:
            ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def overlay_lines(original: np.ndarray, lines: list[tuple[tuple[int, int], tuple[int, int]]]) -> np.ndarray:
    """Draw detected line segments on top of the original grayscale image."""
    rgb = np.stack([original.astype(np.uint8)] * 3, axis=-1)
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)

    for (x0, y0), (x1, y1) in lines:
        draw.line((x0, y0, x1, y1), fill=LINE_COLOR, width=LINE_WIDTH)

    return np.asarray(image)


def main() -> None:
    original = load_grayscale_image(IMAGE_PATH)
    gx = convolve(original, SOBEL_X, mode="reflect")
    gy = convolve(original, SOBEL_Y, mode="reflect")
    gradient_magnitude = np.hypot(gx, gy)
    magnitude_uint8 = normalize_to_uint8(gradient_magnitude)

    threshold_value = float(threshold_otsu(magnitude_uint8))
    edge_map = magnitude_uint8 >= threshold_value
    lines = probabilistic_hough_line(edge_map, threshold=10, line_length=80, line_gap=10)
    overlay = overlay_lines(original, lines)

    save_grayscale_image(normalize_to_uint8(np.abs(gx)), OUTPUT_DIR / "sobel_x.png")
    save_grayscale_image(normalize_to_uint8(np.abs(gy)), OUTPUT_DIR / "sobel_y.png")
    save_grayscale_image(magnitude_uint8, OUTPUT_DIR / "gradient_magnitude.png")
    save_grayscale_image(edge_map.astype(np.uint8) * 255, OUTPUT_DIR / "edge_map.png")
    Image.fromarray(overlay).save(OUTPUT_DIR / "lines_overlay.png")

    save_panel(
        [
            ("Original", original),
            ("|Sobel X|", normalize_to_uint8(np.abs(gx))),
            ("|Sobel Y|", normalize_to_uint8(np.abs(gy))),
        ],
        OUTPUT_DIR / "sobel_summary.png",
    )
    save_panel(
        [
            ("Gradient Magnitude", magnitude_uint8),
            (f"Thresholded Edges (T={threshold_value:.1f})", edge_map.astype(np.uint8) * 255),
            ("Detected Lines Overlay", overlay),
        ],
        OUTPUT_DIR / "edges_and_lines_summary.png",
    )

    summary = {
        "threshold_value": threshold_value,
        "detected_line_segments": len(lines),
        "line_color_rgb": LINE_COLOR,
        "line_width": LINE_WIDTH,
    }
    summary_text = json.dumps(summary, indent=2)
    save_text(summary_text, OUTPUT_DIR / "summary.json")
    print(summary_text)
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
