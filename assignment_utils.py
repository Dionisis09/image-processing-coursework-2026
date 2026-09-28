from pathlib import Path

import numpy as np
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = ROOT_DIR / "Images"
OUTPUTS_DIR = ROOT_DIR / "outputs"


def load_grayscale_image(path: Path, as_float: bool = True) -> np.ndarray:
    """Load an image as grayscale."""
    image = Image.open(path).convert("L")
    array = np.asarray(image)
    return array.astype(np.float64) if as_float else array


def save_grayscale_image(image: np.ndarray, path: Path) -> None:
    """Save an array as an 8-bit grayscale image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(clipped, mode="L").save(path)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize an array to [0, 255] for display."""
    image = np.asarray(image, dtype=np.float64)
    min_val = image.min()
    max_val = image.max()

    if max_val == min_val:
        return np.zeros_like(image, dtype=np.uint8)

    scaled = 255.0 * (image - min_val) / (max_val - min_val)
    return scaled.astype(np.uint8)


def ensure_output_dir(name: str) -> Path:
    """Create and return the output directory for an exercise."""
    out_dir = OUTPUTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def mse(reference: np.ndarray, test: np.ndarray) -> float:
    """Mean squared error."""
    ref = np.asarray(reference, dtype=np.float64)
    tst = np.asarray(test, dtype=np.float64)
    return float(np.mean((ref - tst) ** 2))


def psnr(reference: np.ndarray, test: np.ndarray, data_range: float | None = None) -> float:
    """Peak signal-to-noise ratio in dB."""
    error = mse(reference, test)
    if error == 0:
        return float("inf")

    if data_range is None:
        data_range = float(np.max(reference) - np.min(reference))
        if data_range == 0:
            data_range = 1.0

    return float(10.0 * np.log10((data_range**2) / error))


def snr_db(reference: np.ndarray, test: np.ndarray) -> float:
    """Signal-to-noise ratio in dB using the reference as the clean signal."""
    ref = np.asarray(reference, dtype=np.float64)
    tst = np.asarray(test, dtype=np.float64)
    noise = tst - ref
    signal_power = float(np.mean(ref**2))
    noise_power = float(np.mean(noise**2))
    if noise_power == 0:
        return float("inf")
    return float(10.0 * np.log10(signal_power / noise_power))


def add_gaussian_noise_with_exact_snr(
    image: np.ndarray,
    target_snr_db: float,
    rng: np.random.Generator,
    clip_min: float,
    clip_max: float,
    iterations: int = 40,
) -> tuple[np.ndarray, float]:
    """Add Gaussian noise and calibrate its scale so the clipped result matches the target SNR."""
    image = np.asarray(image, dtype=np.float64)
    base_noise = rng.normal(0.0, 1.0, size=image.shape)

    signal_power = float(np.mean(image**2))
    target_noise_power = signal_power / (10 ** (target_snr_db / 10.0))
    initial_scale = np.sqrt(target_noise_power)

    low_scale = 0.0
    high_scale = max(initial_scale, 1e-12)
    noisy_high = np.clip(image + high_scale * base_noise, clip_min, clip_max)

    while snr_db(image, noisy_high) > target_snr_db:
        high_scale *= 2.0
        noisy_high = np.clip(image + high_scale * base_noise, clip_min, clip_max)

    best_noisy = noisy_high
    for _ in range(iterations):
        mid_scale = 0.5 * (low_scale + high_scale)
        noisy_mid = np.clip(image + mid_scale * base_noise, clip_min, clip_max)
        actual_snr = snr_db(image, noisy_mid)

        best_noisy = noisy_mid
        if actual_snr > target_snr_db:
            low_scale = mid_scale
        else:
            high_scale = mid_scale

    actual_noise_power = float(np.mean((best_noisy - image) ** 2))
    return best_noisy, actual_noise_power


def save_text(text: str, path: Path) -> None:
    """Save text content to a UTF-8 file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
