import json
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter

from assignment_utils import (
    IMAGES_DIR,
    add_gaussian_noise_with_exact_snr,
    ensure_output_dir,
    load_grayscale_image,
    mse,
    psnr,
    save_grayscale_image,
    save_text,
    snr_db,
)


OUTPUT_DIR = ensure_output_dir("askisi4_restoration")
INPUT_DIR = IMAGES_DIR / "Ασκηση 4"
IMAGE_PATH = INPUT_DIR / "new_york.png"
PSF_P_PATH = INPUT_DIR / "psf.p"
TARGET_SNR_DB = 10.0
WIENER_WINDOW = (5, 5)
RNG = np.random.default_rng(123)


def add_gaussian_noise_for_snr(image: np.ndarray, snr_db: float, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Add zero-mean Gaussian noise to reach a target SNR."""
    return add_gaussian_noise_with_exact_snr(image, snr_db, rng, 0.0, 255.0)


def estimate_noise_power(noisy: np.ndarray) -> float:
    """Estimate additive noise power from a high-frequency residual."""
    smooth = gaussian_filter(noisy, sigma=1.5, mode="reflect")
    residual = noisy - smooth
    sigma_hat = np.median(np.abs(residual - np.median(residual))) / 0.6745
    return float(max(sigma_hat**2, 1e-12))


def wiener_filter_frequency(noisy: np.ndarray, noise_power: float) -> np.ndarray:
    """Apply Wiener denoising in the frequency domain for additive noise (H=1)."""
    rows, cols = noisy.shape
    noisy_freq = np.fft.fft2(noisy)
    noisy_psd = (np.abs(noisy_freq) ** 2) / (rows * cols)
    signal_psd = np.maximum(noisy_psd - noise_power, 1e-12)
    wiener_gain = signal_psd / (signal_psd + noise_power)
    restored = np.real(np.fft.ifft2(wiener_gain * noisy_freq))
    return np.clip(restored, 0.0, 255.0)


def save_wiener_figure(original: np.ndarray, noisy: np.ndarray, known: np.ndarray, unknown: np.ndarray, path: Path) -> None:
    """Save a summary figure for Wiener filtering."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    panels = [
        ("Original", original),
        ("Noisy (SNR=10 dB)", noisy),
        ("Wiener - Known Noise", known),
        ("Wiener - Unknown Noise", unknown),
    ]

    for ax, (title, image) in zip(axes, panels):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def matlab_export_script_text(work_dir: Path, output_dir: Path) -> str:
    """Create the MATLAB helper script that exports blur data from psf.p."""
    work_dir_posix = work_dir.as_posix()
    output_dir_posix = output_dir.as_posix()
    image_path_posix = (work_dir / "new_york.png").as_posix()

    return f"""addpath('{work_dir_posix}');
I = im2double(imread('{image_path_posix}'));
delta = zeros(size(I));
delta(1,1) = 1;
H_impulse = psf(delta);
blurred = psf(I);
save('{output_dir_posix}/psf_export.mat', 'H_impulse', 'blurred');
disp('Saved psf_export.mat');
"""


def write_matlab_helper() -> Path:
    """Write the MATLAB helper script to disk."""
    helper_path = OUTPUT_DIR / "askisi4_export_psf_from_matlab.m"
    save_text(matlab_export_script_text(INPUT_DIR, OUTPUT_DIR), helper_path)
    return helper_path


def maybe_run_matlab_helper(helper_path: Path) -> str:
    """Try to run the MATLAB helper if MATLAB is available."""
    matlab_path = shutil.which("matlab")
    if matlab_path is None:
        fallback_paths = [
            Path("C:/Program Files/MATLAB/R2026a/bin/matlab.exe"),
            Path("C:/Program Files/MATLAB/R2026a/bin/win64/MATLAB.exe"),
        ]
        for candidate in fallback_paths:
            if candidate.exists():
                matlab_path = str(candidate)
                break

    if matlab_path is None:
        return "MATLAB not found on this machine. Part B skipped after generating helper script."

    command = [matlab_path, "-batch", f"run('{helper_path.as_posix()}')"]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        return f"MATLAB execution failed: {exc.stderr.strip() or exc.stdout.strip()}"

    return "MATLAB helper executed successfully."


def load_psf_export_if_available() -> tuple[np.ndarray, np.ndarray] | None:
    """Load MATLAB-exported impulse response and blurred image if present."""
    export_path = OUTPUT_DIR / "psf_export.mat"
    if not export_path.exists():
        return None

    data = loadmat(export_path)
    impulse = np.asarray(data["H_impulse"], dtype=np.float64)
    blurred = np.asarray(data["blurred"], dtype=np.float64)
    return impulse, blurred


def inverse_filter_with_threshold(blurred: np.ndarray, impulse: np.ndarray, threshold: float) -> np.ndarray:
    """Restore a blurred image using thresholded inverse filtering."""
    h_freq = np.fft.fft2(impulse)
    g_freq = np.fft.fft2(blurred)

    mask = np.abs(h_freq) >= threshold
    restored_freq = np.zeros_like(g_freq, dtype=np.complex128)
    restored_freq[mask] = g_freq[mask] / h_freq[mask]
    restored = np.real(np.fft.ifft2(restored_freq))
    return np.clip(restored, 0.0, 1.0)


def save_inverse_filter_curve(thresholds: list[float], errors: list[float], path: Path) -> None:
    """Save the MSE-vs-threshold curve."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(thresholds, errors, marker="o")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("MSE")
    ax.set_title("Inverse Filtering Threshold Sweep")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def threshold_candidates_from_frequency_response(h_freq: np.ndarray) -> list[float]:
    """Build a threshold sweep that actually changes the inverse-filter mask."""
    abs_h = np.abs(h_freq)
    quantiles = np.quantile(abs_h, np.linspace(0.0, 0.95, 11))
    thresholds = sorted({float(q) for q in quantiles})
    if thresholds[0] != 0.0:
        thresholds.insert(0, 0.0)
    return thresholds


def main() -> None:
    original = load_grayscale_image(IMAGE_PATH)
    noisy, noise_power = add_gaussian_noise_for_snr(original, TARGET_SNR_DB, RNG)
    estimated_noise_power = estimate_noise_power(noisy)

    known_noise_wiener = wiener_filter_frequency(noisy, noise_power)
    unknown_noise_wiener = wiener_filter_frequency(noisy, estimated_noise_power)

    save_grayscale_image(original, OUTPUT_DIR / "new_york_original.png")
    save_grayscale_image(noisy, OUTPUT_DIR / "new_york_noisy_snr10.png")
    save_grayscale_image(known_noise_wiener, OUTPUT_DIR / "wiener_known_noise.png")
    save_grayscale_image(unknown_noise_wiener, OUTPUT_DIR / "wiener_unknown_noise.png")
    save_wiener_figure(original, noisy, known_noise_wiener, unknown_noise_wiener, OUTPUT_DIR / "wiener_summary.png")

    helper_path = write_matlab_helper()
    matlab_status = maybe_run_matlab_helper(helper_path)

    summary: dict[str, object] = {
        "target_snr_db": TARGET_SNR_DB,
        "noise_power": noise_power,
        "actual_snr_db": snr_db(original, noisy),
        "estimated_noise_power_unknown_case": estimated_noise_power,
        "wiener_window": WIENER_WINDOW,
        "part_a_metrics": {
            "noisy": {"mse": mse(original, noisy), "psnr_db": psnr(original, noisy, data_range=255.0)},
            "known_noise_wiener": {
                "mse": mse(original, known_noise_wiener),
                "psnr_db": psnr(original, known_noise_wiener, data_range=255.0),
            },
            "unknown_noise_wiener": {
                "mse": mse(original, unknown_noise_wiener),
                "psnr_db": psnr(original, unknown_noise_wiener, data_range=255.0),
            },
        },
        "part_b_status": matlab_status,
        "psf_helper_script": str(helper_path),
        "psf_p_path": str(PSF_P_PATH),
    }

    psf_export = load_psf_export_if_available()
    if psf_export is not None:
        summary["part_b_status"] = "MATLAB export loaded successfully and Part B was completed."
        impulse, blurred = psf_export
        save_grayscale_image(np.clip(blurred * 255.0, 0, 255), OUTPUT_DIR / "blurred_from_psf.png")
        save_grayscale_image(np.clip(impulse * 255.0, 0, 255), OUTPUT_DIR / "estimated_impulse_response.png")

        h_freq = np.fft.fft2(impulse)
        h_mag = np.log1p(np.abs(np.fft.fftshift(h_freq)))
        save_grayscale_image((255.0 * (h_mag - h_mag.min()) / (h_mag.max() - h_mag.min() + 1e-12)), OUTPUT_DIR / "frequency_response_log.png")

        thresholds = threshold_candidates_from_frequency_response(h_freq)
        threshold_errors = []
        best_threshold = None
        best_restored = None
        best_error = float("inf")
        no_threshold_error = None

        original_unit = original / 255.0
        blurred_unit = np.clip(blurred, 0.0, 1.0)

        for threshold in thresholds:
            restored = inverse_filter_with_threshold(blurred_unit, impulse, threshold)
            error = mse(original_unit, restored)
            threshold_errors.append(error)
            if threshold == 0.0:
                no_threshold_error = error
                save_grayscale_image(restored * 255.0, OUTPUT_DIR / "inverse_filter_no_threshold.png")
            if error < best_error:
                best_error = error
                best_threshold = float(threshold)
                best_restored = restored

        if best_restored is not None and best_threshold is not None:
            save_grayscale_image(best_restored * 255.0, OUTPUT_DIR / "inverse_filter_best_restored.png")
            save_inverse_filter_curve(list(thresholds), threshold_errors, OUTPUT_DIR / "inverse_filter_threshold_curve.png")
            summary["part_b_inverse_filter"] = {
                "best_threshold": best_threshold,
                "best_mse": best_error,
                "tested_thresholds": [float(x) for x in thresholds],
                "tested_mse": threshold_errors,
                "no_threshold_mse": no_threshold_error,
                "no_threshold_behavior": (
                    "Without thresholding, the inverse filter uses every frequency component. "
                    "For this PSF the smallest |H(u,v)| values are not extremely close to zero, "
                    "so the instability is limited, but thresholding still improves the reconstruction slightly."
                ),
            }

    summary_text = json.dumps(summary, indent=2)
    save_text(summary_text, OUTPUT_DIR / "summary.json")
    print(summary_text)
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
