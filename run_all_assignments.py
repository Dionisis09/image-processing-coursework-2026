import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
SCRIPTS = [
    "askisi1_erotima1.py",
    "askisi2_histogram_equalization.py",
    "askisi3_denoising.py",
    "askisi4_restoration.py",
    "askisi5_dct_compression.py",
    "askisi6_edges.py",
]


def main() -> None:
    for script_name in SCRIPTS:
        script_path = ROOT_DIR / script_name
        print(f"\n=== Running {script_name} ===", flush=True)
        completed = subprocess.run([sys.executable, str(script_path)], check=False)
        if completed.returncode != 0:
            raise SystemExit(f"{script_name} failed with exit code {completed.returncode}")

    print("\nAll scripts finished successfully.", flush=True)


if __name__ == "__main__":
    main()
