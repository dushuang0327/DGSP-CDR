from pathlib import Path
import subprocess
import sys


# Project root:
# DGSP-CDR/
# ├── main.py
# └── scripts/
#     └── reproduce_main.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MAIN_FILE = PROJECT_ROOT / "main.py"


def run_main_experiment():
    """
    Reproduce the main DGSP-CDR five-drug experiment.

    This script uses the experiment settings defined in:
        config/param_config.py
        config/train_params.json

    The command explicitly fixes the main experimental options:
        - method: code_adv
        - metric used for model selection: AUROC
        - cell-line response measurement: AUC
        - 5-fold cross-validation
        - training enabled
        - feature normalization enabled
        - TCGA evaluation
        - hyperparameter search disabled
    """

    if not MAIN_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find main.py at: {MAIN_FILE}"
        )

    command = [
        sys.executable,
        str(MAIN_FILE),

        "--method",
        "code_adv",

        "--metric",
        "auroc",

        "--measurement",
        "AUC",

        "--n",
        "5",

        "--train",

        "--norm",

        "--no-pdtc",

        "--no-hpt",
    ]

    print("=" * 70)
    print("DGSP-CDR Main Experiment Reproduction")
    print("=" * 70)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Python       : {sys.executable}")
    print(f"Main script  : {MAIN_FILE}")

    print("\nCommand:")
    print(" ".join(command))

    print("\nStarting experiment...\n")

    try:
        subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            check=True,
        )

    except subprocess.CalledProcessError as exc:
        print("\n" + "=" * 70)
        print("DGSP-CDR experiment failed.")
        print(f"Return code: {exc.returncode}")
        print("=" * 70)
        sys.exit(exc.returncode)

    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
        sys.exit(130)

    print("\n" + "=" * 70)
    print("DGSP-CDR main experiment completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    run_main_experiment()
