from pathlib import Path
import sys
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# DGSP-CDR modules
# ============================================================

from src.mlp import MLP
from src.encoder_decoder_final import EncoderDecoder_basis
from src.cut_stats_subset import select_data


# ============================================================
# Smoke-test configuration
# ============================================================

SEED = 1795

LATENT_DIM = 8
NUM_BASIS = 5
NUM_ENSEMBLE = 5

SMOKE_DATA_DIR = PROJECT_ROOT / "scripts" / "smoke_data"
FEATURE_FILE = SMOKE_DATA_DIR / "gene_features.csv"
LABEL_FILE = SMOKE_DATA_DIR / "labels.csv"


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_finite(name: str, tensor: torch.Tensor):
    if not torch.isfinite(tensor).all():
        raise RuntimeError(
            f"{name} contains NaN or Inf values."
        )


def load_smoke_data(device):
    """
    Load the lightweight smoke-test dataset from:

        examples/smoke_data/gene_features.csv
        examples/smoke_data/labels.csv
    """

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find smoke-test feature file: {FEATURE_FILE}"
        )

    if not LABEL_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find smoke-test label file: {LABEL_FILE}"
        )

    feature_df = pd.read_csv(
        FEATURE_FILE,
        index_col=0
    )

    label_df = pd.read_csv(
        LABEL_FILE
    )

    if "sample" not in label_df.columns:
        raise ValueError(
            "labels.csv must contain a 'sample' column."
        )

    if "label" not in label_df.columns:
        raise ValueError(
            "labels.csv must contain a 'label' column."
        )

    feature_samples = feature_df.index.astype(str).tolist()
    label_samples = label_df["sample"].astype(str).tolist()

    if feature_samples != label_samples:
        raise ValueError(
            "Sample order in gene_features.csv and labels.csv does not match."
        )

    if feature_df.isnull().any().any():
        raise ValueError(
            "gene_features.csv contains missing values."
        )

    if label_df["label"].isnull().any():
        raise ValueError(
            "labels.csv contains missing labels."
        )

    labels = label_df["label"].astype(int).values

    if not np.isin(labels, [0, 1]).all():
        raise ValueError(
            "Smoke-test labels must be binary values 0 or 1."
        )

    x = torch.tensor(
        feature_df.values,
        dtype=torch.float32,
        device=device
    )

    y = torch.tensor(
        labels,
        dtype=torch.float32,
        device=device
    ).unsqueeze(1)

    return x, y, feature_df


# ============================================================
# Test 1: representation + basis projection + classifier
# ============================================================

def test_representation_and_classifier():
    """
    Verify the basic DGSP-CDR execution path:

        smoke-test gene-expression features
        -> MLP encoder
        -> basis projection
        -> classifier
        -> loss computation
        -> backward propagation

    This is an execution test only and does not reproduce
    manuscript performance.
    """

    print("[1/3] Testing representation and classifier...")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # --------------------------------------------------------
    # Load smoke-test dataset
    # --------------------------------------------------------

    x, y, feature_df = load_smoke_data(
        device
    )

    num_samples = x.shape[0]
    input_dim = x.shape[1]

    if num_samples < 2:
        raise RuntimeError(
            "Smoke-test dataset must contain at least 2 samples."
        )

    if input_dim < 1:
        raise RuntimeError(
            "Smoke-test dataset contains no features."
        )

    # --------------------------------------------------------
    # Encoder
    # --------------------------------------------------------

    encoder = MLP(
        input_dim=input_dim,
        output_dim=LATENT_DIM,
        hidden_dims=[16],
        dop=0.1
    ).to(device)

    # --------------------------------------------------------
    # Lightweight basis representations
    # --------------------------------------------------------

    basis_vec = torch.randn(
        NUM_BASIS,
        LATENT_DIM,
        dtype=torch.float32,
        device=device
    )

    # --------------------------------------------------------
    # Binary response classifier
    # --------------------------------------------------------

    decoder = MLP(
        input_dim=LATENT_DIM,
        output_dim=1,
        hidden_dims=[8],
        dop=0.1
    ).to(device)

    classifier = EncoderDecoder_basis(
        encoder=encoder,
        decoder=decoder,
        basis_vec=basis_vec,
        testing_drug_len=NUM_BASIS,
        inv_temp=0.1,
        normalize_flag=True,
        cosine_flag=True
    ).to(device)

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    logits, fused_features = classifier(
        x
    )

    expected_logit_shape = (
        num_samples,
        1
    )

    expected_feature_shape = (
        num_samples,
        LATENT_DIM
    )

    if tuple(logits.shape) != expected_logit_shape:
        raise RuntimeError(
            f"Unexpected classifier output shape: "
            f"{tuple(logits.shape)}; "
            f"expected {expected_logit_shape}."
        )

    if tuple(fused_features.shape) != expected_feature_shape:
        raise RuntimeError(
            f"Unexpected fused-feature shape: "
            f"{tuple(fused_features.shape)}; "
            f"expected {expected_feature_shape}."
        )

    check_finite(
        "Classifier logits",
        logits
    )

    check_finite(
        "Fused features",
        fused_features
    )

    # --------------------------------------------------------
    # One optimization step
    # --------------------------------------------------------

    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=1e-4
    )

    optimizer.zero_grad()

    loss = loss_fn(
        logits,
        y
    )

    check_finite(
        "Classification loss",
        loss
    )

    loss.backward()

    optimizer.step()

    print(
        f"      PASS "
        f"(samples={num_samples}, "
        f"features={input_dim}, "
        f"loss={loss.item():.6f}, "
        f"device={device})"
    )

    return (
        fused_features
        .detach()
        .cpu()
        .numpy()
    )


# ============================================================
# Test 2: ensemble pseudo-label selection
# ============================================================

def test_pseudo_label_selection(features: np.ndarray):
    """
    Verify the selective pseudo-labeling component:

        multiple classifier outputs
        -> confidence thresholding
        -> ensemble voting
        -> neighborhood-based selection
        -> selected pseudo-labels
    """

    print("[2/3] Testing pseudo-label selection...")

    num_samples = features.shape[0]

    if num_samples <= 10:
        raise RuntimeError(
            "Smoke-test dataset must contain more than 10 samples "
            "for pseudo-label selection."
        )

    ensemble_outputs = []

    # --------------------------------------------------------
    # Construct lightweight outputs from five classifiers
    # --------------------------------------------------------

    for classifier_id in range(
        NUM_ENSEMBLE
    ):

        prediction_dict = {}

        for index in range(
            num_samples
        ):

            if index < num_samples // 2:
                probability = (
                    0.20
                    + classifier_id * 0.005
                )
            else:
                probability = (
                    0.80
                    - classifier_id * 0.005
                )

            prediction_dict[index] = {
                "fet": features[index],
                "prob": probability
            }

        ensemble_outputs.append(
            prediction_dict
        )

    # --------------------------------------------------------
    # Run DGSP-CDR pseudo-label selection
    # --------------------------------------------------------

    selected_indices, selected_labels = select_data(
        ensemble_outputs,
        class_0_th=0.4,
        class_1_th=0.6,
        budget=0.5,
        K=5
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if len(selected_indices) == 0:
        raise RuntimeError(
            "Pseudo-label selection returned no samples."
        )

    if len(selected_indices) != len(selected_labels):
        raise RuntimeError(
            "The numbers of selected indices and "
            "pseudo-labels do not match."
        )

    if isinstance(
        selected_labels,
        torch.Tensor
    ):
        labels = (
            selected_labels
            .detach()
            .cpu()
            .numpy()
        )
    else:
        labels = np.asarray(
            selected_labels
        )

    if not np.isin(
        labels,
        [0, 1]
    ).all():
        raise RuntimeError(
            "Invalid pseudo-label detected."
        )

    num_negative = int(
        np.sum(labels == 0)
    )

    num_positive = int(
        np.sum(labels == 1)
    )

    print(
        f"      PASS "
        f"(selected={len(selected_indices)}, "
        f"negative={num_negative}, "
        f"positive={num_positive})"
    )


# ============================================================
# Test 3: environment
# ============================================================

def test_environment():
    """
    Print basic environment information.
    """

    print("[3/3] Checking environment...")

    print(
        f"      Python: "
        f"{sys.version.split()[0]}"
    )

    print(
        f"      PyTorch: "
        f"{torch.__version__}"
    )

    print(
        f"      NumPy: "
        f"{np.__version__}"
    )

    print(
        f"      pandas: "
        f"{pd.__version__}"
    )

    print(
        f"      CUDA available: "
        f"{torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():
        print(
            f"      GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print("      PASS")


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("DGSP-CDR Lightweight Smoke Test")
    print("=" * 70)

    print(
        f"Smoke-test data: {SMOKE_DATA_DIR}"
    )

    print(
        "This test verifies basic execution of the core pipeline."
    )

    print(
        "It does not reproduce the full experimental results."
    )

    print()

    set_seed(
        SEED
    )

    try:

        # ----------------------------------------------------
        # Representation + classifier
        # ----------------------------------------------------

        features = (
            test_representation_and_classifier()
        )

        # ----------------------------------------------------
        # Pseudo-label selection
        # ----------------------------------------------------

        test_pseudo_label_selection(
            features
        )

        # ----------------------------------------------------
        # Environment
        # ----------------------------------------------------

        test_environment()

    except Exception as exc:

        print()
        print("=" * 70)
        print("SMOKE TEST FAILED")
        print("=" * 70)

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        sys.exit(1)

    print()
    print("=" * 70)
    print("SMOKE TEST PASSED")
    print("=" * 70)

    print(
        "Core representation, classification, "
        "and pseudo-label selection components "
        "executed successfully."
    )


if __name__ == "__main__":
    main()
