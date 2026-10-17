
"""Leakage-conscious LeanNet benchmark on Breast Cancer Wisconsin.

Run from the repository root:
    python -m train.benchmark_breast_cancer
"""

import csv
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from engine.tensor import Tensor, softmax_cross_entropy
from nn.layers import Linear
from nn.optim import SGD
from nn.pruning import prune_global


SEEDS = [42, 43, 44, 45, 46]
SPARSITIES = [0.50, 0.75, 0.90, 0.95]
EPOCHS = 100
FINETUNE_EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 0.03

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def load_data():
    """Split first, then fit preprocessing on training data only."""
    data = load_breast_cancer()
    X = data.data.astype(np.float64)

    # Make malignant the positive class (1), benign class (0).
    y = (data.target == 0).astype(np.int64)

    # Hold out test data before fitting any preprocessing.
    X_remaining, X_test, y_remaining, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=2026,
    )

    # Split the remaining 80% into 60% train and 20% validation
    # of the full dataset.
    X_train, X_val, y_train, y_val = train_test_split(
        X_remaining,
        y_remaining,
        test_size=0.25,
        stratify=y_remaining,
        random_state=2027,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print("Dataset: Breast Cancer Wisconsin Diagnostic")
    print(f"Train: {X_train.shape}, malignant={y_train.sum()}")
    print(f"Validation: {X_val.shape}, malignant={y_val.sum()}")
    print(f"Test: {X_test.shape}, malignant={y_test.sum()}")
    print("Scaler fitted on training data only.")

    return X_train, y_train, X_val, y_val, X_test, y_test


def forward(layer1, layer2, X):
    x = Tensor(X)
    return layer2(layer1(x).relu())


def parameters_of(layer1, layer2):
    return layer1.parameters() + layer2.parameters()


def build_model(seed):
    np.random.seed(seed)
    layer1 = Linear(30, 16)
    layer2 = Linear(16, 2)
    params = parameters_of(layer1, layer2)
    return layer1, layer2, params


def train_epochs(layer1, layer2, optimizer, X, y, epochs, seed):
    """Mini-batch training using only the supplied training partition."""
    rng = np.random.default_rng(seed)

    for _ in range(epochs):
        indices = rng.permutation(len(X))

        for start in range(0, len(X), BATCH_SIZE):
            batch_idx = indices[start:start + BATCH_SIZE]
            xb = X[batch_idx]
            yb = y[batch_idx]

            optimizer.zero_grad()
            logits = forward(layer1, layer2, xb)
            loss = softmax_cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()


def snapshot(params):
    """Copy parameter values and masks so each experiment starts identically."""
    state = []

    for p in params:
        mask = getattr(p, "mask", None)
        state.append(
            (
                p.value.copy(),
                None if mask is None else mask.copy(),
            )
        )

    return state


def restore(params, state):
    for p, (value, mask) in zip(params, state):
        p.value[...] = value

        if mask is not None:
            p.mask[...] = mask


def prune_by_magnitude(params, target_sparsity):
    """Globally prune the smallest absolute active weights."""
    weight_params = [p for p in params if hasattr(p, "mask")]

    entries = []
    total = sum(p.mask.size for p in weight_params)
    already_pruned = sum(
        int(np.count_nonzero(p.mask == 0)) for p in weight_params
    )
    target_pruned = int(round(target_sparsity * total))
    additional = max(0, target_pruned - already_pruned)

    for p_index, p in enumerate(weight_params):
        for flat_index in np.flatnonzero(p.mask.ravel() != 0):
            entries.append(
                (abs(p.value.ravel()[flat_index]), p_index, flat_index)
            )

    additional = min(additional, len(entries))
    entries.sort(key=lambda item: item[0])

    for _, p_index, flat_index in entries[:additional]:
        p = weight_params[p_index]
        p.mask.ravel()[flat_index] = 0
        p.value.ravel()[flat_index] = 0.0


def actual_sparsity(params):
    weights = [p for p in params if hasattr(p, "mask")]
    total = sum(p.mask.size for p in weights)
    zeros = sum(np.count_nonzero(p.mask == 0) for p in weights)
    return float(zeros / total)


def evaluate(layer1, layer2, X, y):
    """Calculate metrics; class 1 means malignant."""
    logits = forward(layer1, layer2, X).value

    # Stable softmax probabilities.
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    predictions = probabilities.argmax(axis=1)
    malignant_probability = probabilities[:, 1]

    return {
        "accuracy": accuracy_score(y, predictions),
        "malignant_precision": precision_score(
            y, predictions, pos_label=1, zero_division=0
        ),
        "malignant_recall": recall_score(
            y, predictions, pos_label=1, zero_division=0
        ),
        "malignant_f1": f1_score(
            y, predictions, pos_label=1, zero_division=0
        ),
        "roc_auc": roc_auc_score(y, malignant_probability),
    }


def mean_metric(rows, key):
    return float(np.mean([row[key] for row in rows]))


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # Keep every seed's trained configuration until validation-based
    # selection is complete. The test set is not used in this loop.
    saved_states = {}
    validation_rows = []

    configurations = [("dense", 0.0)]
    for method in ("magnitude", "saliency"):
        configurations.extend((method, s) for s in SPARSITIES)

    for seed in SEEDS:
        layer1, layer2, params = build_model(seed)
        optimizer = SGD(params, lr=LEARNING_RATE)

        train_epochs(
            layer1, layer2, optimizer,
            X_train, y_train, EPOCHS, seed,
        )
        dense_state = snapshot(params)

        # Evaluate dense validation performance.
        restore(params, dense_state)
        dense_val = evaluate(layer1, layer2, X_val, y_val)
        validation_rows.append({
            "seed": seed,
            "method": "dense",
            "target_sparsity": 0.0,
            "actual_sparsity": actual_sparsity(params),
            **dense_val,
        })
        saved_states[("dense", 0.0, seed)] = snapshot(params)

        for method, sparsity in configurations[1:]:
            restore(params, dense_state)

            # Compute saliency from TRAINING data only.
            if method == "saliency":
                optimizer.zero_grad()
                train_logits = forward(layer1, layer2, X_train)
                train_loss = softmax_cross_entropy(train_logits, y_train)
                train_loss.backward()
                prune_global(params, sparsity)
            else:
                prune_by_magnitude(params, sparsity)

            # Rebuild the optimizer for a clean fine-tuning run.
            optimizer = SGD(params, lr=LEARNING_RATE)
            train_epochs(
                layer1,
                layer2,
                optimizer,
                X_train,
                y_train,
                FINETUNE_EPOCHS,
                seed + 1000 + int(sparsity * 100),
            )

            # Explicitly check the pruning invariant.
            for p in params:
                if hasattr(p, "mask"):
                    assert np.all(p.value[p.mask == 0] == 0.0)

            val_metrics = evaluate(layer1, layer2, X_val, y_val)
            validation_rows.append({
                "seed": seed,
                "method": method,
                "target_sparsity": sparsity,
                "actual_sparsity": actual_sparsity(params),
                **val_metrics,
            })
            saved_states[(method, sparsity, seed)] = snapshot(params)

        print(f"Completed training seed {seed}.")

    # Aggregate validation metrics across seeds.
    grouped = {}
    for row in validation_rows:
        key = (row["method"], row["target_sparsity"])
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (method, sparsity), rows in grouped.items():
        summary_rows.append({
            "method": method,
            "target_sparsity": sparsity,
            "mean_actual_sparsity": mean_metric(
                rows, "actual_sparsity"
            ),
            "val_accuracy": mean_metric(rows, "accuracy"),
            "val_malignant_precision": mean_metric(
                rows, "malignant_precision"
            ),
            "val_malignant_recall": mean_metric(
                rows, "malignant_recall"
            ),
            "val_malignant_f1": mean_metric(rows, "malignant_f1"),
            "val_roc_auc": mean_metric(rows, "roc_auc"),
        })

    summary_rows.sort(key=lambda r: r["val_malignant_f1"], reverse=True)

    validation_path = RESULTS_DIR / "breast_cancer_validation.csv"
    with validation_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nValidation results (ranked by malignant F1):")
    for row in summary_rows:
        print(
            f"{row['method']:9s} "
            f"sparsity={row['mean_actual_sparsity']:.1%} "
            f"F1={row['val_malignant_f1']:.4f} "
            f"recall={row['val_malignant_recall']:.4f} "
            f"AUC={row['val_roc_auc']:.4f}"
        )

    # Select the best pruned configuration using validation only.
    pruned_rows = [r for r in summary_rows if r["method"] != "dense"]
    selected = max(pruned_rows, key=lambda r: r["val_malignant_f1"])
    selected_key = (selected["method"], selected["target_sparsity"])

    print(
        "\nSelected using validation only:",
        selected_key,
        f"(mean validation malignant F1="
        f"{selected['val_malignant_f1']:.4f})",
    )

    # Final test evaluation: dense baseline and selected pruned model.
    # No configuration changes are made after seeing these test results.
    test_rows = []

    final_configs = [
        ("dense", 0.0),
        selected_key,
    ]

    for method, sparsity in final_configs:
        seed_metrics = []

        for seed in SEEDS:
            layer1, layer2, params = build_model(seed)
            restore(params, saved_states[(method, sparsity, seed)])

            metrics = evaluate(layer1, layer2, X_test, y_test)
            seed_metrics.append(metrics)

        test_rows.append({
            "method": method,
            "target_sparsity": sparsity,
            "actual_sparsity": (
                0.0 if method == "dense"
                else selected["mean_actual_sparsity"]
            ),
            **{
                f"test_{metric}": mean_metric(seed_metrics, metric)
                for metric in seed_metrics[0]
            },
        })

    test_path = RESULTS_DIR / "breast_cancer_test.csv"
    with test_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=test_rows[0].keys())
        writer.writeheader()
        writer.writerows(test_rows)

    print("\nFinal held-out test results (mean across seeds):")
    for row in test_rows:
        print(
            f"{row['method']:9s} "
            f"sparsity={row['actual_sparsity']:.1%} "
            f"accuracy={row['test_accuracy']:.4f} "
            f"malignant_recall={row['test_malignant_recall']:.4f} "
            f"malignant_F1={row['test_malignant_f1']:.4f} "
            f"AUC={row['test_roc_auc']:.4f}"
        )

    print(f"\nSaved validation summary: {validation_path}")
    print(f"Saved final test results: {test_path}")


if __name__ == "__main__":
    main()
