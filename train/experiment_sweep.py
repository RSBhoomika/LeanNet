
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from engine.tensor import Tensor, softmax_cross_entropy
from nn.layers import Linear
from nn.optim import SGD
from nn.pruning import prune_global
from train.train_mlp import (
    make_dataset,
    evaluate,
    train_epoch,
    refresh_importance,
    forward,
)


OUTPUT_DIR = Path("results")
SEEDS = list(range(42, 52))
SPARSITIES = [0.0, 0.50, 0.75, 0.90, 0.95]
BASELINE_EPOCHS = 100
FINE_TUNE_EPOCHS = 10
BATCH_SIZE = 32


def build_model(seed):
    """Create a reproducible model and its SGD optimizer."""
    np.random.seed(seed)

    layer1 = Linear(2, 16)
    layer2 = Linear(16, 2)
    parameters = layer1.parameters() + layer2.parameters()

    optimizer = SGD(parameters, lr=0.1)
    return layer1, layer2, parameters, optimizer


def count_weight_entries(parameters):
    """Count weight entries represented by masks."""
    return sum(
        p.mask.size for p in parameters if hasattr(p, "mask")
    )


def calculate_sparsity(parameters):
    """Calculate the fraction of weight entries that are masked."""
    total = count_weight_entries(parameters)
    zeros = sum(
        int(np.count_nonzero(p.mask == 0))
        for p in parameters
        if hasattr(p, "mask")
    )
    return zeros / total if total else 0.0


def prune_by_magnitude(parameters, target_sparsity):
    """Globally prune the smallest-magnitude active weights."""
    weights = [p for p in parameters if hasattr(p, "mask")]

    total = sum(p.mask.size for p in weights)
    target_zeros = int(round(total * target_sparsity))
    current_zeros = sum(
        int(np.count_nonzero(p.mask == 0)) for p in weights
    )
    count = max(0, target_zeros - current_zeros)

    candidates = []
    for layer_index, p in enumerate(weights):
        for index in np.flatnonzero(p.mask.ravel() == 1):
            score = abs(float(p.value.ravel()[index]))
            candidates.append((score, layer_index, int(index)))

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    if count > len(candidates):
        raise ValueError("Requested sparsity exceeds available weights.")

    for _, layer_index, index in candidates[:count]:
        p = weights[layer_index]
        p.mask.ravel()[index] = 0.0
        p.value.ravel()[index] = 0.0


def save_parameters(parameters):
    """Snapshot parameter values after dense training."""
    return [p.value.copy() for p in parameters]


def restore_parameters(parameters, snapshot):
    """Restore dense-trained values into a fresh model."""
    for p, saved_value in zip(parameters, snapshot):
        p.value[...] = saved_value

        if hasattr(p, "mask"):
            p.mask[...] = 1.0


def train_dense(seed, X_train, y_train):
    """Train one dense baseline and return its weights and metrics."""
    layer1, layer2, parameters, optimizer = build_model(seed)

    # Keep data shuffling reproducible and separate from initialization.
    np.random.seed(seed + 1000)

    for _ in range(BASELINE_EPOCHS):
        train_epoch(
            layer1, layer2, optimizer, X_train, y_train,
            batch_size=BATCH_SIZE,
        )

    snapshot = save_parameters(parameters)
    return snapshot, layer1, layer2, parameters


def refresh_absolute_batch_gradients(
    layer1, layer2, parameters, X, y, batch_size=32
):
    """Estimate importance using mean absolute mini-batch gradients."""
    gradient_sums = [
        np.zeros_like(p.value) for p in parameters
    ]
    batch_count = 0

    # Use deterministic, non-shuffled batches for importance estimation.
    for start in range(0, len(X), batch_size):
        X_batch = X[start:start + batch_size]
        y_batch = y[start:start + batch_size]

        for p in parameters:
            p.grad[...] = 0.0

        logits = forward(layer1, layer2, X_batch)
        loss = softmax_cross_entropy(logits, y_batch)
        loss.backward()

        for i, p in enumerate(parameters):
            gradient_sums[i] += np.abs(p.grad)

        batch_count += 1

    if batch_count == 0:
        raise ValueError("Cannot calculate importance on empty data.")

    for p, total in zip(parameters, gradient_sums):
        p.grad[...] = total / batch_count


def run_experiment(
    seed, criterion, target_sparsity, snapshot, X_train, y_train, X_test, y_test
):
    """Evaluate one criterion and sparsity, starting from identical weights."""
    layer1, layer2, parameters, optimizer = build_model(seed + 10000)
    restore_parameters(parameters, snapshot)

    baseline_loss, baseline_accuracy = evaluate(
        layer1, layer2, X_test, y_test
    )

    if target_sparsity > 0:
        if criterion == "saliency":
            refresh_absolute_batch_gradients(
                layer1, layer2, parameters, X_train, y_train
            )
            prune_global(parameters, target_sparsity)
        elif criterion == "magnitude":
            prune_by_magnitude(parameters, target_sparsity)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

    actual_sparsity = calculate_sparsity(parameters)
    pruned_loss, pruned_accuracy = evaluate(
        layer1, layer2, X_test, y_test
    )

    # Reset the batch-order RNG so the methods receive comparable
    # mini-batch orderings during fine-tuning.
    np.random.seed(seed + 20000)

    for _ in range(FINE_TUNE_EPOCHS):
        train_epoch(
            layer1, layer2, optimizer, X_train, y_train,
            batch_size=BATCH_SIZE,
        )

    final_loss, final_accuracy = evaluate(
        layer1, layer2, X_test, y_test
    )

    return {
        "seed": seed,
        "criterion": criterion,
        "target_sparsity": target_sparsity,
        "actual_sparsity": actual_sparsity,
        "dense_accuracy_pct": baseline_accuracy,
        "accuracy_before_finetune_pct": pruned_accuracy,
        "accuracy_after_finetune_pct": final_accuracy,
        "dense_test_loss": baseline_loss,
        "loss_before_finetune": pruned_loss,
        "loss_after_finetune": final_loss,
        "weight_entries": count_weight_entries(parameters),
        "zero_weight_entries": sum(
            int(np.count_nonzero(p.mask == 0))
            for p in parameters if hasattr(p, "mask")
        ),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y = make_dataset(n_samples=1200, seed=42)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    rows = []

    for seed in SEEDS:
        print(f"\n{'=' * 60}\nSeed {seed}: training dense baseline")

        snapshot, _, _, _ = train_dense(seed, X_train, y_train)

        for criterion in ("saliency", "magnitude"):
            for target in SPARSITIES:
                result = run_experiment(
                    seed, criterion, target, snapshot,
                    X_train, y_train, X_test, y_test,
                )
                rows.append(result)

                print(
                    f"{criterion:9s} | target={target:5.0%} | "
                    f"actual={result['actual_sparsity']:6.2%} | "
                    f"before={result['accuracy_before_finetune_pct']:6.2f}% | "
                    f"after={result['accuracy_after_finetune_pct']:6.2f}%"
                )

    csv_path = OUTPUT_DIR / "sparsity_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Plot mean accuracy and standard deviation across the three seeds.
    fig, ax = plt.subplots(figsize=(8, 5))

    for criterion in ("saliency", "magnitude"):
        means = []
        stds = []

        for target in SPARSITIES:
            values = [
                row["accuracy_after_finetune_pct"]
                for row in rows
                if row["criterion"] == criterion
                and row["target_sparsity"] == target
            ]
            means.append(float(np.mean(values)))
            stds.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)

        ax.errorbar(
            np.array(SPARSITIES) * 100,
            means,
            yerr=stds,
            marker="o",
            capsize=4,
            label=criterion,
        )

    ax.set_xlabel("Target weight sparsity (%)")
    ax.set_ylabel("Test accuracy after fine-tuning (%)")
    ax.set_title("LeanNet: Accuracy vs. Weight Sparsity")
    ax.set_xticks(np.array(SPARSITIES) * 100)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    plot_path = OUTPUT_DIR / "sparsity_accuracy.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    print(f"\nSaved raw results: {csv_path}")
    print(f"Saved Pareto-style plot: {plot_path}")


if __name__ == "__main__":
    main()
