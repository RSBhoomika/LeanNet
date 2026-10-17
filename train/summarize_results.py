
import csv
import statistics
from collections import defaultdict
from pathlib import Path

RESULTS_FILE = Path("results/sparsity_sweep.csv")

groups = defaultdict(list)

with RESULTS_FILE.open("r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        key = (
            row["criterion"],
            float(row["target_sparsity"]),
        )
        groups[key].append(row)

print("\nPRUNING EXPERIMENT SUMMARY")
print("=" * 78)
print(
    f"{'Method':<12} {'Target':>8} {'Actual':>8} "
    f"{'Accuracy mean':>15} {'Std dev':>10} {'Mean loss':>12}"
)
print("-" * 78)

for (method, target), rows in sorted(groups.items()):
    accuracies = [
        float(row["accuracy_after_finetune_pct"])
        for row in rows
    ]
    actual_sparsities = [
        float(row["actual_sparsity"]) * 100
        for row in rows
    ]
    losses = [
        float(row["loss_after_finetune"])
        for row in rows
    ]

    mean_accuracy = statistics.mean(accuracies)
    std_accuracy = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
    mean_sparsity = statistics.mean(actual_sparsities)
    mean_loss = statistics.mean(losses)

    print(
        f"{method:<12} {target * 100:>7.0f}% "
        f"{mean_sparsity:>7.2f}% "
        f"{mean_accuracy:>12.2f}% "
        f"{std_accuracy:>9.2f} "
        f"{mean_loss:>12.4f}"
    )

print("=" * 78)
print("Accuracy and standard deviation are measured in percentage points.")
print(f"Seeds per configuration: {len(rows)} (expected: 10)")
