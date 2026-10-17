
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS_FILE = Path("results/sparsity_sweep.csv")
OUTPUT_FILE = Path("results/sparsity_accuracy.png")

# Group results by pruning method and target sparsity.
groups = defaultdict(list)

with RESULTS_FILE.open("r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        method = row["criterion"]
        sparsity = float(row["actual_sparsity"])
        accuracy = float(row["accuracy_after_finetune_pct"])

        groups[(method, sparsity)].append(accuracy)

methods = {
    "magnitude": "Magnitude pruning",
    "saliency": "Saliency pruning",
}

fig, ax = plt.subplots(figsize=(9, 6))

for method, label in methods.items():
    points = []

    for (group_method, sparsity), accuracies in groups.items():
        if group_method != method:
            continue

        mean_accuracy = sum(accuracies) / len(accuracies)

        if len(accuracies) > 1:
            variance = sum(
                (value - mean_accuracy) ** 2
                for value in accuracies
            ) / (len(accuracies) - 1)
            std_dev = variance ** 0.5
        else:
            std_dev = 0.0

        points.append((sparsity, mean_accuracy, std_dev))

    points.sort(key=lambda point: point[0])

    x = [point[0] for point in points]
    means = [point[1] for point in points]
    errors = [point[2] for point in points]

    ax.errorbar(
        x,
        means,
        yerr=errors,
        marker="o",
        capsize=4,
        linewidth=2,
        label=label,
    )

ax.set(
    title="LeanNet: Accuracy vs. Weight Sparsity",
    xlabel="Actual weight sparsity (%)",
    ylabel="Accuracy after fine-tuning (%)",
)

ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_FILE, dpi=150)
print(f"Plot saved to: {OUTPUT_FILE}")

plt.show()
