
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

RESULTS_FILE = Path("results/sparsity_sweep.csv")

# Store accuracy by (seed, sparsity, method).
results = defaultdict(dict)

with RESULTS_FILE.open("r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        seed = int(row["seed"])
        sparsity = float(row["target_sparsity"])
        method = row["criterion"]
        accuracy = float(row["accuracy_after_finetune_pct"])

        results[(seed, sparsity)][method] = accuracy

print("\nPAIRED PRUNING METHOD COMPARISON")
print("=" * 88)
print(
    f"{'Target':>8} {'Magnitude':>12} {'Saliency':>12} "
    f"{'Saliency advantage':>20} {'Std dev':>12} {'Seeds':>7}"
)
print("-" * 88)

for sparsity in sorted({key[1] for key in results}):
    differences = []
    magnitude_scores = []
    saliency_scores = []

    for (seed, target), scores in results.items():
        if target != sparsity:
            continue

        # Compare only seeds that have results for both methods.
        if "magnitude" not in scores or "saliency" not in scores:
            continue

        magnitude_scores.append(scores["magnitude"])
        saliency_scores.append(scores["saliency"])
        differences.append(scores["saliency"] - scores["magnitude"])

    if not differences:
        print(f"{sparsity:>7.0%}  No paired results")
        continue

    avg_difference = mean(differences)
    difference_std = stdev(differences) if len(differences) > 1 else 0.0

    print(
        f"{sparsity:>7.0%} "
        f"{mean(magnitude_scores):>11.2f}% "
        f"{mean(saliency_scores):>11.2f}% "
        f"{avg_difference:>+19.2f} pp "
        f"{difference_std:>11.2f} "
        f"{len(differences):>7}"
    )

print("=" * 88)
print("Positive advantage means saliency performed better.")
print("The standard deviation is across paired per-seed differences.")
