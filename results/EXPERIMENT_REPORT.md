# LeanNet: Weight Pruning Experiment Report

## Objective

Evaluate how weight sparsity affects model accuracy and compare magnitude-based pruning with saliency-based pruning.

## Experimental setup

- **Task:** Synthetic binary classification
- **Model:** Multilayer perceptron (2 → 16 → 2)
- **Baseline training:** 100 epochs
- **Fine-tuning after pruning:** 10 epochs
- **Random seeds:** 42–51 (10 seeds)
- **Pruning methods:** Magnitude and saliency
- **Target sparsities:** 0%, 50%, 75%, 90%, 95%
- **Evaluation metric:** Accuracy after fine-tuning

Magnitude pruning removes weights with the smallest absolute values. Saliency pruning uses the absolute weight-gradient product, `|weight × gradient|`, to estimate weight importance.

## Results

| Method | Target sparsity | Actual sparsity | Mean accuracy | Std. dev. |
|---|---:|---:|---:|---:|
| Magnitude | 0% | 0.00% | 98.62% | 0.28 |
| Saliency | 0% | 0.00% | 98.62% | 0.28 |
| Magnitude | 50% | 50.00% | 98.25% | 0.47 |
| Saliency | 50% | 50.00% | 98.38% | 0.31 |
| Magnitude | 75% | 75.00% | 88.83% | 10.93 |
| Saliency | 75% | 75.00% | 90.38% | 9.26 |
| Magnitude | 90% | 90.62% | 64.88% | 6.60 |
| Saliency | 90% | 90.62% | 66.25% | 9.10 |
| Magnitude | 95% | 95.31% | 52.08% | 7.29 |
| Saliency | 95% | 95.31% | 50.75% | 6.88 |

Accuracy and standard deviation are expressed in percentage points.

## Paired method comparison

The mean per-seed accuracy difference (saliency minus magnitude) was:

| Target sparsity | Mean difference |
|---:|---:|
| 0% | +0.00 percentage points |
| 50% | +0.13 percentage points |
| 75% | +1.54 percentage points |
| 90% | +1.38 percentage points |
| 95% | -1.33 percentage points |

These differences are descriptive and do not establish statistical significance. The standard deviation of paired differences was 0.48, 8.44, 5.40, and 9.14 percentage points at 50%, 75%, 90%, and 95% sparsity, respectively.

## Observations

1. At 50% sparsity, both methods retain accuracy close to the dense baseline.
2. At 75% sparsity, mean accuracy decreases and variability across seeds increases.
3. At 90% and 95% sparsity, accuracy degrades substantially.
4. Saliency has a modest average advantage at 75% and 90%, but the results vary across seeds.
5. Neither method demonstrates reliable performance