# LeanNet — Experimental Results

## 1. Overview

LeanNet is a lightweight neural network and automatic differentiation framework built to study training, optimization, and weight pruning. These experiments evaluate whether neural networks can retain predictive performance while reducing the number of active connections.

This report covers two experiment groups:

1. **Synthetic experiments** — controlled experiments for studying pruning behavior.
2. **Breast Cancer Wisconsin benchmark** — evaluation on a real-world binary classification dataset using dense training, magnitude pruning, and gradient-based saliency pruning.

The real-world benchmark is intended to evaluate the framework's behavior on a standard tabular dataset. It is an engineering experiment, not a clinically validated diagnostic system.

---

## 2. Experimental Setup

### Dataset

The Breast Cancer Wisconsin Diagnostic dataset was loaded using `sklearn.datasets.load_breast_cancer()`.

| Property | Value |
|---|---:|
| Total samples | 569 |
| Input features | 30 |
| Classes | 2 |
| Malignant samples | 212 |
| Benign samples | 357 |
| Positive class for evaluation | Malignant |

Malignant cases are explicitly treated as the positive class so that precision, recall, and F1 score reflect malignant-case detection.

### Data splitting and preprocessing

The dataset was split into stratified training, validation, and test partitions.

| Partition | Samples | Purpose |
|---|---:|---|
| Training | 341 (60%) | Fit model parameters |
| Validation | 114 (20%) | Compare pruning configurations and select a model |
| Test | 114 (20%) | Final evaluation of selected model |
| **Total** | **569** | |

The data split uses fixed random seeds for reproducibility. `StandardScaler` is fitted only on the training partition; the fitted scaler is then applied to the validation and test partitions. This prevents information from the validation and test features from influencing preprocessing parameters.

### Neural network architecture

The benchmark uses a multilayer perceptron (MLP):

- Input layer: 30 features
- Hidden layer: 16 neurons with ReLU activation
- Output layer: 2 class logits
- Loss: softmax cross-entropy
- Optimizer: stochastic gradient descent (SGD)
- Learning rate: 0.03
- Batch size: 32
- Dense training: 100 epochs
- Pruned-model fine-tuning: 15 epochs
- Initialization seeds: 42, 43, 44, 45, and 46

The dense model trained for each seed provides the starting weights for its corresponding pruning configurations.

---

## 3. Pruning Methods

### Dense baseline

The dense model retains all eligible weight connections and provides the baseline against which pruning is compared.

### Magnitude pruning

Magnitude pruning removes weights with the smallest absolute values:

\[
s_i = |w_i|
\]

Weights with lower scores are pruned first until the requested sparsity target is reached. This is a simple, commonly used pruning baseline.

### Gradient-based saliency pruning

LeanNet's saliency pruning uses the score:

\[
s_i = |w_i \cdot g_i|
\]

where \(w_i\) is a weight and \(g_i\) is its gradient. Weights with lower scores are considered less important according to this criterion and are pruned first.

The saliency scores are computed using training data, not validation or test data.

### Sparsity levels

The experiments evaluate target sparsity levels of 50%, 75%, 90%, and 95% for both pruning methods.

Sparsity represents the proportion of eligible weight connections set to zero. Actual sparsity can differ slightly from the target because weights are discrete; for this network, the highest target produced approximately 94.9% sparsity.

After pruning, the remaining weights are fine-tuned. The framework's pruning masks ensure that pruned weights remain zero during optimization.

---

## 4. Model Selection Protocol

The benchmark uses validation performance to select the pruned configuration.

1. Train dense models using five initialization seeds.
2. Create magnitude-pruned and saliency-pruned variants from the trained dense weights.
3. Fine-tune the pruned variants.
4. Evaluate configurations on the validation partition.
5. Select the configuration with the highest mean malignant-class F1 score.
6. Evaluate the dense baseline and the selected pruned configuration on the held-out test partition.

The selected configuration was **50% gradient-based saliency pruning**. Its selection was based on validation performance; the test results were not used to choose the configuration.

---

## 5. Validation Results

The following table reports the mean malignant-class F1, malignant recall, and ROC-AUC across the five initialization seeds. Results are ordered by mean validation malignant-class F1.

| Method | Actual sparsity | Malignant F1 | Malignant recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Saliency pruning | 50.0% | 0.9723 | 0.9767 | 0.9944 |
| Magnitude pruning | 75.0% | 0.9676 | 0.9674 | 0.9937 |
| Dense baseline | 0.0% | 0.9610 | 0.9721 | 0.9942 |
| Saliency pruning | 75.0% | 0.9567 | 0.9674 | 0.9925 |
| Magnitude pruning | 50.0% | 0.9547 | 0.9721 | 0.9952 |
| Magnitude pruning | 90.0% | 0.9492 | 0.9488 | 0.9902 |
| Saliency pruning | 90.0% | 0.9183 | 0.9163 | 0.9842 |
| Magnitude pruning | 94.9% | 0.8868 | 0.8698 | 0.9772 |
| Saliency pruning | 94.9% | 0.8686 | 0.8791 | 0.9591 |

### Validation observations

- The 50% saliency-pruned model achieved the highest mean malignant-class F1 score among the evaluated configurations.
- Both methods showed a general decline in performance at very high sparsity.
- Moderate pruning did not necessarily reduce validation performance relative to the dense baseline.
- The results suggest that the effect of pruning depends on both the pruning criterion and the sparsity level.

These observations describe this particular experiment and do not establish that saliency pruning is universally superior to magnitude pruning.

---

## 6. Final Held-Out Test Results

After selecting the 50% saliency-pruned configuration using validation F1, the dense baseline and selected model were evaluated on the held-out test partition.

Metrics below are averaged across the five initialization seeds on the same fixed test split.

| Metric | Dense baseline | 50% saliency pruning | Difference |
|---|---:|---:|---:|
| Accuracy | 96.32% | 97.19% | +0.87 percentage points |
| Malignant recall | 96.67% | 97.14% | +0.47 percentage points |
| Malignant F1 | 95.09% | 96.21% | +1.12 percentage points |
| ROC-AUC | 0.9968 | 0.9975 | +0.0007 |

### Test observations

On this test split, the selected 50% saliency-pruned model achieved slightly higher mean accuracy, malignant recall, malignant F1, and ROC-AUC than the dense baseline.

This is encouraging evidence that moderate saliency pruning can preserve performance on this dataset. However, the test partition contains only 114 examples, and all five seeds use the same test examples. The results should therefore be interpreted as a comparison across initializations on one split, not as five independent test-set replications.

The observed differences do not establish a statistically significant or generally reproducible improvement.

---

## 7. Reproducibility and Artifacts

The benchmark is implemented in:

- `train/benchmark_breast_cancer.py`
- `tests/test_breast_cancer_benchmark.py`

Generated result files:

- `results/breast_cancer_validation.csv` — validation summaries for the evaluated configurations.
- `results/breast_cancer_test.csv` — final test summaries for the dense baseline and selected pruned model.

The benchmark uses fixed split seeds and five model initialization seeds. The preprocessing and split behavior, reproducibility, and magnitude-pruning behavior are covered by automated tests.

The project test suite completed successfully with **43 tests passed** at the time these results were recorded.

To rerun the benchmark from the project root:

```bash
python -m train.benchmark_breast_cancer
```

To run the test suite:

```bash
python -m pytest -v
```

---

## 8. Limitations

- **Single data split:** The benchmark uses one fixed stratified train/validation/test split. Different splits may produce different results.
- **Limited test-set size:** The test partition contains 114 samples, so individual predictions can materially affect the reported metrics.
- **Shared test examples across seeds:** Averaging across initialization seeds measures variability from model initialization on the same examples; it does not provide independent test-set replications.
- **Limited architecture and hyperparameter exploration:** The benchmark uses one small MLP architecture and a fixed training configuration.
- **Pruning scope:** The reported sparsity concerns eligible weight connections, not biases or the entire model's storage footprint. Zero weights do not automatically translate to faster inference unless the execution backend supports sparse computation.
- **No clinical validation:** The dataset is a standard machine-learning benchmark. These results do not establish clinical safety, suitability for diagnosis, or real-world medical performance.

---

## 9. Next Steps

Potential follow-up experiments include:

1. **Repeated stratified cross-validation** to evaluate sensitivity to the data split.
2. **Per-seed results and variability estimates** to make differences between methods easier to assess.
3. **More sparsity levels** around the promising 50% region to identify the accuracy–sparsity trade-off.
4. **Inference benchmarks** to measure actual latency and memory usage instead of treating sparsity alone as a speed improvement.
5. **Additional datasets and architectures** to test whether the findings generalize beyond this benchmark.
6. **More detailed pruning analysis** to compare weight counts, parameter counts, model size, and predictive performance.

---

## 10. Conclusion

The Breast Cancer Wisconsin experiment demonstrates that LeanNet can train a small MLP, apply two global pruning strategies, fine-tune pruned weights, and evaluate configurations using separate training, validation, and test partitions.

In this run, 50% gradient-based saliency pruning was selected using validation malignant-class F1. On the held-out test split, the selected model retained strong predictive performance and modestly outperformed the dense baseline on the reported mean metrics.

The appropriate conclusion is that **moderate pruning was promising in this experiment**. Repeated data splits, variability analysis, and runtime measurements are needed before drawing broader conclusions about generalization or efficiency.

*The synthetic experiments and their original results should remain documented separately from this real-world benchmark.*
