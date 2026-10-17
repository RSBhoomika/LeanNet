# LeanNet — From-Scratch Neural Network and Pruning

LeanNet is a lightweight neural network framework built with Python and NumPy. It implements reverse-mode automatic differentiation, neural network layers, optimization algorithms, and weight pruning to explore how model sparsity affects predictive performance.

The project combines low-level neural network implementation with controlled experiments on synthetic data and a real-world classification benchmark. Its focus is understanding the mechanics of backpropagation, optimizer behavior, pruning, and the trade-off between sparsity and predictive performance.

## Features

- **Automatic differentiation:** Computation graphs, reverse-mode backpropagation, and gradient accumulation.
- **Neural network layers:** Trainable linear layers with masks for pruned weights.
- **Activation and loss functions:** ReLU, tanh, sigmoid, and numerically stable softmax cross-entropy.
- **Optimizers:** Stochastic Gradient Descent (SGD) and Adam.
- **Weight pruning:** Global magnitude pruning and first-order Taylor saliency pruning.
- **Fine-tuning:** Continue training after pruning while keeping pruned connections inactive.
- **Experiment sweeps:** Compare pruning strategies across sparsity levels and random seeds.
- **Real-world benchmark:** Evaluate pruning on the Breast Cancer Wisconsin Diagnostic dataset.
- **Automated testing:** Validate gradients, layers, optimizers, preprocessing, reproducibility, and pruning behavior.

## Project Structure

```text
LeanNet/
├── engine/
│   └── tensor.py
├── nn/
│   ├── __init__.py
│   ├── layers.py
│   ├── optim.py
│   └── pruning.py
├── train/
│   ├── train_mlp.py
│   ├── experiment_sweep.py
│   ├── compare_methods.py
│   ├── plot_results.py
│   ├── summarize_results.py
│   └── benchmark_breast_cancer.py
├── tests/
│   ├── test_gradcheck.py
│   ├── test_layers.py
│   ├── test_optim.py
│   ├── test_pruning.py
│   └── test_breast_cancer_benchmark.py
├── results/
│   ├── EXPERIMENT_REPORT.md
│   ├── sparsity_sweep.csv
│   ├── sparsity_accuracy.png
│   ├── breast_cancer_validation.csv
│   └── breast_cancer_test.csv
├── DESIGN.md
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.12 recommended
- NumPy
- Matplotlib
- pytest
- scikit-learn

### Installation

Create and activate a virtual environment in PowerShell:

```powershell
python -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks virtual environment activation, invoke the environment's Python executable directly:

```powershell
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
```

Run the commands below from the project root.

## Running the Project

### 1. Train the baseline model

```powershell
python -m train.train_mlp
```

This script trains a small multilayer perceptron on a synthetic binary classification dataset. It evaluates the dense model, progressively prunes weights, and fine-tunes the network.

### 2. Run the synthetic pruning experiment sweep

```powershell
python -m train.experiment_sweep
```

This experiment compares magnitude-based and gradient-based saliency pruning across target sparsities of 0%, 50%, 75%, 90%, and 95%, using three random seeds.

Generated outputs:

- `results/sparsity_sweep.csv` — experiment metrics.
- `results/sparsity_accuracy.png` — sparsity-versus-accuracy visualization.

**Note:** Running the sweep overwrites its corresponding output files.

### 3. Run the Breast Cancer Wisconsin benchmark

```powershell
python -m train.benchmark_breast_cancer
```

This experiment evaluates a small MLP on the Breast Cancer Wisconsin Diagnostic dataset.

The benchmark:

1. Creates stratified training, validation, and test partitions.
2. Fits feature scaling on the training partition only.
3. Trains dense models using five initialization seeds.
4. Evaluates magnitude pruning and gradient-based saliency pruning at multiple sparsity levels.
5. Selects the best pruned configuration using mean validation malignant-class F1.
6. Evaluates the dense baseline and selected pruned configuration on the held-out test set.

Generated outputs:

- `results/breast_cancer_validation.csv` — validation metrics for the evaluated configurations.
- `results/breast_cancer_test.csv` — final test metrics for the dense baseline and selected pruned model.

### 4. Run the automated tests

```powershell
python -m pytest -v
```

The latest recorded run passed **43 tests**. The tests cover automatic differentiation, layers, optimizers, pruning invariants, dataset splitting, preprocessing, reproducibility, and magnitude pruning.

## How Pruning Works

LeanNet uses a binary mask to determine which weights remain active:

\[
W_{\mathrm{effective}} = W \odot M
\]

Here, \(W\) is the trainable weight matrix, \(M\) is its binary mask, and \(\odot\) denotes element-wise multiplication.

### Magnitude pruning

Magnitude pruning ranks weights by absolute value:

\[
s_i = |w_i|
\]

Weights with the smallest magnitudes are removed first.

### First-order Taylor saliency pruning

Saliency pruning uses the weight-gradient product:

\[
s_i = |w_i g_i|
\]

where \(w_i\) is a weight and \(g_i\) is the gradient of the loss with respect to that weight.

The score approximates the importance of a weight using first-order loss information. It is a heuristic and does not guarantee the actual loss change after pruning.

### Fine-tuning

After pruning, the remaining weights are fine-tuned to help compensate for the removed connections. Pruning masks are maintained during optimization so that masked weights remain zero.

## Results

LeanNet includes two distinct experiment groups. Their results should be interpreted separately because they use different datasets and experimental configurations.

### Synthetic experiments

In one baseline run, the dense model achieved **98.33% test accuracy**. After gradual pruning to 50% weight sparsity and fine-tuning, test accuracy was **96.67%**, a decrease of 1.67 percentage points.

The three-seed synthetic sweep showed that both pruning methods could recover to approximately 98% test accuracy at 50% sparsity after fine-tuning. Results became less consistent at higher sparsity levels, particularly at 90% and 95%.

These results are specific to the synthetic data, architecture, training configuration, and random seeds used in the experiment.

### Breast Cancer Wisconsin benchmark

The benchmark contains 569 samples and 30 numeric features. Malignant cases are treated as the positive class for evaluation.

| Partition | Samples |
|---|---:|
| Training | 341 |
| Validation | 114 |
| Test | 114 |

The model selected using validation malignant-class F1 was the 50% saliency-pruned configuration.

Final held-out test metrics, averaged across five initialization seeds:

| Metric | Dense baseline | 50% saliency pruning |
|---|---:|---:|
| Accuracy | 96.32% | 97.19% |
| Malignant recall | 96.67% | 97.14% |
| Malignant F1 | 95.09% | 96.21% |
| ROC-AUC | 0.9968 | 0.9975 |

On this particular test split, the selected pruned model achieved slightly higher mean values for the reported metrics than the dense baseline.

These results are preliminary: all five initialization seeds use the same held-out test examples, and the test partition contains 114 samples. The comparison does not establish statistical significance or guarantee that the improvement will generalize to other splits.

The full methodology, validation results, test results, and limitations are documented in [`results/EXPERIMENT_REPORT.md`](results/EXPERIMENT_REPORT.md).

## Limitations

- The neural network implementation is intentionally small and prioritizes clarity over production performance.
- The experiments use fixed data splits and a limited number of initialization seeds.
- Performance may decrease substantially at high sparsity.
- Zero-valued masked weights do not automatically reduce memory usage or inference latency on standard dense computation backends.
- The breast cancer benchmark is a machine-learning experiment, not a clinically validated diagnostic system.
- Broader conclusions require repeated data splits, variability estimates, and additional datasets.

## Future Improvements

- Evaluate performance using repeated stratified cross-validation.
- Report per-seed results, mean values, and variability estimates.
- Compare gradual pruning with one-shot pruning.
- Explore layer-wise pruning and alternative importance criteria.
- Measure actual inference latency, model storage, and memory consumption.
- Add experiment configuration files and more extensive reproducibility metadata.

## Project Goals

LeanNet is designed to make neural network mechanics easier to understand, test, and experiment with. Its main goal is to investigate how automatic differentiation, optimization, and pruning interact, while reporting both successful results and the limitations of the experiments.