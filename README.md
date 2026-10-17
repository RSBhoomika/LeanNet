# LeanNet — From-Scratch Neural Network and Pruning

LeanNet is a lightweight neural network framework built with Python and NumPy. It implements reverse-mode automatic differentiation, neural network layers, optimization algorithms, and weight pruning to explore how model sparsity affects predictive performance.

The project focuses on understanding the mechanics behind neural network training and evaluating the trade-off between model sparsity and accuracy.

## Features

- **Automatic differentiation:** A computation graph with reverse-mode backpropagation and gradient accumulation.
- **Neural network layers:** Trainable linear layers with masking support for pruned weights.
- **Activation and loss functions:** ReLU, tanh, sigmoid, and softmax cross-entropy.
- **Optimizers:** Stochastic Gradient Descent (SGD) and Adam.
- **Weight pruning:** Global weight pruning using first-order Taylor saliency and magnitude-based criteria.
- **Fine-tuning:** Retraining after pruning to recover predictive performance.
- **Experiment sweeps:** Evaluation across multiple sparsity levels and random seeds.
- **Testing:** Automated tests for gradients, layers, optimizers, and pruning behavior.

## Project Structure

```text
LeanNet/
├── nn/
│   ├── ...                 # Autodiff, layers, losses, optimizers, etc.
├── train/
│   ├── train_mlp.py        # Baseline training and gradual pruning
│   └── experiment_sweep.py # Multi-seed pruning experiments
├── tests/
│   ├── test_gradcheck.py
│   ├── test_layers.py
│   ├── test_optim.py
│   └── test_pruning.py
├── results/
│   ├── sparsity_sweep.csv
│   └── sparsity_accuracy.png
├── DESIGN.md
├── requirements.txt
└── README.md
```

The exact contents of `nn/` may evolve as the framework develops.

## Requirements

- Python 3.12 recommended
- NumPy
- Matplotlib
- pytest

Install dependencies in a virtual environment:

```powershell
python -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks virtual environment activation, use the Python executable directly:

```powershell
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the Project

### 1. Train the baseline model and gradually prune it

```powershell
python -m train.train_mlp
```

This script trains a small multilayer perceptron on a synthetic binary classification dataset. It evaluates the dense model, progressively prunes its weights, and fine-tunes the model after each pruning stage.

### 2. Run the pruning experiment sweep

```powershell
python -m train.experiment_sweep
```

The experiment compares saliency-based pruning with magnitude-based pruning at target sparsities of 0%, 50%, 75%, 90%, and 95%, across three random seeds.

Results are saved to:

- `results/sparsity_sweep.csv` — per-run metrics.
- `results/sparsity_accuracy.png` — generated sparsity-versus-accuracy plot.

Running the sweep overwrites these output files.

### 3. Run the test suite

```powershell
python -m pytest -v
```

The test suite covers automatic differentiation, layers, optimizer behavior, and pruning invariants.

## How Pruning Works

Pruning removes selected weights by setting their masks to zero. The effective weight matrix is:

\[
W_{\text{effective}} = W \odot M
\]

where \(W\) is the trainable weight matrix, \(M\) is its binary mask, and \(\odot\) denotes element-wise multiplication.

Two pruning criteria are evaluated:

**Magnitude pruning**

Ranks weights by their absolute value:

\[
s_i = |w_i|
\]

Weights with smaller magnitudes are pruned first.

**First-order Taylor saliency**

Ranks weights using the magnitude of the weight-gradient product:

\[
s_i = |w_i g_i|
\]

where \(g_i\) is the gradient of the loss with respect to weight \(w_i\). This estimates the first-order importance of a weight to the loss.

After pruning, the model is fine-tuned to help the remaining weights compensate for the removed connections.

## Results

In one baseline run, the dense model achieved **98.33% test accuracy**. After gradual pruning to **50% weight sparsity** and fine-tuning, it achieved **96.67% test accuracy**—a decrease of 1.67 percentage points.

In the three-seed sweep, both pruning methods recovered to approximately 98% test accuracy at 50% sparsity after fine-tuning. Results became less consistent at higher sparsity levels, especially at 90% and 95%.

These results are specific to the current synthetic dataset, model architecture, training configuration, and random seeds. They are illustrative experiments, not evidence that one pruning criterion is universally better.

## Current Limitations

- The training task uses a synthetic dataset rather than a real-world benchmark.
- The network is intentionally small and designed for learning and experimentation.
- High sparsity can significantly reduce accuracy.
- Sparse masks reduce the number of active weights, but do not automatically guarantee faster inference or lower wall-clock runtime on standard hardware.
- The current experiments evaluate predictive performance and weight sparsity rather than deployment speed or memory usage.

## Future Improvements

- Evaluate on a real dataset with a clearly defined train/validation/test split.
- Expand the experiment sweep and report mean and standard deviation across seeds.
- Compare gradual pruning with one-shot pruning.
- Measure parameter storage, inference latency, and memory use.
- Add experiment configuration files and reproducible result summaries.

