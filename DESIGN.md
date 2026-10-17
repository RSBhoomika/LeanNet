# LeanNet — Technical Design

## 1. Overview

LeanNet is a small neural network framework implemented using Python and NumPy. Its purpose is to make the training process and weight pruning mechanisms understandable, testable, and reproducible.

The framework consists of four main parts:

1. Automatic differentiation.
2. Neural network layers and loss computation.
3. Parameter optimization.
4. Weight pruning and experimental evaluation.

The implementation prioritizes correctness and clarity over production-scale performance.

## 2. Architecture

The high-level training pipeline is:

```text
Synthetic Dataset
       |
       v
  Neural Network
  Linear -> ReLU -> Linear
       |
       v
 Cross-Entropy Loss
       |
       v
 Automatic Differentiation
       |
       v
 Gradient Computation
       |
       v
  SGD / Adam Update
       |
       v
  Trained Dense Model
       |
       v
   Weight Pruning
       |
       v
  Fine-Tuning
       |
       v
Accuracy and Sparsity Evaluation
```

The baseline training script uses a two-layer multilayer perceptron for binary classification. The experiment sweep evaluates pruning strategies using the same general training approach.

## 3. Automatic Differentiation

### Objective

Automatic differentiation computes gradients of a scalar loss with respect to trainable parameters.

Instead of manually deriving every gradient for each network, LeanNet tracks tensor operations in a computation graph and applies the chain rule during backpropagation.

For a computation:

\[
z = f(y), \qquad y = g(x)
\]

the chain rule gives:

\[
\frac{\partial z}{\partial x}
=
\frac{\partial z}{\partial y}
\frac{\partial y}{\partial x}
\]

The backward pass traverses the graph and propagates gradients from the output toward the inputs.

### Design considerations

- Store the data needed to compute local gradients.
- Accumulate gradients when a value contributes to multiple paths.
- Support array operations and broadcasting.
- Handle matrix multiplication and reductions correctly.
- Check analytical gradients against numerical approximations.

For a parameter \(x\), a finite-difference approximation is:

\[
\frac{\partial L}{\partial x}
\approx
\frac{L(x+\epsilon)-L(x-\epsilon)}{2\epsilon}
\]

The gradient tests use this principle to identify errors in backpropagation.

## 4. Neural Network Layers

The `Linear` layer implements an affine transformation:

\[
Y = XW_{\text{effective}} + b
\]

where:

- \(X\) is the input batch.
- \(W_{\text{effective}}\) is the effective weight matrix.
- \(b\) is the bias vector.

Weight initialization uses a Xavier-style uniform distribution, and biases are initialized to zero.

### Weight masks

Each weight has a corresponding mask:

\[
W_{\text{effective}} = W \odot M
\]

where \(M\) contains ones for active weights and zeros for pruned weights.

This design allows the original parameter shape to remain unchanged while selected connections are disabled.

The effective matrix is used during the forward pass. Gradients and optimizer updates must respect the mask so that pruned connections remain inactive.

## 5. Loss Function

The classifier uses softmax cross-entropy.

For logits \(z\), the softmax probability for class \(i\) is:

\[
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
\]

For a target class \(y\), the cross-entropy loss is:

\[
L = -\log(p_y)
\]

A numerically stable implementation is important when logits have large magnitudes. Tests cover extreme logits and the corresponding gradients.

## 6. Optimization

LeanNet supports two optimizers.

### Stochastic Gradient Descent

The basic SGD update is:

\[
\theta_{t+1} = \theta_t - \eta g_t
\]

where \(\theta\) is a parameter, \(g_t\) is its gradient, and \(\eta\) is the learning rate.

### Adam

Adam maintains exponential moving averages of gradients and squared gradients.

\[
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
\]

\[
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
\]

After bias correction, the parameter update is:

\[
\theta_{t+1}
=
\theta_t -
\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
\]

Optimizer state must be handled carefully when pruning masks change. In particular, stale Adam momentum must not reactivate pruned weights. The implementation includes tests for this behavior.

## 7. Pruning Strategy

Pruning selects weights to remove according to an importance score.

### 7.1 Magnitude pruning

The magnitude criterion is:

\[
s_i = |w_i|
\]

Weights with the smallest absolute values are considered least important and are removed first.

### 7.2 First-order Taylor saliency

The saliency criterion is:

\[
s_i = |w_i g_i|
\]

where \(w_i\) is a weight and \(g_i\) is its gradient.

The underlying first-order approximation for setting a weight to zero is:

\[
\Delta L \approx -w_i g_i
\]

The absolute value is used as a nonnegative importance score. This is an approximation, not a guarantee of the actual loss change after pruning.

### 7.3 Global pruning

Global pruning compares eligible weights across the selected parameter tensors rather than enforcing the same sparsity independently in each layer.

Given a target sparsity \(s\), the desired number of zeroed weight entries is approximately:

\[
N_{\text{pruned}} \approx sN
\]

where \(N\) is the total number of eligible weights.

Since weights are discrete entries, the actual sparsity may differ slightly from the target.

The pruning implementation validates the target sparsity, respects existing masks, and uses deterministic score ordering for tie handling.

## 8. Fine-Tuning

Pruning changes the network's effective parameters and may temporarily increase the loss or decrease accuracy.

Fine-tuning continues training with the surviving connections so that active weights can adapt.

The experimental pipeline records metrics at three stages:

1. Dense baseline.
2. Immediately after pruning, before fine-tuning.
3. After fine-tuning.

This separation makes it possible to measure both the immediate effect of pruning and the extent to which retraining recovers performance.

## 9. Experimental Design

The sweep compares saliency and magnitude pruning at target sparsities of:

- 0%
- 50%
- 75%
- 90%
- 95%

It uses three random seeds: 42, 43, and 44.

The model is trained densely first. The trained weights are then used as the starting point for pruning experiments, allowing the criteria to be compared from a common baseline for each seed.

The experiment records:

- Dense test accuracy and loss.
- Accuracy and loss before fine-tuning.
- Accuracy and loss after fine-tuning.
- Target and actual sparsity.
- Total weight entries and zero-weight entries.

Outputs are written to `results/sparsity_sweep.csv` and `results/sparsity_accuracy.png`.

Using multiple seeds helps expose variability due to initialization and data generation. Three seeds provide an initial comparison, but a larger sample is needed for stronger statistical conclusions.

## 10. Testing Strategy

The automated test suite covers four areas.

| Test module | Purpose |
|---|---|
| `test_gradcheck.py` | Validate tensor operations and gradients |
| `test_layers.py` | Validate layer shapes, parameters, and gradients |
| `test_optim.py` | Validate parameter updates and optimizer configuration |
| `test_pruning.py` | Validate masks, sparsity, and optimizer behavior after pruning |

The latest recorded test run passed all 39 tests.

Run the suite using:

```powershell
python -m pytest -v
```

## 11. Limitations and Future Work

### Current limitations

- The classification dataset is synthetic.
- The network is small and intended for experimentation.
- Results at high sparsity are variable and can be substantially worse than the dense baseline.
- Masked weights do not necessarily yield faster inference on conventional dense matrix-multiplication hardware.
- The current evaluation focuses on accuracy, loss, and weight sparsity.

### Potential extensions

- Evaluate on a real-world benchmark dataset.
- Increase the number of random seeds and report mean and standard deviation.
- Compare one-shot and gradual pruning.
- Explore layer-wise pruning and alternative importance scores.
- Measure actual memory consumption and inference latency.
- Add configuration files and experiment metadata for reproducibility.

## 12. Design Principles

The project follows these principles:

- **Explicitness:** Keep the core mathematical operations understandable.
- **Correctness:** Test gradients, parameter updates, and pruning invariants.
- **Reproducibility:** Use explicit seeds in experiments.
- **Separation of concerns:** Keep training, numerical operations, pruning, and evaluation distinct.
- **Honest evaluation:** Report both model sparsity and predictive performance, including failure cases.
