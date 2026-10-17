# LeanNet — Technical Design

## 1. Overview

LeanNet is a lightweight neural network framework implemented using Python and NumPy. It is designed to make automatic differentiation, neural network training, optimization, and weight pruning understandable, testable, and reproducible.

The implementation is organized into four main responsibilities:

1. Tensor operations and reverse-mode automatic differentiation.
2. Neural network layers and loss computation.
3. Parameter optimization and pruning-mask management.
4. Training experiments, evaluation, and result reporting.

The framework prioritizes correctness and clarity over production-scale performance.

## 2. Architecture

The source code separates tensor computation, neural network components, and experiment orchestration.

```text
                    Training Data
                         |
                         v
                  Neural Network
                 Linear -> ReLU
                    -> Linear
                         |
                         v
                Softmax Cross-Entropy
                         |
                         v
                Automatic Differentiation
                         |
                         v
                  Gradient Computation
                         |
                         v
                    SGD / Adam
                         |
                         v
                    Dense Model
                         |
                         v
                    Pruning
                         |
                         v
                     Fine-Tuning
                         |
                         v
                Metrics and Evaluation
```

### Main modules

| Module | Responsibility |
|---|---|
| `engine/tensor.py` | Tensor operations, computation graph, and reverse-mode differentiation |
| `nn/layers.py` | Trainable neural network layers |
| `nn/optim.py` | SGD and Adam parameter updates |
| `nn/pruning.py` | Pruning operations and mask management |
| `train/train_mlp.py` | Baseline synthetic-data training |
| `train/experiment_sweep.py` | Synthetic pruning experiments |
| `train/benchmark_breast_cancer.py` | Real-world dataset benchmark |
| `tests/` | Automated correctness and benchmark tests |
| `results/` | Experiment outputs and report |

## 3. Automatic Differentiation

### Objective

Automatic differentiation computes gradients of a scalar loss with respect to model parameters by tracking the operations that produced the loss.

For a composition of functions,

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

During the forward pass, LeanNet builds a computation graph. During the backward pass, it propagates gradients from the loss toward the parameters.

### Design considerations

The tensor engine must:

- Retain the information needed to compute local derivatives.
- Accumulate gradients when a tensor contributes to multiple computation paths.
- Handle matrix multiplication, reductions, and broadcasting.
- Support the activation and loss operations required by the network.
- Validate gradients against numerical approximations.

A central finite-difference estimate of a derivative is:

\[
\frac{\partial L}{\partial x}
\approx
\frac{L(x+\epsilon)-L(x-\epsilon)}{2\epsilon}
\]

Gradient-checking tests compare analytical gradients with numerical estimates within a tolerance. Numerical checks help detect implementation errors but are not a proof of correctness for every possible input.

## 4. Neural Network Layers

The `Linear` layer implements an affine transformation:

\[
Y = XW_{\mathrm{effective}} + b
\]

where:

- \(X\) is the input batch.
- \(W_{\mathrm{effective}}\) is the effective weight matrix.
- \(b\) is the bias vector.

The layer exposes trainable parameters for optimization and supports weight masks for pruning. Weight initialization uses a Xavier-style uniform distribution, with biases initialized to zero.

### Weight masks

The effective weight matrix is defined as:

\[
W_{\mathrm{effective}} = W \odot M
\]

where \(M\) is a binary matrix with one entry per weight.

- A mask value of 1 keeps the weight active.
- A mask value of 0 disables the weight.

The mask preserves the original parameter shape while changing which connections contribute to the forward computation.

Pruning-aware gradient and optimizer handling must respect the mask so that inactive weights remain zero. Biases are not included in the reported weight sparsity.

## 5. Activation and Loss Functions

The network supports ReLU, tanh, sigmoid, and softmax cross-entropy.

### ReLU

\[
\operatorname{ReLU}(x) = \max(0,x)
\]

ReLU introduces nonlinearity between linear layers.

### Softmax

For logits \(z\), the probability assigned to class \(i\) is:

\[
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
\]

### Cross-entropy

For a target class \(y\):

\[
L = -\log(p_y)
\]

The softmax cross-entropy implementation must remain numerically stable for large positive or negative logits. Tests cover extreme logits and the corresponding gradients.

## 6. Optimization

LeanNet implements SGD and Adam.

### 6.1 Stochastic Gradient Descent

The basic SGD update is:

\[
\theta_{t+1} = \theta_t - \eta g_t
\]

where \(\theta_t\) is a parameter, \(g_t\) is its gradient, and \(\eta\) is the learning rate.

### 6.2 Adam

Adam maintains moving averages of gradients and squared gradients:

\[
m_t = \beta_1m_{t-1}+(1-\beta_1)g_t
\]

\[
v_t = \beta_2v_{t-1}+(1-\beta_2)g_t^2
\]

After bias correction, the parameter update is:

\[
\theta_{t+1}
=
\theta_t-\eta
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
\]

Optimizer state needs careful handling when pruning masks change. In particular, stale Adam momentum must not reactivate masked weights. LeanNet includes tests for optimizer behavior after pruning and mask changes.

## 7. Pruning Strategies

Pruning disables selected weight connections according to an importance criterion. LeanNet evaluates magnitude-based and gradient-based saliency pruning.

### 7.1 Magnitude pruning

The importance score for weight \(w_i\) is:

\[
s_i = |w_i|
\]

Weights with smaller absolute values are removed first. Magnitude pruning is straightforward and does not require a separate gradient-based importance calculation.

### 7.2 First-order Taylor saliency

The saliency score is:

\[
s_i = |w_i g_i|
\]

where \(g_i\) is the gradient of the loss with respect to \(w_i\).

A first-order approximation for changing a weight by \(\Delta w_i\) is:

\[
\Delta L \approx g_i\Delta w_i
\]

Setting the weight to zero gives \(\Delta w_i=-w_i\), so the approximation becomes:

\[
\Delta L \approx -w_i g_i
\]

The absolute value of the weight-gradient product is used as a nonnegative importance score. This is a heuristic: it ignores higher-order interactions and does not guarantee the actual loss change after removing a weight.

### 7.3 Global pruning

Global pruning compares eligible weights across the selected parameter tensors, rather than independently enforcing the same sparsity in every layer.

For target sparsity \(s\) and \(N\) eligible weight entries, the approximate number of pruned entries is:

\[
N_{\mathrm{pruned}} \approx sN
\]

The exact achievable sparsity depends on the number of eligible weights and the integer rounding used by the implementation.

The reported sparsity is:

\[
\text{Sparsity} =
\frac{\text{Number of zeroed eligible weights}}
{\text{Total eligible weights}}
\]

Target and actual sparsity can differ slightly. For example, the benchmark's 95% target produces approximately 94.9% actual sparsity.

Pruning masks prevent selected weights from contributing to the forward pass. The optimizers must also preserve masked zeros during updates.

## 8. Fine-Tuning After Pruning

Pruning removes connections from the network and can increase loss or reduce predictive performance.

Fine-tuning continues optimization using the surviving connections so that active parameters can adapt.

The experiment pipeline distinguishes:

1. **Dense baseline:** metrics before pruning.
2. **Immediately pruned model:** metrics after pruning and before fine-tuning.
3. **Fine-tuned model:** metrics after additional optimization.

This separation helps distinguish the immediate impact of pruning from the performance recovered through retraining.

## 9. Experimental Design

LeanNet contains two separate experimental tracks.

### 9.1 Synthetic pruning experiments

The synthetic experiments use a small binary classification task to compare pruning methods under controlled conditions.

The sweep evaluates target sparsities of:

- 0%
- 50%
- 75%
- 90%
- 95%

It uses three random seeds: 42, 43, and 44.

For each seed, the dense model is trained before pruning experiments are conducted. The trained weights provide a common starting point for the pruning configurations associated with that seed.

The experiment records predictive metrics, target and actual sparsity, and the number of zeroed weight entries.

Outputs include:

- `results/sparsity_sweep.csv`
- `results/sparsity_accuracy.png`

These results characterize the synthetic task and should not be treated as evidence of performance on other datasets.

### 9.2 Breast Cancer Wisconsin benchmark

The real-world benchmark uses the Breast Cancer Wisconsin Diagnostic dataset from scikit-learn.

| Property | Value |
|---|---:|
| Samples | 569 |
| Numeric features | 30 |
| Classes | 2 |
| Training samples | 341 |
| Validation samples | 114 |
| Test samples | 114 |

The positive class is defined as malignant.

#### Data preprocessing

The benchmark uses stratified partitions:

- 60% training.
- 20% validation.
- 20% test.

`StandardScaler` is fitted on training data only. The same fitted transformation is applied to validation and test data. This avoids leakage of validation or test feature statistics into preprocessing.

#### Model configuration

The model is a small MLP:

- Input: 30 features.
- Hidden layer: 16 neurons with ReLU.
- Output: 2 logits.
- Loss: softmax cross-entropy.
- Optimizer: SGD.
- Learning rate: 0.03.
- Batch size: 32.
- Dense training: 100 epochs.
- Fine-tuning: 15 epochs.
- Initialization seeds: 42, 43, 44, 45, and 46.

#### Evaluation protocol

The benchmark compares the dense baseline against magnitude and saliency pruning at multiple sparsity levels.

The selection procedure is:

1. Train dense models using five initialization seeds.
2. Create pruning variants from the trained dense weights.
3. Compute saliency scores using training data.
4. Fine-tune the pruned variants.
5. Compare configurations using validation metrics.
6. Select the configuration with the highest mean validation malignant-class F1.
7. Evaluate the dense baseline and selected pruned configuration on the held-out test partition.

The selected configuration is 50% gradient-based saliency pruning.

The validation and final test results are saved to:

- `results/breast_cancer_validation.csv`
- `results/breast_cancer_test.csv`

### 9.3 Reported breast cancer test results

Mean results across five initialization seeds on the same fixed test partition:

| Metric | Dense baseline | Selected saliency model |
|---|---:|---:|
| Actual sparsity | 0.0% | 50.0% |
| Accuracy | 96.32% | 97.19% |
| Malignant recall | 96.67% | 97.14% |
| Malignant F1 | 95.09% | 96.21% |
| ROC-AUC | 0.9968 | 0.9975 |

On this split, the selected pruned model achieved slightly higher mean values for the listed metrics than the dense baseline.

These results should be interpreted cautiously. The five initialization seeds share the same test examples, and the test set contains 114 samples. The comparison does not establish statistical significance or guarantee improvement across independent data splits.

The dataset is used for machine-learning experimentation; the benchmark does not establish clinical safety or diagnostic suitability.

## 10. Testing Strategy

The test suite covers tensor operations, layers, optimization, pruning, and benchmark preprocessing.

| Test module | Purpose |
|---|---|
| `tests/test_gradcheck.py` | Tensor operations, gradient correctness, broadcasting, and numerical stability |
| `tests/test_layers.py` | Layer shapes, trainable parameters, and gradient computation |
| `tests/test_optim.py` | Parameter updates and optimizer validation |
| `tests/test_pruning.py` | Pruning masks, sparsity validation, and optimizer behavior after pruning |
| `tests/test_breast_cancer_benchmark.py` | Dataset splitting, scaling, reproducibility, and magnitude pruning |

The latest recorded test run passed all 43 tests.

Run the suite from the project root:

```powershell
python -m pytest -v
```

## 11. Limitations

- The framework is intentionally small and is not optimized for large neural networks.
- The experiments use a limited set of architectures, hyperparameters, and random seeds.
- The synthetic and real-world experiments use different datasets and should be interpreted independently.
- A single fixed train/validation/test split does not establish generalization across different partitions.
- Zeroed weights do not automatically reduce memory usage or wall-clock inference time on dense computation backends.
- The benchmark's test metrics do not establish clinical diagnostic performance.

## 12. Future Work

Potential extensions include:

1. Repeated stratified cross-validation and additional data splits.
2. Per-seed result reporting with mean and standard deviation.
3. Comparison of gradual pruning and one-shot pruning.
4. Layer-wise pruning and alternative importance criteria.
5. Actual inference latency and memory measurements.
6. More extensive configuration and environment metadata.
7. Evaluation on additional datasets and architectures.

## 13. Design Principles

LeanNet follows five core principles:

- **Explicitness:** Keep the underlying mathematical operations understandable.
- **Correctness:** Test gradients, optimizer updates, and pruning invariants.
- **Reproducibility:** Use explicit seeds and separate training, validation, and test data.
- **Separation of concerns:** Keep tensor operations, model components, pruning, and evaluation distinct.
- **Honest evaluation:** Report limitations alongside positive results and avoid treating sparsity as proof of runtime efficiency.
