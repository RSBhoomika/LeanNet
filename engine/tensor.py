
import numpy as np


def unbroadcast(grad, target_shape):
    """Undo NumPy broadcasting so grad matches the original input shape."""
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)

    for axis, size in enumerate(target_shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad.reshape(target_shape)


class Tensor:
    def __init__(self, value, requires_grad=False, _children=(), _op=""):
        self.value = np.asarray(value, dtype=np.float64)
        self.grad = np.zeros_like(self.value)
        self.requires_grad = requires_grad
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Tensor(value={self.value}, grad={self.grad})"

    def _accumulate(self, grad):
        if self.requires_grad:
            self.grad += grad

    def backward(self, grad=None):
        if self.value.size != 1 and grad is None:
            raise ValueError("Non-scalar output requires an initial gradient")

        if grad is None:
            grad = np.ones_like(self.value)
        else:
            grad = np.asarray(grad, dtype=np.float64)
            if grad.shape != self.value.shape:
                raise ValueError("Initial gradient shape must match output")

        # Build a topological ordering of the computation graph.
        topo = []
        visited = set()

        def build(node):
            if node not in visited:
                visited.add(node)
                for parent in node._prev:
                    build(parent)
                topo.append(node)

        build(self)

        # Reset gradients so repeated backward calls don't
        # accidentally reuse gradients from a previous pass.
        for node in topo:
            node.grad = np.zeros_like(node.value)

        self.grad = grad.copy()

        for node in reversed(topo):
            node._backward()

    def __add__(self, other):
        other = as_tensor(other)
        out = Tensor(
            self.value + other.value,
            self.requires_grad or other.requires_grad,
            (self, other), "+"
        )

        def backward():
            self._accumulate(unbroadcast(out.grad, self.value.shape))
            other._accumulate(unbroadcast(out.grad, other.value.shape))

        out._backward = backward
        return out

    __radd__ = __add__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-as_tensor(other))

    def __rsub__(self, other):
        return as_tensor(other) - self

    def __mul__(self, other):
        other = as_tensor(other)
        out = Tensor(
            self.value * other.value,
            self.requires_grad or other.requires_grad,
            (self, other), "*"
        )

        def backward():
            self._accumulate(
                unbroadcast(out.grad * other.value, self.value.shape)
            )
            other._accumulate(
                unbroadcast(out.grad * self.value, other.value.shape)
            )

        out._backward = backward
        return out

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = as_tensor(other)
        out = Tensor(
            self.value / other.value,
            self.requires_grad or other.requires_grad,
            (self, other), "/"
        )

        def backward():
            self._accumulate(
                unbroadcast(out.grad / other.value, self.value.shape)
            )
            other._accumulate(
                unbroadcast(
                    -out.grad * self.value / (other.value ** 2),
                    other.value.shape
                )
            )

        out._backward = backward
        return out

    def __rtruediv__(self, other):
        return as_tensor(other) / self

    def __matmul__(self, other):
        other = as_tensor(other)
        if self.value.ndim != 2 or other.value.ndim != 2:
            raise ValueError("This implementation supports 2-D matmul")
        out = Tensor(
            self.value @ other.value,
            self.requires_grad or other.requires_grad,
            (self, other), "@"
        )

        def backward():
            self._accumulate(out.grad @ other.value.T)
            other._accumulate(self.value.T @ out.grad)

        out._backward = backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(
            self.value.sum(axis=axis, keepdims=keepdims),
            self.requires_grad, (self,), "sum"
        )

        if axis is None:
            axes = tuple(range(self.value.ndim))
        else:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            axes = tuple(a % self.value.ndim for a in axes)

        def backward():
            grad = out.grad
            if not keepdims:
                for ax in sorted(axes):
                    grad = np.expand_dims(grad, axis=ax)
            self._accumulate(np.broadcast_to(grad, self.value.shape))

        out._backward = backward
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            count = self.value.size
        else:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            count = int(np.prod([self.value.shape[a] for a in axes]))

        return self.sum(axis=axis, keepdims=keepdims) / count

    def relu(self):
        out = Tensor(
            np.maximum(self.value, 0),
            self.requires_grad, (self,), "relu"
        )

        def backward():
            self._accumulate(out.grad * (self.value > 0))

        out._backward = backward
        return out

    def tanh(self):
        val = np.tanh(self.value)
        out = Tensor(val, self.requires_grad, (self,), "tanh")

        def backward():
            self._accumulate(out.grad * (1 - val**2))

        out._backward = backward
        return out

    def sigmoid(self):
        # Numerically stable sigmoid.
        x = self.value
        val = np.empty_like(x)
        pos = x >= 0
        val[pos] = 1 / (1 + np.exp(-x[pos]))
        exp_x = np.exp(x[~pos])
        val[~pos] = exp_x / (1 + exp_x)

        out = Tensor(val, self.requires_grad, (self,), "sigmoid")

        def backward():
            self._accumulate(out.grad * val * (1 - val))

        out._backward = backward
        return out


def as_tensor(value):
    return value if isinstance(value, Tensor) else Tensor(value)


def softmax_cross_entropy(logits, labels):
    """Mean cross-entropy for logits shaped (batch, classes)."""
    logits = as_tensor(logits)
    labels = np.asarray(labels, dtype=np.int64)

    if logits.value.ndim != 2:
        raise ValueError("logits must have shape (batch, classes)")

    batch, classes = logits.value.shape

    if labels.shape != (batch,):
        raise ValueError("labels must have shape (batch,)")
    if np.any(labels < 0) or np.any(labels >= classes):
        raise ValueError("label outside valid class range")

    # Shift logits to avoid overflow in exp.
    shifted = logits.value - logits.value.max(axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    probs = exp_shifted / exp_shifted.sum(axis=1, keepdims=True)

    logsumexp = np.log(exp_shifted.sum(axis=1))
    log_probs = shifted - logsumexp[:, None]

    loss_value = -np.mean(
        log_probs[np.arange(batch), labels]
    )

    loss = Tensor(
        loss_value, logits.requires_grad, (logits,), "softmax_cross_entropy"
    )

    def backward():
        grad = probs.copy()
        grad[np.arange(batch), labels] -= 1
        grad /= batch
        logits._accumulate(grad * loss.grad)

    loss._backward = backward
    return loss
