
import numpy as np
from engine.tensor import Tensor, softmax_cross_entropy


def numerical_grad(fn, x, h=1e-5):
    """Central-difference gradient of a scalar-valued function."""
    x = np.asarray(x, dtype=np.float64).copy()
    grad = np.zeros_like(x)

    for idx in np.ndindex(x.shape):
        original = x[idx]

        x[idx] = original + h
        plus = float(fn(x))

        x[idx] = original - h
        minus = float(fn(x))

        x[idx] = original
        grad[idx] = (plus - minus) / (2 * h)

    return grad


def check_grad(fn, x, atol=1e-5, rtol=1e-4):
    x = np.asarray(x, dtype=np.float64)

    # Analytical gradient from our engine.
    tensor = Tensor(x.copy(), requires_grad=True)
    loss = fn(tensor)
    loss.backward()
    analytical = tensor.grad.copy()

    # Numerical gradient, independently calculated.
    numerical = numerical_grad(
        lambda v: fn(Tensor(v)).value, x
    )

    np.testing.assert_allclose(
        analytical, numerical, atol=atol, rtol=rtol
    )


def test_add():
    check_grad(lambda x: (x + 3.0).sum(), [0.3, -0.5, 1.2])


def test_sub():
    check_grad(lambda x: (x - 2.0).sum(), [0.3, -0.5, 1.2])


def test_mul():
    check_grad(lambda x: (x * x).sum(), [0.3, -0.5, 1.2])


def test_div():
    check_grad(lambda x: (x / 2.0).sum(), [0.3, 0.5, 1.2])


def test_matmul():
    a = np.array([[0.2, 0.4], [0.7, -0.3]])
    b = np.array([[0.5, -0.2], [0.1, 0.8]])

    def fn(x):
        w = Tensor(b)
        return (x @ w).sum()

    check_grad(fn, a)


def test_sum():
    check_grad(lambda x: x.sum(), [[0.2, 0.4], [0.7, -0.3]])


def test_mean():
    check_grad(lambda x: x.mean(), [[0.2, 0.4], [0.7, -0.3]])


def test_relu():
    check_grad(
        lambda x: x.relu().sum(),
        [-1.2, -0.3, 0.7, 1.4]
    )


def test_tanh():
    check_grad(lambda x: x.tanh().sum(), [-1.2, -0.3, 0.7, 1.4])


def test_sigmoid():
    check_grad(lambda x: x.sigmoid().sum(), [-1.2, -0.3, 0.7, 1.4])


def test_broadcast_add():
    bias = np.array([0.2, -0.3, 0.7])
    data = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ])

    def fn(x):
        b = Tensor(bias)
        return (x + b).sum()

    check_grad(fn, data)


def test_broadcast_gradient_accumulates():
    x = Tensor([2.0, 3.0], requires_grad=True)
    y = (x * x).sum() + (x * 3.0).sum()
    y.backward()

    # dy/dx = 2x + 3
    np.testing.assert_allclose(x.grad, [7.0, 9.0])


def test_softmax_cross_entropy():
    labels = np.array([0, 2])

    def fn(x):
        return softmax_cross_entropy(x, labels)

    check_grad(
        fn,
        [[0.2, -0.4, 0.7], [0.8, 0.1, -0.2]]
    )



def test_division_gradients_for_both_operands():
    a = np.array([1.2, 2.4, 3.1])
    b = np.array([2.0, 3.0, 4.0])

    def fn_a(x):
        return (x / Tensor(b)).sum()

    def fn_b(x):
        return (Tensor(a) / x).sum()

    check_grad(fn_a, a)
    check_grad(fn_b, b)


def test_matmul_gradients_for_both_operands():
    a = np.array([[0.2, 0.4], [0.7, -0.3]])
    b = np.array([[0.5, -0.2], [0.1, 0.8]])

    def fn_a(x):
        return (x @ Tensor(b)).sum()

    def fn_b(x):
        return (Tensor(a) @ x).sum()

    check_grad(fn_a, a)
    check_grad(fn_b, b)


def test_sum_over_axis():
    x = np.array([
        [0.2, 0.4, 0.6],
        [0.7, -0.3, 0.9],
    ])

    check_grad(lambda t: t.sum(axis=0).sum(), x)
    check_grad(lambda t: t.sum(axis=1).sum(), x)


def test_mean_over_axis():
    x = np.array([
        [0.2, 0.4, 0.6],
        [0.7, -0.3, 0.9],
    ])

    check_grad(lambda t: t.mean(axis=0).sum(), x)
    check_grad(lambda t: t.mean(axis=1).sum(), x)


def test_extreme_logits():
    logits = np.array([
        [1000.0, 999.0, -1000.0],
        [-1000.0, 1000.0, 0.0],
    ])
    labels = np.array([0, 1])

    loss = softmax_cross_entropy(
        Tensor(logits, requires_grad=True), labels
    )
    assert np.isfinite(loss.value).all()

    loss.backward()
    assert np.isfinite(loss.grad).all()


def test_relu_away_from_zero():
    check_grad(
        lambda x: x.relu().sum(),
        [-2.0, -0.5, 0.5, 2.0]
    )
