import numpy as np
import pytest

from engine.tensor import Tensor
from nn.optim import SGD
from nn.optim import Adam


def test_sgd_updates_parameter():
    p = Tensor([1.0, 2.0], requires_grad=True)
    p.grad = np.array([0.5, -1.0])

    optimizer = SGD([p], lr=0.1)
    optimizer.step()

    np.testing.assert_allclose(p.value, [0.95, 2.1])


def test_zero_grad_resets_gradients():
    p = Tensor([1.0, 2.0], requires_grad=True)
    p.grad = np.array([3.0, -4.0])

    optimizer = SGD([p])
    optimizer.zero_grad()

    np.testing.assert_array_equal(p.grad, [0.0, 0.0])


def test_sgd_rejects_nonpositive_learning_rate():
    p = Tensor([1.0], requires_grad=True)

    with pytest.raises(ValueError):
        SGD([p], lr=0.0)

    with pytest.raises(ValueError):
        SGD([p], lr=-0.1)


def test_sgd_skips_non_trainable_parameters():
    p = Tensor([1.0], requires_grad=False)
    p.grad = np.array([2.0])

    optimizer = SGD([p], lr=0.1)
    optimizer.step()

    np.testing.assert_array_equal(p.value, [1.0])


def test_adam_updates_parameter():
    p = Tensor([1.0, 2.0], requires_grad=True)
    p.grad = np.array([0.5, -1.0])

    optimizer = Adam([p], lr=0.01)
    optimizer.step()

    assert not np.array_equal(p.value, [1.0, 2.0])
    assert np.all(np.isfinite(p.value))


def test_adam_reduces_simple_quadratic_loss():
    p = Tensor([3.0], requires_grad=True)
    optimizer = Adam([p], lr=0.05)

    initial_loss = float(p.value[0] ** 2)

    for _ in range(100):
        p.grad = 2 * p.value
        optimizer.step()

    final_loss = float(p.value[0] ** 2)
    assert final_loss < initial_loss


def test_adam_rejects_invalid_hyperparameters():
    p = Tensor([1.0], requires_grad=True)

    with pytest.raises(ValueError):
        Adam([p], lr=0)

    with pytest.raises(ValueError):
        Adam([p], beta1=1.0)

    with pytest.raises(ValueError):
        Adam([p], beta2=-0.1)

    with pytest.raises(ValueError):
        Adam([p], eps=0)
