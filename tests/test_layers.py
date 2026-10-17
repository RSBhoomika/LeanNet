
import numpy as np
import pytest

from engine.tensor import Tensor
from nn.layers import Linear


def test_linear_output_shape():
    layer = Linear(2, 4)
    x = Tensor(np.ones((3, 2)))

    output = layer(x)

    assert output.value.shape == (3, 4)


def test_linear_parameters_require_grad():
    layer = Linear(2, 4)

    assert layer.weight.requires_grad
    assert layer.bias.requires_grad
    assert len(layer.parameters()) == 2


def test_linear_gradients_are_computed():
    layer = Linear(2, 3)
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))

    output = layer(x)
    loss = output.sum()
    loss.backward()

    assert layer.weight.grad.shape == (2, 3)
    assert layer.bias.grad.shape == (1, 3)
    assert np.all(np.isfinite(layer.weight.grad))
    assert np.all(np.isfinite(layer.bias.grad))
    assert np.any(layer.weight.grad != 0)


def test_linear_rejects_wrong_feature_count():
    layer = Linear(2, 3)
    x = Tensor(np.ones((2, 5)))

    with pytest.raises(ValueError):
        layer(x)


def test_linear_rejects_non_2d_input():
    layer = Linear(2, 3)
    x = Tensor(np.ones(2))

    with pytest.raises(ValueError):
        layer(x)
