
import numpy as np
import pytest

from engine.tensor import Tensor
from nn.layers import Linear
from nn.optim import SGD, Adam
from nn.pruning import prune_global


def test_pruning_masks_low_importance_weights():
    layer = Linear(2, 2)
    layer.weight.value[:] = [[0.01, 2.0], [3.0, 4.0]]
    layer.weight.grad[:] = 1.0

    prune_global(layer.parameters(), 0.25)

    assert np.count_nonzero(layer.weight.mask == 0) == 1
    assert layer.weight.mask[0, 0] == 0
    assert layer.weight.value[0, 0] == 0


def test_pruned_weight_has_zero_forward_gradient():
    layer = Linear(2, 2)
    layer.weight.mask[0, 0] = 0
    layer.weight.value[0, 0] = 0

    x = Tensor([[1.0, 2.0]])
    output = layer(x)
    output.sum().backward()

    assert layer.weight.grad[0, 0] == 0


@pytest.mark.parametrize("optimizer_class", [SGD, Adam])
def test_optimizer_does_not_regrow_pruned_weights(optimizer_class):
    layer = Linear(2, 2)
    layer.weight.mask[0, 0] = 0
    layer.weight.value[0, 0] = 0

    optimizer = optimizer_class(layer.parameters(), lr=0.1)

    for _ in range(3):
        layer.weight.grad[:] = 1.0
        layer.bias.grad[:] = 1.0
        optimizer.step()

    assert layer.weight.value[0, 0] == 0
    assert layer.weight.mask[0, 0] == 0


def test_adam_resets_moments_when_weight_is_revived():
    layer = Linear(2, 2)
    optimizer = Adam(layer.parameters(), lr=0.1)

    weight = layer.weight
    weight.value[0, 0] = 1.0

    # Build nonzero optimizer state while the connection is active.
    weight.grad[:] = 0.0
    weight.grad[0, 0] = 1.0
    layer.bias.grad[:] = 0.0
    optimizer.step()

    assert optimizer.m[0][0, 0] != 0
    assert optimizer.v[0][0, 0] != 0

    # Prune the connection and take an optimizer step.
    weight.mask[0, 0] = 0
    weight.value[0, 0] = 0.0
    weight.grad[:] = 1.0
    optimizer.step()

    assert optimizer.m[0][0, 0] == 0
    assert optimizer.v[0][0, 0] == 0
    assert weight.value[0, 0] == 0

    # Revive it with zero gradient: stale momentum must not move it.
    weight.mask[0, 0] = 1
    weight.grad[:] = 0.0
    value_before = weight.value[0, 0].copy()

    optimizer.step()

    assert weight.value[0, 0] == pytest.approx(value_before)
    assert optimizer.m[0][0, 0] == 0
    assert optimizer.v[0][0, 0] == 0


def test_adam_explicit_reset_after_mask_changes_between_steps():
    layer = Linear(2, 2)
    optimizer = Adam(layer.parameters(), lr=0.1)

    weight = layer.weight

    # Accumulate momentum.
    weight.grad[:] = 0.0
    weight.grad[0, 0] = 1.0
    optimizer.step()

    assert optimizer.m[0][0, 0] != 0

    # Simulate an external prune-and-revive operation between steps.
    weight.mask[0, 0] = 0
    weight.value[0, 0] = 0.0
    weight.mask[0, 0] = 1

    # Tell Adam that its state for this parameter is no longer valid.
    optimizer.reset_state_for(weight)

    assert optimizer.m[0][0, 0] == 0
    assert optimizer.v[0][0, 0] == 0

    weight.grad[:] = 0.0
    value_before = weight.value[0, 0].copy()

    optimizer.step()

    assert weight.value[0, 0] == pytest.approx(value_before)


def test_pruning_rejects_invalid_sparsity():
    layer = Linear(2, 2)

    with pytest.raises(ValueError):
        prune_global(layer.parameters(), 1.1)


def test_pruning_rejects_missing_masks():
    p = Tensor([1.0], requires_grad=True)

    with pytest.raises(ValueError):
        prune_global([p], 0.5)
