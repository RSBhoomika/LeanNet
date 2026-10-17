
import numpy as np

from engine.tensor import Tensor


class Linear:
    """Fully connected layer with optional connection pruning."""

    def __init__(self, in_features: int, out_features: int):
        if in_features <= 0 or out_features <= 0:
            raise ValueError("Feature dimensions must be positive.")

        limit = np.sqrt(6.0 / (in_features + out_features))

        self.weight = Tensor(
            np.random.uniform(
                -limit, limit, size=(in_features, out_features)
            ),
            requires_grad=True,
        )
        self.bias = Tensor(
            np.zeros((1, out_features)),
            requires_grad=True,
        )

        # 1 = active connection; 0 = pruned connection.
        self.weight.mask = np.ones_like(self.weight.value)

    def __call__(self, x: Tensor) -> Tensor:
        if x.value.ndim != 2:
            raise ValueError("Input must be a 2D array: (batch, features).")

        if x.value.shape[1] != self.weight.value.shape[0]:
            raise ValueError(
                f"Expected {self.weight.value.shape[0]} input features, "
                f"got {x.value.shape[1]}."
            )

        mask = Tensor(self.weight.mask)
        effective_weight = self.weight * mask

        return x @ effective_weight + self.bias

    def parameters(self) -> list[Tensor]:
        return [self.weight, self.bias]
