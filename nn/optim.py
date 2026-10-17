
import numpy as np


def _validate_parameters(parameters):
    parameters = list(parameters)

    if any(
        not hasattr(p, "value") or not hasattr(p, "grad")
        for p in parameters
    ):
        raise TypeError("All parameters must be Tensor-like objects.")

    for p in parameters:
        if p.value.shape != p.grad.shape:
            raise ValueError("Parameter values and gradients must match.")

        mask = getattr(p, "mask", None)
        if mask is not None:
            if mask.shape != p.value.shape:
                raise ValueError("Parameter mask shape must match its value.")
            if not np.all(np.isfinite(mask)):
                raise ValueError("Parameter masks must be finite.")
            if not np.all((mask == 0) | (mask == 1)):
                raise ValueError("Masks must contain only 0 or 1.")

    return parameters


def _mask_for(parameter):
    """Return the parameter mask, or None for an unmasked parameter."""
    return getattr(parameter, "mask", None)


class SGD:
    """Stochastic Gradient Descent with binary-mask support."""

    def __init__(self, parameters, lr=0.01):
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError("Learning rate must be positive and finite.")

        self.parameters = _validate_parameters(parameters)
        self.lr = float(lr)

    def zero_grad(self):
        for p in self.parameters:
            p.grad = p.grad * 0.0

    def step(self):
        for p in self.parameters:
            if not p.requires_grad:
                continue

            mask = _mask_for(p)

            if mask is None:
                p.value -= self.lr * p.grad
            else:
                # Masked connections receive no parameter update.
                p.value -= self.lr * (p.grad * mask)

                # Enforce exact zeros for pruned connections.
                p.value *= mask


class Adam:
    """Adam with bias correction and explicit mask-transition handling.

    A connection's first and second moments are reset when it is
    masked or when it is observed transitioning from masked to active.

    If a mask is changed from active -> masked -> active entirely
    between optimizer steps, that intermediate transition cannot be
    inferred from the current mask alone. In that situation, call
    reset_state_for(parameter) after changing the mask.
    """

    def __init__(
        self,
        parameters,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
    ):
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError("Learning rate must be positive and finite.")
        if not 0 <= beta1 < 1:
            raise ValueError("beta1 must be in [0, 1).")
        if not 0 <= beta2 < 1:
            raise ValueError("beta2 must be in [0, 1).")
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be positive and finite.")

        self.parameters = _validate_parameters(parameters)

        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.t = 0

        self.m = [np.zeros_like(p.value) for p in self.parameters]
        self.v = [np.zeros_like(p.value) for p in self.parameters]

        # Snapshot masks to detect changes between optimizer steps.
        self._previous_masks = [
            None if _mask_for(p) is None else _mask_for(p).copy()
            for p in self.parameters
        ]

    def zero_grad(self):
        for p in self.parameters:
            p.grad = p.grad * 0.0

    def reset_state_for(self, parameter):
        """Reset Adam moments for a parameter after external mask changes."""
        try:
            i = next(
                i for i, p in enumerate(self.parameters)
                if p is parameter
            )
        except StopIteration:
            raise ValueError("Parameter is not managed by this optimizer.")

        self.m[i].fill(0.0)
        self.v[i].fill(0.0)

        mask = _mask_for(parameter)
        self._previous_masks[i] = (
            None if mask is None else mask.copy()
        )

    def step(self):
        self.t += 1

        for i, p in enumerate(self.parameters):
            mask = _mask_for(p)

            if mask is None:
                grad = p.grad
            else:
                previous = self._previous_masks[i]

                if previous is None:
                    # The parameter acquired a mask after initialization.
                    self.m[i].fill(0.0)
                    self.v[i].fill(0.0)
                else:
                    # Clear state for newly pruned and newly revived entries.
                    changed = previous != mask
                    self.m[i][changed] = 0.0
                    self.v[i][changed] = 0.0

                # Always clear state on currently masked connections.
                inactive = mask == 0
                self.m[i][inactive] = 0.0
                self.v[i][inactive] = 0.0

                grad = p.grad * mask

            if p.requires_grad:
                self.m[i] = (
                    self.beta1 * self.m[i]
                    + (1.0 - self.beta1) * grad
                )
                self.v[i] = (
                    self.beta2 * self.v[i]
                    + (1.0 - self.beta2) * (grad ** 2)
                )

                m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)

                p.value -= self.lr * m_hat / (
                    np.sqrt(v_hat) + self.eps
                )

            if mask is not None:
                # Do not allow optimizer updates to regrow masked weights.
                p.value *= mask

                # Snapshot the mask for the next step.
                self._previous_masks[i] = mask.copy()
