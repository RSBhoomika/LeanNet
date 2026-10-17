
import numpy as np


def prune_global(parameters, target_sparsity):
    """Prune globally using first-order Taylor saliency |weight * gradient|.

    target_sparsity is the desired overall fraction of masked weight
    entries across all parameters that have a mask.

    Previously pruned entries remain pruned. This function only removes
    additional entries to reach the requested target.
    """
    if (
        not np.isscalar(target_sparsity)
        or not np.isfinite(target_sparsity)
        or not 0.0 <= target_sparsity <= 1.0
    ):
        raise ValueError(
            "target_sparsity must be finite and between 0 and 1."
        )

    weights = [
        p for p in parameters
        if hasattr(p, "mask")
    ]

    if not weights:
        raise ValueError("No masked weight parameters were provided.")

    for p in weights:
        if not hasattr(p, "value") or not hasattr(p, "grad"):
            raise TypeError("Masked parameters must have value and grad.")

        if p.mask.shape != p.value.shape:
            raise ValueError("Mask shape must match parameter shape.")

        if p.grad.shape != p.value.shape:
            raise ValueError("Gradient shape must match parameter shape.")

        if not np.all(np.isfinite(p.mask)):
            raise ValueError("Masks must be finite.")

        if not np.all((p.mask == 0) | (p.mask == 1)):
            raise ValueError("Masks must contain only 0 or 1.")

    total = sum(p.mask.size for p in weights)
    target_pruned = int(round(total * float(target_sparsity)))

    current_pruned = sum(
        int(np.count_nonzero(p.mask == 0))
        for p in weights
    )

    to_prune = max(0, target_pruned - current_pruned)

    if to_prune == 0:
        return

    candidates = []

    for param_index, p in enumerate(weights):
        active = np.flatnonzero(p.mask.ravel() == 1)

        # First-order saliency estimate for removing a connection.
        importance = np.abs(p.value * p.grad).ravel()

        for flat_index in active:
            score = importance[flat_index]

            if not np.isfinite(score):
                raise ValueError("Importance scores must be finite.")

            candidates.append(
                (float(score), param_index, int(flat_index))
            )

    if to_prune > len(candidates):
        raise ValueError("Cannot prune the requested number of weights.")

    # Deterministic tie-breaking by parameter index and flat index.
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    for _, param_index, flat_index in candidates[:to_prune]:
        p = weights[param_index]

        p.mask.ravel()[flat_index] = 0.0
        p.value.ravel()[flat_index] = 0.0
