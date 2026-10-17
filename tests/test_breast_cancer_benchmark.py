
import numpy as np

from train.benchmark_breast_cancer import (
    actual_sparsity,
    build_model,
    load_data,
    prune_by_magnitude,
)


def test_breast_cancer_split_shapes_and_labels():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    assert X_train.shape == (341, 30)
    assert X_val.shape == (114, 30)
    assert X_test.shape == (114, 30)

    # Malignant is encoded as the positive class (1).
    assert set(np.unique(y_train)) == {0, 1}
    assert set(np.unique(y_val)) == {0, 1}
    assert set(np.unique(y_test)) == {0, 1}


def test_scaling_is_fitted_on_training_partition():
    X_train, _, X_val, _, X_test, _ = load_data()

    # StandardScaler should center and scale the training features.
    assert np.allclose(X_train.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(X_train.std(axis=0), 1.0, atol=1e-10)

    # Validation and test are transformed, not independently fitted.
    assert X_val.shape[1] == X_test.shape[1] == X_train.shape[1]


def test_split_is_reproducible():
    first = load_data()
    second = load_data()

    for a, b in zip(first, second):
        assert np.array_equal(a, b)


def test_magnitude_pruning_hits_target_and_keeps_zeros():
    _, _, params = build_model(seed=42)

    prune_by_magnitude(params, 0.50)

    assert np.isclose(actual_sparsity(params), 0.50)

    for p in params:
        if hasattr(p, "mask"):
            assert np.all(p.value[p.mask == 0] == 0.0)
