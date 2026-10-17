
import numpy as np

from engine.tensor import Tensor, softmax_cross_entropy
from nn.layers import Linear
from nn.optim import SGD
from nn.pruning import prune_global


def make_dataset(n_samples=1200, seed=42):
    """Create a synthetic XOR-like binary classification dataset."""
    rng = np.random.default_rng(seed)

    X = rng.uniform(-1.0, 1.0, size=(n_samples, 2))
    y = (X[:, 0] * X[:, 1] > 0).astype(np.int64)

    return X, y


def forward(layer1, layer2, x):
    """Forward pass through the two-layer MLP."""
    x = Tensor(x)
    hidden = layer1(x).relu()
    logits = layer2(hidden)
    return logits


def evaluate(layer1, layer2, X, y):
    """Calculate cross-entropy loss and classification accuracy."""
    logits = forward(layer1, layer2, X)
    loss = softmax_cross_entropy(logits, y)

    predictions = np.argmax(logits.value, axis=1)
    accuracy = np.mean(predictions == y) * 100.0

    return float(loss.value.item()), float(accuracy)


def train_epoch(
    layer1,
    layer2,
    optimizer,
    X,
    y,
    batch_size=32,
):
    """Train for one epoch using randomly shuffled mini-batches."""
    n_samples = len(X)
    indices = np.random.permutation(n_samples)

    for start in range(0, n_samples, batch_size):
        batch_indices = indices[start:start + batch_size]

        X_batch = X[batch_indices]
        y_batch = y[batch_indices]

        optimizer.zero_grad()

        logits = forward(layer1, layer2, X_batch)
        loss = softmax_cross_entropy(logits, y_batch)

        loss.backward()
        optimizer.step()


def calculate_sparsity(parameters):
    """Return the fraction of masked-out weight entries."""
    masks = [
        p.mask
        for p in parameters
        if hasattr(p, "mask")
    ]

    if not masks:
        return 0.0

    total_weights = sum(mask.size for mask in masks)
    remaining_weights = sum(
        int(np.count_nonzero(mask)) for mask in masks
    )

    return 1.0 - remaining_weights / total_weights


def refresh_importance(layer1, layer2, X, y, optimizer):
    """Recalculate weight gradients using the full training set."""
    optimizer.zero_grad()

    logits = forward(layer1, layer2, X)
    loss = softmax_cross_entropy(logits, y)

    loss.backward()

    return float(loss.value.item())


def main():
    # Reproducible experiment.
    np.random.seed(42)

    # ---------------------------------------------------------
    # 1. Prepare dataset
    # ---------------------------------------------------------
    X, y = make_dataset(n_samples=1200, seed=42)

    split = int(0.8 * len(X))

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # ---------------------------------------------------------
    # 2. Build model and optimizer
    # ---------------------------------------------------------
    layer1 = Linear(2, 16)
    layer2 = Linear(16, 2)

    parameters = layer1.parameters() + layer2.parameters()

    optimizer = SGD(
        parameters,
        lr=0.1,
    )

    # ---------------------------------------------------------
    # 3. Train dense baseline
    # ---------------------------------------------------------
    print("Training dense baseline...")

    baseline_epochs = 100

    for epoch in range(baseline_epochs):
        train_epoch(
            layer1,
            layer2,
            optimizer,
            X_train,
            y_train,
            batch_size=32,
        )

        if (epoch + 1) % 10 == 0:
            train_loss, train_acc = evaluate(
                layer1, layer2, X_train, y_train
            )
            test_loss, test_acc = evaluate(
                layer1, layer2, X_test, y_test
            )

            print(
                f"Epoch {epoch + 1:3d}/{baseline_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Test Loss: {test_loss:.4f} | "
                f"Test Accuracy: {test_acc:.2f}%"
            )

    baseline_loss, baseline_accuracy = evaluate(
        layer1, layer2, X_test, y_test
    )

    print("\nDense baseline")
    print(f"Test loss:     {baseline_loss:.4f}")
    print(f"Test accuracy: {baseline_accuracy:.2f}%")

    # ---------------------------------------------------------
    # 4. Gradual pruning and fine-tuning
    # ---------------------------------------------------------
    print("\nStarting gradual pruning...")

    sparsity_targets = [0.10, 0.20, 0.30, 0.40, 0.50]
    fine_tune_epochs = 10

    for target_sparsity in sparsity_targets:
        # Recalculate importance for the current model.
        refresh_importance(
            layer1,
            layer2,
            X_train,
            y_train,
            optimizer,
        )

        # Prune toward the requested overall sparsity.
        prune_global(
            parameters,
            target_sparsity=target_sparsity,
        )

        actual_sparsity = calculate_sparsity(parameters)

        loss_before, accuracy_before = evaluate(
            layer1, layer2, X_test, y_test
        )

        print(
            f"\nTarget sparsity: {target_sparsity:.0%} | "
            f"Actual sparsity: {actual_sparsity:.2%}"
        )
        print(
            f"Before fine-tuning: "
            f"loss={loss_before:.4f}, "
            f"accuracy={accuracy_before:.2f}%"
        )

        # Allow the surviving weights to adapt.
        for _ in range(fine_tune_epochs):
            train_epoch(
                layer1,
                layer2,
                optimizer,
                X_train,
                y_train,
                batch_size=32,
            )

        loss_after, accuracy_after = evaluate(
            layer1, layer2, X_test, y_test
        )

        print(
            f"After fine-tuning:  "
            f"loss={loss_after:.4f}, "
            f"accuracy={accuracy_after:.2f}%"
        )

    # ---------------------------------------------------------
    # 5. Final report
    # ---------------------------------------------------------
    final_loss, final_accuracy = evaluate(
        layer1, layer2, X_test, y_test
    )

    final_sparsity = calculate_sparsity(parameters)

    print("\n" + "=" * 48)
    print("FINAL RESULTS")
    print("=" * 48)

    print(f"Dense baseline accuracy: {baseline_accuracy:.2f}%")
    print(f"Pruned model accuracy:   {final_accuracy:.2f}%")
    print(f"Accuracy change:         {final_accuracy - baseline_accuracy:+.2f} percentage points")
    print(f"Final test loss:         {final_loss:.4f}")
    print(f"Final weight sparsity:   {final_sparsity:.2%}")
    print("=" * 48)

    if abs(final_sparsity - 0.50) > 1e-6:
        print(
            "WARNING: Final sparsity is not 50%. "
            "Check the behavior of prune_global()."
        )


if __name__ == "__main__":
    main()
