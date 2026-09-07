from src.data.dataloader import get_dataloader

train_loader, val_loader, test_loader = get_dataloader(
    batch_size=64,
    task="classification"
)

print(f"Train samples: {len(train_loader.dataset)}")
print(f"Validation samples: {len(val_loader.dataset)}")
print(f"Test samples: {len(test_loader.dataset)}")

X, y, failure = next(iter(train_loader))

print("\nBatch:")
print("X:", X.shape)
print("y:", y.shape)
print("failure:", failure.shape)