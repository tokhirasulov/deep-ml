import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr):    
    history = []
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    N = len(X_train)
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(N)
        x_shfl = X_train[perm]
        y_shfl = y_train[perm]
        model.train()
        total = 0.0
        for start in range(0, N, batch_size):
            x_batch = x_shfl[start:start+batch_size]
            y_batch = y_shfl[start:start+batch_size]
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x_batch)
        train_loss = total / N

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = loss_fn(val_logits, y_val).item()

            preds = val_logits.argmax(dim=1)
            val_acc = (preds == y_val).float().mean().item()
        history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_acc
        })
    return history

