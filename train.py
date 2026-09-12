"""
Training and evaluation functions used for ResNet-50, ViT-Small, and Zoobot.
Extracted from the project's main Colab notebook.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torchmetrics.classification import MulticlassCalibrationError
import matplotlib.pyplot as plt

NUM_CLASSES = 10


def train_model(model, model_name, train_loader, val_loader, device,
                 epochs=30, lr=1e-4):
    """
    Trains any model using the same settings.
    Saves best weights to disk during training.
    Returns the model and training history.
    """
    model = model.to(device)

    # AdamW optimizer -- works well for both CNNs and Transformers
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Gradually reduces learning rate as training progresses
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Loss function -- label_smoothing=0.1 prevents overconfidence
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):

        # ---- TRAINING PHASE ----
        model.train()
        t_loss = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        scheduler.step()

        # ---- VALIDATION PHASE ----
        model.eval()
        correct, total, v_loss = 0, 0, 0.0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                v_loss += criterion(out, lbls).item()
                correct += (out.argmax(1) == lbls).sum().item()
                total += lbls.size(0)

        val_acc = correct / total

        history['train_loss'].append(t_loss / len(train_loader))
        history['val_loss'].append(v_loss / len(val_loader))
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_{model_name}.pth')

        if (epoch + 1) % 5 == 0:
            print(f"[{model_name}] Ep {epoch+1}/{epochs} | "
                  f"TrainLoss {t_loss/len(train_loader):.4f} | "
                  f"ValAcc {val_acc:.4f}")

    print(f"\n[{model_name}] Best ValAcc: {best_val_acc:.4f}")
    return model, history


def evaluate_model(model, model_name, loader, device):
    """
    Evaluates a trained model on the test set.
    Returns metrics dict + probability tensor + true labels tensor.
    """
    model.eval()
    all_preds, all_true, all_probs = [], [], []

    with torch.no_grad():
        for imgs, lbls in loader:
            out = model(imgs.to(device))
            probs = torch.softmax(out, dim=1)
            all_probs.append(probs.cpu())
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_true.extend(lbls.numpy())

    probs_t = torch.cat(all_probs)
    true_t = torch.tensor(all_true)

    acc = accuracy_score(all_true, all_preds)
    f1 = f1_score(all_true, all_preds, average='macro')
    prec = precision_score(all_true, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_true, all_preds, average='macro')

    ece_fn = MulticlassCalibrationError(num_classes=NUM_CLASSES, n_bins=15)
    ece = ece_fn(probs_t, true_t).item()

    params = sum(p.numel() for p in model.parameters()) / 1e6

    result = {
        'Model': model_name,
        'Accuracy': round(acc * 100, 2),
        'Precision': round(prec * 100, 2),
        'Recall': round(rec * 100, 2),
        'F1-macro': round(f1 * 100, 2),
        'ECE': round(ece, 4),
        'Params(M)': round(params, 2)
    }
    print("\nTest Results:")
    print(result)
    return result, probs_t, true_t


def plot_training_curves(history, model_name):
    """
    Plots training loss and validation accuracy over epochs.
    Saves the figure as a PNG file.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_title(f'{model_name} -- Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(history['val_acc'], color='green', label='Val Accuracy')
    ax2.set_title(f'{model_name} -- Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f'curves_{model_name}.png', dpi=300)
    plt.show()
    print(f"File saved: curves_{model_name}.png")


def train_zoobot(zoobot_model, train_loader, val_loader, device):
    """
    Zoobot-specific two-stage training:
    Stage 1: frozen encoder, head-only, 10 epochs, lr=1e-3
    Stage 2: full fine-tune, 20 epochs, lr=1e-5
    """
    # ---- STAGE 1 ----
    for param in zoobot_model.encoder.parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in zoobot_model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in zoobot_model.parameters() if not p.requires_grad)
    print(f"Trainable : {trainable/1e6:.4f}M  <- head only")
    print(f"Frozen    : {frozen/1e6:.2f}M    <- pretrained encoder")

    zoobot_model, hist_s1 = train_model(
        zoobot_model, 'Zoobot_stage1', train_loader, val_loader, device,
        epochs=10, lr=1e-3
    )

    # ---- STAGE 2 ----
    for param in zoobot_model.encoder.parameters():
        param.requires_grad = True

    zoobot_model, hist_s2 = train_model(
        zoobot_model, 'Zoobot', train_loader, val_loader, device,
        epochs=20, lr=1e-5
    )

    return zoobot_model, hist_s1, hist_s2
