# pipeline.py
import json
import toml
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as T
from inference import create_resnet, DEVICE
import json
from datetime import datetime
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_transforms(train=True):
    if train:
        return T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
    else:
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

def get_dataloaders(cfg):
    data_cfg = cfg["data"]
    batch = data_cfg.get("batch_size", 32)
    train_ds = datasets.ImageFolder(data_cfg["train_path"], transform=get_transforms(train=True))
    val_ds = datasets.ImageFolder(data_cfg["val_path"], transform=get_transforms(train=False))
    test_ds = datasets.ImageFolder(data_cfg["test_path"], transform=get_transforms(train=False))
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=data_cfg.get("num_workers", 4))
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=data_cfg.get("num_workers", 4))
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=data_cfg.get("num_workers", 4))
    return train_loader, val_loader, test_loader

def build_optimizer(model, opt_name, lr, momentum, weight_decay):
    if opt_name.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif opt_name.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError("Unsupported optimizer: " + opt_name)

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for x,y in loader:
        x = x.to(DEVICE); y = y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * x.size(0)
        preds = out.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total += x.size(0)
    return total_loss/total, total_correct/total

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for x,y in loader:
            x = x.to(DEVICE); y = y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)
            total_loss += float(loss.item()) * x.size(0)
            preds = out.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total += x.size(0)
    return total_loss/total, total_correct/total


def run_pipeline(json_path="pipeline_config.json", toml_path="model_params.toml", out_path=None):
    cfg = json.load(open(json_path))
    params = toml.load(open(toml_path))

    set_seed(cfg["experiment"].get("seed", 42))

    model_name = cfg["model"]["name"]
    num_classes = cfg["model"].get("num_classes", 1000)
    pretrained = cfg["model"].get("pretrained", True)

    model = create_resnet(model_name, pretrained=pretrained, num_classes=num_classes)

    arch_params = params.get(model_name, {})
    defaults = params.get("defaults", {})
    lr = arch_params.get("learning_rate", defaults.get("learning_rate", arch_params.get("learning_rate", 1e-3)))
    optimizer_name = arch_params.get("optimizer", "sgd")
    momentum = arch_params.get("momentum", 0.0)
    weight_decay = defaults.get("weight_decay", 0.0)
    epochs = defaults.get("epochs", 5)

    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, optimizer_name, lr, momentum, weight_decay)

    best_val_acc = 0.0
    for epoch in range(1, epochs+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f"[{model_name}] Epoch {epoch}/{epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"best_{model_name}.pt")

    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print("Test accuracy:", test_acc)

    # Prepare result dictionary
    result = {
        "model": model_name,
        "num_classes": num_classes,
        "pretrained": bool(pretrained),
        "architecture_params": {
            "learning_rate": lr,
            "optimizer": optimizer_name,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "epochs": epochs
        },
        "metrics": {
            "best_val_acc": best_val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc
        },
        "saved_checkpoint": os.path.abspath(f"best_{model_name}.pt") if best_val_acc > 0 else None,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    # Write results to file if out_path provided (or create a default timestamped file)
    if out_path is None:
        out_path = f"Result/pipeline_result_{model_name}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Results written to: {out_path}")
    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run training pipeline and save results")
    parser.add_argument("--config", "-c", default="pipeline_config.json", help="path to pipeline JSON config")
    parser.add_argument("--params", "-p", default="model_params.toml", help="path to TOML params")
    parser.add_argument("--out", "-o", default=None, help="output result JSON path (optional)")
    args = parser.parse_args()
    run_pipeline(json_path=args.config, toml_path=args.params, out_path=args.out)


# python pipeline.py --config my_cfg.json --params my_params.toml --out results/my_run.json