# grid_search.py
import itertools
import torch
from inference import create_resnet, DEVICE
from pipeline import get_dataloaders, train_one_epoch, evaluate, build_optimizer, set_seed
import torch.nn as nn

def run_single(cfg, arch_name, lr, opt_name, momentum, epochs=3):
    model = create_resnet(arch_name, pretrained=cfg["model"].get("pretrained", True), num_classes=cfg["model"].get("num_classes", 1000))
    train_loader, val_loader, _ = get_dataloaders({"data": {**cfg["data"], "batch_size": cfg["data"].get("batch_size",32)}})
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, opt_name, lr, momentum, weight_decay=0.0)
    best_val = 0.0
    for epoch in range(epochs):
        train_one_epoch(model, train_loader, criterion, optimizer)
        _, val_acc = evaluate(model, val_loader, criterion)
        if val_acc > best_val:
            best_val = val_acc
    return best_val

def grid_search(grid_json_path="grid_search.json"):
    import json
    cfg = json.load(open(grid_json_path))
    grid = cfg["grid"]
    arch = cfg["model"]["name"]
    epochs = cfg.get("max_epochs", 3)
    results = []
    keys = list(grid.keys())
    all_vals = [grid[k] for k in keys]
    combinations = list(itertools.product(*all_vals))
    print(f"Running grid search with {len(combinations)} combinations")
    for combo in combinations:
        combo_dict = dict(zip(keys, combo))
        lr = combo_dict.get("learning_rate")
        opt = combo_dict.get("optimizer")
        mom = combo_dict.get("momentum", 0.0)
        print("Testing:", combo_dict)
        set_seed(42)
        val_acc = run_single(cfg, arch, lr, opt, mom, epochs=epochs)
        result = {"params": combo_dict, "val_acc": val_acc}
        results.append(result)
        print("Result:", result)
    with open("Result/grid_results.json", "w") as f:
        import json
        json.dump(results, f, indent=2)
    return results

if __name__ == "__main__":
    grid_search()
