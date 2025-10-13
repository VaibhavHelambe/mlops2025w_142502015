# inference.py
import torch
import torchvision.transforms as T
from PIL import Image
import torchvision.models as models
from typing import Union, List
import torchvision

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_resnet(name: str, pretrained: bool = True, num_classes: int = 1000):
    name = name.lower()
    if name == "resnet34":
        model = models.resnet34(weights=pretrained)
    elif name == "resnet50":
        model = models.resnet50(weights=pretrained)
    elif name == "resnet101":
        model = models.resnet101(weights=pretrained)
    elif name == "resnet152":
        model = models.resnet152(weights=pretrained)
    else:
        raise ValueError(f"Unknown resnet variant: {name}")
    if num_classes != 1000:
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
    model.to(DEVICE)
    return model

INFER_TRANSFORMS = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return INFER_TRANSFORMS(img).unsqueeze(0)

def predict(model: torch.nn.Module, input_tensor: torch.Tensor, topk: int = 5):
    model.eval()
    input_tensor = input_tensor.to(DEVICE)
    with torch.no_grad():
        out = model(input_tensor)
        probs = torch.nn.functional.softmax(out, dim=1)
        top_probs, top_idx = probs.topk(topk, dim=1)
    return top_probs.cpu().numpy(), top_idx.cpu().numpy()

if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 3:
        print("Usage: python inference.py <resnet34|resnet50|resnet101|resnet152> <image_path> [output_file]")
        sys.exit(1)

    variant = sys.argv[1]
    img_path = sys.argv[2]
    out_file = sys.argv[3] if len(sys.argv) > 3 else "Result/inference_results.json"

    # Load model and image
    model = create_resnet(variant, pretrained=True)
    x = load_image(img_path)
    probs, idx = predict(model, x, topk=5)

    # Load ImageNet class names from torchvision
    try:
        imagenet_classes = torchvision.models.ResNet_Weights.DEFAULT.meta["categories"]
    except Exception:
        # Fallback if meta is not available
        imagenet_classes = [str(i) for i in range(1000)]

    # Prepare result dictionary
    result = {
        "model": variant,
        "image": img_path,
        "predictions": [
            {
                "class_index": int(idx[0][i]),
                "class_name": imagenet_classes[int(idx[0][i])] if int(idx[0][i]) < len(imagenet_classes) else str(idx[0][i]),
                "probability": float(probs[0][i])
            }
            for i in range(len(idx[0]))
        ]
    }

    # Print to console
    print(json.dumps(result, indent=2))

    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    # Save to file
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Results saved to {out_file}")

    # python inference.py resnet50 data/test/cat/cat_0.png