import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import matplotlib.pyplot as plt

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: cat, dog
model.load_state_dict(torch.load("../models/resnet18_catsdogs.pth", map_location=device))
model = model.to(device)
model.eval()

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Directories
inference_dir = "../data/inference data/"

# Map indices to labels
idx_to_class = {0: "cat", 1: "dog"}

# Run inference and collect results
print("Running inference on images in:", inference_dir)
results = []
for img_file in os.listdir(inference_dir):
    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Ground truth from filename (cat/dog prefix)
        gt_label = "cat" if img_file.startswith("cat") else "dog"

        # Load and preprocess
        img_path = os.path.join(inference_dir, img_file)
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            pred_label = idx_to_class[pred_idx]

        # Print result
        print(f"{img_file}: Predicted = {pred_label} ({probs[pred_idx]:.2f}), Ground Truth = {gt_label}")

        # Save for visualization
        results.append((image, pred_label, gt_label))

        # Limit to 10 images for grid
        if len(results) == 10:
            break

# Plot results in a grid
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

for ax, (image, pred, gt) in zip(axes, results):
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(f"P: {pred} | GT: {gt}", fontsize=10,
                 color=("green" if pred == gt else "red"))

plt.tight_layout()

save_path = "../results/inference_results.png"
plt.savefig(save_path)
plt.show()
