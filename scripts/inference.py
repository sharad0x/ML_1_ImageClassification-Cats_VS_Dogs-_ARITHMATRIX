import os
from PIL import Image
import torch
from torchvision import transforms, models

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
model_path = r"..\models\resnet18_catsdogs.pth"
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)

# Classes
classes = ['cat', 'dog']

# Transform
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Test images folder
test_dir = r"..\data\test"
test_images = [f for f in os.listdir(test_dir) if f.endswith(('.png','.jpg','.jpeg'))]

# Inference
for img_file in test_images[:10]:
    img_path = os.path.join(test_dir, img_file)
    image = Image.open(img_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
    print(f"{img_file}: Predicted = {classes[pred.item()]}")
