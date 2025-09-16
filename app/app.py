import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    # Recreate ResNet18 architecture (same as training)
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: cat & dog

    # Load weights
    state_dict = torch.load(
        "../models/resnet18_catsdogs.pth",  # adjust path if needed
        map_location=torch.device("cpu")
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

# Class labels
classes = ["cat", "dog"]

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# Streamlit UI
st.title("🐶🐱 Cat vs Dog Classifier")
st.write("Upload an image to classify it as **Cat** or **Dog** with probabilities.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Preprocess
    img_tensor = transform(image).unsqueeze(0)

    # Inference
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        pred_label = classes[pred_idx]

    # Side-by-side layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded Image", width=250)

    with col2:
        st.subheader(f"Prediction: **{pred_label.upper()}** 🐾")
        st.write(f"Cat Probability: {probs[0].item():.4f}")
        st.write(f"Dog Probability: {probs[1].item():.4f}")
