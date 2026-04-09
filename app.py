
import os
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import timm
import gradio as gr
import torch.nn as nn
import torch.optim as optim
import os
os.system("pip install torchvision")

# Hardcoded classes as they are derived from the dataset during training
classes = ['mild', 'moderate', 'no_dr', 'proliferate_dr', 'severe']
idx_to_class = {idx: cls for idx, cls in enumerate(classes)}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Enhanced preprocessing for retina images
def crop_retina_fundus(image):
    '''Remove black/dark borders and crop to fundus region'''
    img_array = np.array(image)

    # Convert to grayscale for border detection
    if len(img_array.shape) == 3:
        gray = np.mean(img_array, axis=2).astype(np.uint8) 
    else:
        gray = img_array.astype(np.uint8)

    # Find non-black pixels (threshold at 30 to account for dark areas)
    threshold = 30
    non_black = gray > threshold

    # Find bounding box of fundus region
    rows = np.any(non_black, axis=1)
    cols = np.any(non_black, axis=0)

    if rows.any() and cols.any():
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        # Crop with small margin
        margin = 5
        ymin = max(0, ymin - margin)
        xmin = max(0, xmin - margin)
        ymax = min(img_array.shape[0], ymax + margin)
        xmax = min(img_array.shape[1], xmax + margin)

        return Image.fromarray(img_array[ymin:ymax, xmin:xmax])

    return image

val_transforms = transforms.Compose([
    transforms.Lambda(crop_retina_fundus),
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocess(image):
    '''Preprocess a single image for inference'''
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))
    
    return val_transforms(image).unsqueeze(0).to(device)

# Load pretrained EfficientNet model (pretrained=False as we load our weights)
model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=len(classes))
model = model.to(device)

# Load the trained model weights
try:
    # In Hugging Face Spaces, the model file will be in the same directory as app.py
    model.load_state_dict(torch.load('retina_efficientnet_final.pth', map_location=device))
    model.eval()
    print("✓ Model loaded successfully for inference")
except FileNotFoundError:
    print("✗ Error: Model file 'retina_efficientnet_final.pth' not found. Please ensure it's uploaded to your Hugging Face Space.")
    # Exit or handle error gracefully, e.g., by using a dummy model
    exit()

def predict(image):
    '''Predict diabetic retinopathy stage from uploaded image'''
    model.eval()
    with torch.no_grad():
        img_tensor = preprocess(image)
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        pred = torch.argmax(probabilities, dim=1)
        confidence = probabilities[0][pred].item()
    
    predicted_class = idx_to_class[pred.item()]
    return f"{predicted_class} (Confidence: {confidence:.2%})"

# Create Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Retina Image"),
    outputs=gr.Textbox(label="Prediction Result"),
    title="Diabetic Retinopathy Detection",
    description="Upload a retina fundus image to detect diabetic retinopathy stage",
    theme="default"
)

# Launch the interface (for Hugging Face Spaces, this will be run automatically)
if __name__ == "__main__":
    interface.launch(share=False) # share=False for local/deployment context
