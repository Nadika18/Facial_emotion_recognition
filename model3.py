from PIL import Image
import os
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torchvision.transforms.functional as TF
import seaborn as sns

# Define the emotion categories
emotion_categories = {
    1: "Neutral",
    2: "Happiness",
    3: "Sadness",
    4: "Surprise",
    5: "Afraid",
    6: "Disgusted",
    7: "Angry",
    8: "Contempt"
}

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(EmotionCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 6 * 6, 512),  # Adjusted input size (128 * 6 * 6)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN(num_classes=8).to(device)
model.load_state_dict(torch.load("model4.h5"))
model.eval()

# Function to decode emotion labels
def decode_emotion_label(label):
    return emotion_categories[label]

# Function to create a radar chart for emotion categories and probabilities
def radar_chart_for_image(model, image):
    model.eval()
    image = image.to(device)
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        probabilities = torch.softmax(output, dim=1).squeeze().cpu().numpy()

    angles = np.linspace(0, 2 * np.pi, len(emotion_categories), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    probabilities = np.concatenate((probabilities, [probabilities[0]]))

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, probabilities, 'o-', linewidth=2)
    ax.fill(angles, probabilities, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(list(emotion_categories.values()))
    ax.set_title("Emotion Probabilities", fontsize=14)
    ax.grid(True)
