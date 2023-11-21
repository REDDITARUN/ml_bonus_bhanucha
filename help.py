import numpy as np
from PIL import Image, ImageOps
import torch 
import torch.nn as nn
import csv
import streamlit as st


class VGG_11_CNN(nn.Module):
    def __init__(self, classes):
        super(VGG_11_CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.lin = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, classes)
        )


    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.lin(x)

        return x



def process_image(img):
    img = ImageOps.grayscale(img)  # Use ImageOps for grayscale conversion
    img = img.resize((28, 28))
    img_array = np.asarray(img) / 255.0
    img_array = img_array.reshape((1, 1, 32, 32))  # Adjust the shape for PyTorch
    st.write("Returning Pre processed...")
    return img_array

def prediction_result(model, image_data):
    reader = csv.DictReader(open("datamap.csv"))
    classes = {}
    for row in reader:
        k = int(row["ClassIndex"])
        v = row["ClassName"]
        classes[k] = v
    
    with torch.no_grad():
        output = model(torch.Tensor(image_data))
    
    prediction = torch.argmax(output, axis=1).item()
    
    prediction_class = {"class": classes[prediction]}
    return prediction_class