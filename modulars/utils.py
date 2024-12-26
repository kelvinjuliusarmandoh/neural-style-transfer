from typing import Tuple
import torch
import torchvision.transforms as transforms
from PIL import Image

# Define preprocess image
def preprocess_image(image: int, size: Tuple):
    transformer = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_transformed = transformer(image)
    return image_transformed

def load_image(image_path, size):
    image = Image.open(image_path)
    image = preprocess_image(image, size)
    return image

# Define content content loss, gram matrix, style loss, and total loss

## Content loss
def calculate_content_loss(content_feature, generated_feature):
    return torch.mean((generated_feature - content_feature)**2)
    
# Gram Matrix Function
def calculate_gram_matrix(feature):
    c, h, w = feature.size()
    feature = feature.view(c, h*w) # flattened
    gram_mat = torch.mm(feature, feature.T)
    return gram_mat

# Style Loss
def calculate_style_loss(style_feature, generated_feature):
    # Gram Matrix for Style feature
    style_gram_matrix = calculate_gram_matrix(style_feature)

    # Gram Matrix for Generated feature
    generated_gram_matrix = calculate_gram_matrix(generated_feature)

    # MSE for both of them
    style_loss = torch.mean((generated_gram_matrix - style_gram_matrix)**2)
    return style_loss

# Total Loss
def calculate_total_loss(alpha_, content_loss, betha_, style_loss):
    return alpha_*content_loss + betha_*style_loss
