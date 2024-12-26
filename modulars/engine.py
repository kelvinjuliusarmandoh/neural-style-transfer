"""
Script for training the model
"""

import torch
from modulars.utils import calculate_content_loss, calculate_style_loss, calculate_total_loss, calculate_gram_matrix
from modulars.utils import load_image
from torchvision.utils import save_image
from tqdm.auto import tqdm

def engine(vgg_model, 
           content_image, 
           style_image, 
           total_steps: int,
           image_size: int,
           alpha: int,
           betha: int,
           learning_rate: float,
           device: torch.device):
    """
    Run the training process and return the result image.
    """
    content_image = load_image(content_image, size=image_size).to(device)
    style_image = load_image(style_image, size=image_size).to(device)
    generated_image = content_image.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([generated_image], lr=learning_rate)

    for step in tqdm(range(total_steps)):
        # Get convolution features
        content_features = vgg_model(content_image)
        style_features = vgg_model(style_image)
        generated_features = vgg_model(generated_image)

        content_loss, style_loss = 0, 0

        for content_features, style_features, gen_features in zip(content_features, style_features, generated_features):
            content_loss += calculate_content_loss(content_features, gen_features)
            style_loss += calculate_style_loss(style_features, gen_features)

        total_loss = calculate_total_loss(alpha_=alpha,
                                        content_loss=content_loss,
                                        betha_=betha,
                                        style_loss=style_loss)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step%100 == 0:
            print(f"Content Loss: {content_loss:.3f} | Style Loss: {style_loss:.3f} | Total Loss: {total_loss:.3f}")
            save_image(generated_image, "generated_out6.png")
        
    return generated_image