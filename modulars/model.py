from torchvision.models import vgg19, VGG19_Weights
import torch.nn as nn

class VGGModified(nn.Module):
    def __init__(self,):
        super(VGGModified, self).__init__()
        self.pretrained_weights = VGG19_Weights.DEFAULT
        self.vgg_model = vgg19(weights=self.pretrained_weights).features[:29]
        self.chosen_features = ["0", "5", "10", "19", "28"]

    def forward(self, x):
        # Relevant features
        features = []
        for layer_index, layer in enumerate(self.vgg_model):
            x = layer(x)
            if str(layer_index) in self.chosen_features:
                features.append(x)

        return features