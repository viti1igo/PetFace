import torch
import torch.nn as nn
from torchvision import models

# Set device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

class EfficientNet(nn.Module):
    def __init__(self, embedding_dim=224):
        super().__init__()
        
        # EfficientNet-B0 backbone
        backbone = models.efficientnet_b0(pretrained=True)
        self.feature_extractor = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Embedding head
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 512),  # EfficientNet-B0 outputs 1280 features
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = self.embedding(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x