import torch
import torch.nn as nn
import torch.nn.functional as F


class Siamese(nn.Module):
    """
    Siamese Network for One-Shot Learning as described in Koch et al., 2015.
    """
    def __init__(self, embedding_dim=4096):
        super(Siamese, self).__init__()
        
        # CNN backbone (following paper's Figure 4)
        self.features = nn.Sequential(
            # Conv layer 1: 64 filters, 10x10 kernel
            nn.Conv2d(3, 64, kernel_size=10, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv layer 2: 128 filters, 7x7 kernel  
            nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv layer 3: 128 filters, 4x4 kernel
            nn.Conv2d(128, 128, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv layer 4: 256 filters, 4x4 kernel
            nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 6))
        )
        
        # Fully connected layers (4096 units)
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, embedding_dim),
            nn.Sigmoid(), 
            nn.Dropout(0.5)
        )
        
        # L1 distance layer with learnable weights αⱼ
        self.similarity = nn.Linear(embedding_dim, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize weights for convolutional and linear layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.normal_(m.bias, mean=0.5, std=0.01)
            elif isinstance(m, nn.Linear):
                if m == self.classifier[0]: # FC layer
                    nn.init.normal_(m.weight, mean=0, std=0.2)
                    nn.init.normal_(m.bias, mean=0.5, std=0.01)

    def forward_once(self, x):
        """
        Forward pass through the CNN and FC layers for one input.
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x
    
    def forward(self, input1, input2):
        """
        Forward pass for two inputs through the Siamese network.
        """
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        
        # Compute L1 distance
        l1_distance = torch.abs(output1 - output2)
        
        # Compute similarity score
        similarity = torch.sigmoid(self.similarity(l1_distance))
        
        return similarity