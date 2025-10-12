import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class Trainer:
    """
    Following the training procedure from Koch et al., 2015.
    """
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        print("DEVICE USING:", device)

        # loss function
        self.criterion = nn.BCELoss()

        # layer-wise learning rates
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=0.0001,
            momentum=0.5,
            weight_decay=0.0005 # L2 regularization
        )

        # 1% learning rate decay every epoch
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

    def train_epoch(self, train_loader, epoch):
        """
        Training the model.
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        if epoch < 100:
            momentum = 0.5 + (0.9 - 0.5) * epoch / 100
            for param_group in self.optimizer.param_groups:
                param_group['momentum'] = momentum

        for batch_idx, (img1, img2, labels) in enumerate(train_loader):
            img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(img1, img2).squeeze()
            loss = self.criterion(outputs.squeeze(), labels.float())
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels.float()).sum().item()
        
        # 1% learning rate decay
        self.scheduler.step()
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        print(f"Train Epoch: {epoch} \tLoss: {avg_loss:.6f} \tAccuracy: {accuracy:.2f}%")
        return avg_loss, accuracy

def get_paper_transforms():
    """
    Affine distortions as per Koch et al., 2015.
    """
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Adapt to modern input size
        transforms.RandomAffine(
            degrees=10,              # θ ∈ [-10.0, 10.0]
            translate=(0.1, 0.1),    # tx, ty ∈ [-2, 2] (normalized)
            scale=(0.8, 1.2),        # sx, sy ∈ [0.8, 1.2]
            shear=(-0.3, 0.3),       # ρx, ρy ∈ [-0.3, 0.3]
            fill=255                 # White background
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform