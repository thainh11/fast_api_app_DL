import torch
import torch.nn as nn
from torchvision.models import resnet18

class CatsVsDogsModels(nn.Module):
    def __init__(self, n_classes):
        super(CatsVsDogsModels, self).__init__()
        resnet_model = resnet18(weights="IMAGENET1K_V1")
        self.backbone = nn.Sequential(
            *list(resnet_model.children())[:-1],
        )
        for param in self.backbone.parameters():
            param.requires_grad = False
        in_features = resnet_model.fc.in_features
        self.fc = nn.Linear(in_features, n_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

if __name__ == '__main__':  
    device = "cuda"
    N_CLASSES = 2
    model = CatsVsDogsModels(N_CLASSES).to(device)
    test_input = torch.rand(1, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(test_input)
        print(output.shape)