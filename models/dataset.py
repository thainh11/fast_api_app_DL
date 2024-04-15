import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# Load dataset, if you want to change dataset, change DATASET_NAME
DATASET_NAME = "cats_vs_dogs"
datasets = load_dataset(DATASET_NAME, ignore_verifications=True, cache_dir="D:\Work\\fast_api_app_DL\models")

TEST_SIZE = 0.2
datasets = datasets['train'].train_test_split(test_size=TEST_SIZE, shuffle=True)

IMG_SIZE = 64
img_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])

class CatsVsDogsDataset(Dataset):
    def __init__(self, dataset, transforms=None):
        self.dataset = dataset
        self.transforms = transforms
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        images = self.dataset[index]['image']
        labels = self.dataset[index]['labels']
        if self.transforms:
            images = self.transforms(images)
        labels = torch.tensor(labels, dtype=torch.long)
        
        return images, labels
