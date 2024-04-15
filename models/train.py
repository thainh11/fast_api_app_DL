import torch
from catdog_model import *
from dataset import *
from tqdm import tqdm

TRAIN_BATCH_SIZE = 8
VAL_BACTH_SIZE = 8

train_dataset = CatsVsDogsDataset(
    datasets['train'], 
    transforms=img_transforms
)
test_dataset = CatsVsDogsDataset(
    datasets['test'],
    transforms=img_transforms
)

train_loader = DataLoader(
    train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=VAL_BACTH_SIZE,
    shuffle=False
)

EPOCHS = 3
LR = 1e-3
WEIGHT_DECAY = 1e-5
N_CLASSES = 2
device = "cuda"
model = CatsVsDogsModels(N_CLASSES).to(device)
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=LR, 
    weight_decay=WEIGHT_DECAY
)
criterion = torch.nn.CrossEntropyLoss()


if __name__ == '__main__':
    for epoch in tqdm(range(EPOCHS), desc="Epochs", colour='cyan'):
    # for epoch in range(EPOCHS):
        train_losses = []
        model.train()
        for images, labels in tqdm(train_loader, desc="Images_train", colour='cyan'):
        # for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        train_loss = sum(train_losses) / len(train_losses)
        
        val_losses = []
        model.eval()
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Images_eval", colour='cyan'):
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_losses.append(loss.item())
            val_loss = sum(val_losses) / len(val_losses)
            
            print(f"Epoch: {epoch + 1}, train_loss: {train_loss}, val_loss: {val_loss}")

    SAVE_PATH = "D:\Work\\fast_api_app_DL\models\weights\cats_vs_dogs.pt"
    torch.save(model.state_dict(), SAVE_PATH) 