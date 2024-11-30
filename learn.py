import torch
import torchvision.transforms.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from detect import get_model
from coco import train_loader

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model(num_classes=91)  
model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-4)

# Обучение
for epoch in range(10):
    model.train()
    epoch_loss = 0
    for images, targets in tqdm(train_loader):
        if len(images) == 0 or len(targets) == 0:
            continue  

        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        epoch_loss += losses.item()
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {epoch_loss}")

    torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")
    print(f"Model saved at epoch {epoch + 1}")
