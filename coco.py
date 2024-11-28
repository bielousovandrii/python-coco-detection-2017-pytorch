from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CocoDetection
import torch
import random


class CustomCocoDataset(CocoDetection):
    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        
        # Преобразование аннотаций
        boxes = []
        labels = []
        for obj in target:
            xmin, ymin, w, h = obj["bbox"]
            if w > 0 and h > 0:  # Проверка на корректность размеров
                xmax = xmin + w
                ymax = ymin + h
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(obj["category_id"])
        
        # Если аннотации отсутствуют, вернуть None
        if len(boxes) == 0:
            return None  # Возвращаем None, чтобы потом отфильтровать

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        new_target = {
            "boxes": boxes,
            "labels": labels
        }
        return img, new_target


train_dir = "train2017"
ann_file = "instances_train2017.json"

transform = Compose([Resize((256, 256)), ToTensor()])
train_dataset = CustomCocoDataset(root=train_dir, annFile=ann_file, transform=transform)

def collate_fn(batch):
    # Отфильтровываем пустые элементы (None)
    batch = [b for b in batch if b is not None]
    return tuple(zip(*batch)) if len(batch) > 0 else ([], [])


# Выбор подмножества данных
dataset_size = len(train_dataset)
subset_size = 5000  # Использовать только 5000 изображений
indices = random.sample(range(dataset_size), subset_size)

train_dataset = Subset(train_dataset, indices)

train_loader = DataLoader(
    train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn
)
