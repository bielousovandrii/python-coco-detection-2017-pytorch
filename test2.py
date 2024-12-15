import torch
from PIL import Image, ImageDraw
from torchvision import models, transforms
import os

# Шлях до зображення
test_image_path = "giraff.jpg"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Завантажуємо модель Faster R-CNN з попередньо натренованими вагами
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()

# Завантажуємо зображення
image = Image.open(test_image_path).convert("RGB")
image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)

# Прогнозування об'єктів на зображенні
with torch.no_grad():
    predictions = model(image_tensor)

# Отримуємо детектовані коробки, мітки та впевненості
boxes = predictions[0]['boxes']
labels = predictions[0]['labels']
scores = predictions[0]['scores']

# Визначення ground truth box автоматично за допомогою найбільш впевненого результату
# (можна додати фільтрацію за значенням confidence score)
threshold = 0.5
detected_box = None
for box, score in zip(boxes, scores):
    if score > threshold:
        detected_box = box.tolist()
        break  # Якщо знайдено перший об'єкт із достатньою впевненістю

if detected_box:
    print(f"Detected ground truth box: {detected_box}")
else:
    print("No object detected with sufficient confidence")

# Візуалізуємо результат
draw = ImageDraw.Draw(image)
if detected_box:
    draw.rectangle(detected_box, outline="red", width=3)

# Зберігаємо результат
image.save("output_detected_box.jpg")
image.show()
