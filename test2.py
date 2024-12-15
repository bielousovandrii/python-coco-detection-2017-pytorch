import torch
from PIL import Image, ImageDraw, ImageFont
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
detected_boxes = []
detected_scores = []
for box, score in zip(boxes, scores):
    if score > threshold:
        detected_boxes.append(box.tolist())
        detected_scores.append(score.item())

if detected_boxes:
    print(f"Detected boxes: {detected_boxes}")
    print(f"Scores: {detected_scores}")
else:
    print("No object detected with sufficient confidence")

# Візуалізуємо результат
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()

for box, score in zip(detected_boxes, detected_scores):
    draw.rectangle(box, outline="red", width=3)
    # Додаємо score до рамки
    text = f"{score:.2f}"
    # Малюємо текст біля верхнього лівого кута рамки
    text_position = (box[0], box[1] - 10)
    draw.text(text_position, text, fill="red", font=font)

# Зберігаємо результат
image.save("output_detected_box_with_score.jpg")
image.show()
