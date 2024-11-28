import torch
from PIL import Image
from torchvision.transforms import ToTensor
from detect import get_model
import os
import json

model_folder = "."  # Папка, где хранятся файлы модели
model_files = [f for f in os.listdir(model_folder) if f.endswith('.pth')]

test_image_path = "giraff.jpg"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Загрузка тестового изображения
image = Image.open(test_image_path)
image_tensor = ToTensor()(image).unsqueeze(0).to(device)

best_result = None  # Для хранения лучшего результата
best_model = None   # Для хранения имени модели с лучшим результатом

for model_file in model_files:
    # Загружаем модель
    model = get_model(num_classes=91)
    model_path = os.path.join(model_folder, model_file)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Тестирование модели
    with torch.no_grad():
        outputs = model(image_tensor)

    # Обработка результатов
    predictions = outputs[0]
    filtered_boxes = predictions['boxes']
    filtered_labels = predictions['labels']
    filtered_scores = predictions['scores']

    # Поиск лучшего предсказания в данной модели
    for box, label, score in zip(filtered_boxes, filtered_labels, filtered_scores):
        if best_result is None or score > best_result['score']:
            best_result = {
                "model": model_file,
                "box": box.tolist(),
                "label": label.item(),
                "score": score.item()
            }
            best_model = model_file

# Вывод лучшего результата
if best_result:
    print(f"Best result from model: {best_model}")
    print(f"  Label: {best_result['label']}, Score: {best_result['score']:.2f}, Box: {best_result['box']}")

    # Сохранение в JSON
    with open("best_result.json", "w") as f:
        json.dump(best_result, f, indent=4)
else:
    print("No valid predictions found.")
