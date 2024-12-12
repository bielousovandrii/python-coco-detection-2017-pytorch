import torch
from PIL import Image, ImageDraw
from torchvision.transforms import ToTensor
from detect import get_model
import os
import json
import matplotlib.pyplot as plt

model_folder = "."
model_files = [f for f in os.listdir(model_folder) if f.endswith('.pth')]

test_image_path = "test.jpg"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Ground truth box для розрахунку IoU
ground_truth_box = [634.493408203125, 212.12039184570312, 640.0, 331.7173767089844]
ground_truth_label = 1

# Загрузка тестового изображения
image = Image.open(test_image_path)
image_tensor = ToTensor()(image).unsqueeze(0).to(device)

losses = []  # Сюди додамо значення лосс-функції для кожної епохи
metrics = {}

# Функція для розрахунку IoU
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    iou_result = intersection / union if union > 0 else 0
    return iou_result

# Функція для обчислення precision і recall
def compute_precision_recall(predictions, ground_truth):
    tp = sum(1 for p in predictions if p['label'] == ground_truth_label and p['iou'] >= 0.5)
    fp = sum(1 for p in predictions if p['label'] != ground_truth_label or p['iou'] < 0.5)
    fn = 1 if all(p['label'] != ground_truth_label or p['iou'] < 0.5 for p in predictions) else 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return precision, recall

for model_file in model_files:
    model = get_model(num_classes=91)
    model_path = os.path.join(model_folder, model_file)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(image_tensor)

    predictions = outputs[0]
    filtered_boxes = predictions['boxes']
    filtered_labels = predictions['labels']
    filtered_scores = predictions['scores']

    model_predictions = []
    for box, label, score in zip(filtered_boxes, filtered_labels, filtered_scores):
        box_list = box.tolist()
        iou = compute_iou(box_list, ground_truth_box)
        model_predictions.append({
            "box": box_list,
            "label": label.item(),
            "score": score.item(),
            "iou": iou
        })

    precision, recall = compute_precision_recall(model_predictions, ground_truth_box)
    metrics[model_file] = {
        "precision": precision,
        "recall": recall
    }

# Побудова графіків
precision_values = [m['precision'] for m in metrics.values()]
recall_values = [m['recall'] for m in metrics.values()]
models = list(metrics.keys())

plt.figure(figsize=(10, 5))
plt.plot(models, precision_values, label='Precision')
plt.plot(models, recall_values, label='Recall')
plt.xlabel('Models')
plt.ylabel('Metrics')
plt.title('Precision and Recall for Different Models')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("metrics_graph.png")
plt.show()

# Візуалізація рамок
image_draw = Image.open(test_image_path).convert("RGB")
draw = ImageDraw.Draw(image_draw)

for pred in model_predictions:
    box = pred['box']
    draw.rectangle(box, outline="red", width=3)
    draw.text((box[0], box[1]), f"{pred['label']} {pred['score']:.2f}", fill="red")

image_draw.save("predicted_image.jpg")
