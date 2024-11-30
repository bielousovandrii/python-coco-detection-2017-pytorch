import torch
from PIL import Image
from torchvision.transforms import ToTensor
from detect import get_model
import os
import json

model_folder = "."  # Папка, где хранятся файлы модели
model_files = [f for f in os.listdir(model_folder) if f.endswith('.pth')]

test_image_path = "test.jpg"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Ground truth box для розрахунку IoU (замініть на реальні координати)
ground_truth_box = [634.493408203125, 212.12039184570312, 640.0, 331.7173767089844]
ground_truth_label = 1  # Реальний клас об'єкта

# Загрузка тестового изображения
image = Image.open(test_image_path)
image_tensor = ToTensor()(image).unsqueeze(0).to(device)

best_result = None
best_model = None
metrics = {}

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
    model.load_state_dict(torch.load(model_path, map_location=device,weights_only=True))
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

    for pred in model_predictions:
        if best_result is None or pred['score'] > best_result['score']:
            best_result = {
                "model": model_file,
                "box": pred['box'],
                "label": pred['label'],
                "score": pred['score'],
                "iou": pred['iou']
            }
            best_model = model_file

if best_result:
    mAP = sum(m['precision'] for m in metrics.values()) / len(metrics)
    best_result['precision'] = metrics[best_model]['precision']
    best_result['recall'] = metrics[best_model]['recall']
    best_result['mAP'] = mAP

    print(f"Best result from model: {best_model}")
    print(f"  Label: {best_result['label']}, Score: {best_result['score']:.2f}, Box: {best_result['box']}, IoU: {best_result['iou']:.2f}")
    print(f"  Precision: {best_result['precision']:.2f}, Recall: {best_result['recall']:.2f}, mAP: {best_result['mAP']:.2f}")

    with open("best_result.json", "w") as f:
        json.dump(best_result, f, indent=4)
else:
    print("No valid predictions found.")
