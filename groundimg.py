from PIL import Image

# Відкрити зображення
image_path = "test22.jpg"
image = Image.open(image_path)
image.show()

# Вивести розмір зображення
print(f"Image size: {image.size} (width, height)")

# Інструкція:
# Відкривши зображення, використовуйте редактор для перегляду
# позицій потрібних пікселів. Наприклад:
xmin, ymin = 635, 188  # Верхній лівий кут
xmax, ymax = 640, 290  # Нижній правий кут
ground_truth_box = [xmin, ymin, xmax, ymax]
print(f"Ground Truth Box: {ground_truth_box}")
