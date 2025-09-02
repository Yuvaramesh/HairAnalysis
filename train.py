# import os
# from ultralytics import YOLO

# # -----------------------------
# # Paths
# # -----------------------------
# BASE_DIR = os.path.dirname(__file__)
# DATA_YAML = os.path.join(BASE_DIR, "data.yaml")   # your dataset config file
# PRETRAINED = os.path.join(BASE_DIR, "yolov8n.pt") # YOLOv8 small model

# # -----------------------------
# # Train function
# # -----------------------------
# def train_model():
#     # Load pre-trained model
#     model = YOLO(PRETRAINED)

#     # Train on custom hair dataset
#     results = model.train(
#         data=DATA_YAML,      # points to your dataset config
#         epochs=25,           # increase if needed
#         imgsz=640,           # image size
#         batch=4,             # lower batch size since you have few images
#         name="hair_disease_model",
#         workers=0            # fix Windows dataloader issues
#     )

#     print("✅ Training complete. Best model is saved at:")
#     print("   runs/detect/hair_disease_model/weights/best.pt")

# if __name__ == "__main__":
#     train_model()
import os
import cv2
import albumentations as A
from glob import glob
import shutil

# ==========================
# CONFIG
# ==========================
INPUT_IMAGES = "DataImg"       # original images
INPUT_LABELS = "labels"       # YOLO txt labels
OUTPUT_IMAGES = "Augmented"    # where new images go
OUTPUT_LABELS = "Aulabels"    # where new labels go
AUG_PER_IMAGE = 20  # how many augmentations per image

# Create output dirs
os.makedirs(OUTPUT_IMAGES, exist_ok=True)
os.makedirs(OUTPUT_LABELS, exist_ok=True)

# ==========================
# Define augmentations
# ==========================
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=0.7),
    A.RandomBrightnessContrast(p=0.5),
    A.Blur(blur_limit=3, p=0.3),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# ==========================
# Augmentation Loop
# ==========================
image_paths = glob(os.path.join(INPUT_IMAGES, "*.jpg")) + glob(os.path.join(INPUT_IMAGES, "*.png"))

for img_path in image_paths:
    base = os.path.basename(img_path).split(".")[0]
    label_path = os.path.join(INPUT_LABELS, base + ".txt")

    if not os.path.exists(label_path):
        print(f"⚠️ No label for {img_path}, skipping...")
        continue

    # Load image + labels
    image = cv2.imread(img_path)
    h, w, _ = image.shape
    with open(label_path, "r") as f:
        boxes = []
        classes = []
        for line in f.readlines():
            cls, x, y, bw, bh = map(float, line.strip().split())
            boxes.append([x, y, bw, bh])
            classes.append(int(cls))

    # Generate augmentations
    for i in range(AUG_PER_IMAGE):
        try:
            aug = transform(image=image, bboxes=boxes, class_labels=classes)
            aug_img = aug["image"]
            aug_bboxes = aug["bboxes"]
            aug_classes = aug["class_labels"]

            # Save new image
            out_img_name = f"{base}_aug{i}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_IMAGES, out_img_name), aug_img)

            # Save new label
            out_label_name = f"{base}_aug{i}.txt"
            with open(os.path.join(OUTPUT_LABELS, out_label_name), "w") as f:
                for bbox, cls in zip(aug_bboxes, aug_classes):
                    x, y, bw, bh = bbox
                    f.write(f"{cls} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n")

        except Exception as e:
            print(f"❌ Error on {img_path} aug {i}: {e}")

print("✅ Augmentation complete! Check 'augmented/images' and 'augmented/labels'")
