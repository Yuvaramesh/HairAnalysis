import cv2
import os
import albumentations as A

input_dir = "DataImg"
output_dir = "Augmented"
os.makedirs(output_dir, exist_ok=True)

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
    A.GaussNoise(p=0.3)
])

for img_name in os.listdir(input_dir):
    if img_name.endswith(".jpg") or img_name.endswith(".png"):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)

        for i in range(20):  # 20 augmented versions per image → 10 * 20 = 200
            augmented = transform(image=img)
            aug_img = augmented['image']
            cv2.imwrite(f"{output_dir}/{img_name.split('.')[0]}_aug{i}.jpg", aug_img)
