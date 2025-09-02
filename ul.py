# from ultralytics import YOLO
# import os
# import shutil

# # Load YOLO model (change to your trained model if available)
# model = YOLO("yolov8n.pt")   

# # Input and output directories
# image_dir = "DataImg"       # Folder containing raw images
# label_dir = "labels"        # Final folder where YOLO labels will be saved

# # Ensure label folder exists
# os.makedirs(label_dir, exist_ok=True)

# # Run detection on all images and save results (including .txt labels)
# results = model.predict(
#     source=image_dir,       # Folder of images
#     save_txt=True,          # Save YOLO-format labels
#     save_conf=True,         # Save confidence scores in labels
#     project="run/labels",  # YOLO output folder
#                  # Experiment name
#     exist_ok=True           # Avoid overwriting issues
# )

# # YOLO saves labels in: runs/labels/exp/labels
# output_dir = os.path.join("run","labels")

# # Move YOLO-generated labels into your label_dir
# if os.path.exists(output_dir):
#     for file in os.listdir(output_dir):
#         src = os.path.join(output_dir, file)
#         dst = os.path.join(label_dir, file)
#         shutil.move(src, dst)

# print(f"✅ Labels generated and saved inside: {label_dir}")
from ultralytics import YOLO

# Load your custom trained model
model = YOLO("yolov8n.pt")

# Run inference
results = model.predict(source="DataImg", save=True, save_txt=True, conf=0.25)

print("✅ Done. Results saved in:", results)
