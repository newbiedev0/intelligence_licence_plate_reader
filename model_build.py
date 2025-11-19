import yaml, os
yaml_path = "/content/indian_lp/data.yaml"
with open(yaml_path,'r') as f:
    cfg = yaml.safe_load(f)
print("Before fix:", cfg)
cfg["nc"] = 1
cfg["names"] = ["license_plate"]
with open(yaml_path,'w') as f:
    yaml.dump(cfg, f)
print("fixed data.yaml:")


import glob
DATA_ROOT="/content/indian_lp"
img_patterns = ["**/images/*.jpg", "**/images/*.png", "**/images/*.jpeg"]
img_files = []
for p in img_patterns:
    img_files += glob.glob(os.path.join(DATA_ROOT, p), recursive=True)
label_files = glob.glob(os.path.join(DATA_ROOT, "**", "labels", "*.txt"), recursive=True)
print("Images found:", len(img_files))
print("Label files found:", len(label_files))


img_basenames = set([os.path.splitext(os.path.basename(p))[0] for p in img_files])
label_basenames = set([os.path.splitext(os.path.basename(p))[0] for p in label_files])
missing_labels = img_basenames - label_basenames
missing_images = label_basenames - img_basenames
print("Images with NO label files (sample 10):", list(missing_labels)[:10])
print("Label files with NO image (sample 10):", list(missing_images)[:10])



from ultralytics import YOLO

model = YOLO("yolov8n.pt")

train_args = dict(
    data="/content/indian_lp_cleaned/indian_lp/data.yaml",
    epochs=120,    
    imgsz=640,
    batch=8,
    lr0=0.01,
    workers=4,
    patience=20,
    save=True,
)

print("Starting training with args:", train_args)
model.train(**train_args)


metrics = model.val(data="/content/sample_data/indian_lp/indian_lp/data.yaml", imgsz=640)

print("\n=== YOLO Evaluation Metrics ===")
print("Precision:      ", metrics.box.p)
print("Recall:         ", metrics.box.r)
print("mAP@50:         ", metrics.box.map50)
print("mAP@50-95:      ", metrics.box.map)
print("F1 Score:       ", metrics.box.f1)

