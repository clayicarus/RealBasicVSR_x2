import cv2
import os

IN_DIR = "data/REDS/train_sharp"
OUT_DIR = "data/REDS/train_sharp_bicubic/X2"
SCALE = 2

assert os.path.exists(IN_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

for root, dirs, files in os.walk(IN_DIR):
    new_dir = root[len(IN_DIR): ].strip('/')
    if new_dir != "":
        os.makedirs(os.path.join(OUT_DIR, new_dir))
    for file in files:
        img = cv2.imread(os.path.join(root, file))
        h, w, _ = img.shape
        new_img = cv2.resize(img, (w // SCALE, h // SCALE))
        new_path = os.path.join(OUT_DIR, new_dir, file)
        if cv2.imwrite(new_path, new_img):
            print("write to", new_path)
        else:
            print("failed to write", new_path)
