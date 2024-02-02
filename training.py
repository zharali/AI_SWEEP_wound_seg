# 'C:\Users\Usr\AppData\Roaming\Ultralytics\settings.yaml' for configs
# I changed C:\Users\Haroun\AppData\Local\Programs\Python\Python38\Lib\site-packages\ultralytics\utils\ops.py process_mask => masks.gt(0.5)

from ultralytics import YOLO
from pathlib import Path
import torch
import gc
import numpy as np

def calculate_metrics(mask1, mask2):
    assert mask1.shape == mask2.shape, "Les masques doivent avoir les mêmes dimensions"
    
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    
    iou = np.sum(intersection) / np.sum(union)
    
    true_positive = np.sum(intersection)
    false_positive = np.sum(np.logical_and(np.logical_not(mask1), mask2))
    false_negative = np.sum(np.logical_and(mask1, np.logical_not(mask2)))
    
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    
    dice_coefficient = 2 * true_positive / (2 * true_positive + false_positive + false_negative)
    
    return iou, precision, recall, dice_coefficient

def train_folds(base_path):
    print(base_path)
    
    #models = ["yolov8s-seg"]
    models = ["yolov8n-seg", "yolov8m-seg", "yolov8l-seg","yolov8x-seg"]
    # models = ["yolov8n-seg", "yolov8s-seg", "yolov8m-seg"]
    # models = ["yolov8l-seg","yolov8x-seg"]
    base_path = Path(base_path)

    #for subfolder in base_path.iterdir()
    subfolder = base_path
    if subfolder.is_dir():
        for m in models:
            for i in range(1, 6):  # Itérer de fold1 à fold5
                fold_path = subfolder / f'fold_{i}.yaml'
                if fold_path.exists():
                    model = YOLO(f'{m}.pt')      
                    model.train(
                        data=fold_path, #path to yaml
                        device="0",
                        batch=16,
                        epochs=100,
                        imgsz=640,
                        seed=18,
                        patience=20,
                        save_period=50,
                        workers=16,
                        pretrained=False,
                        cache=False,
                        project=f"{m}/{subfolder.name}/fold_{i}"
                        # ,scale=0.0,flipud=0.0,
                        # fliplr=0.0,mosaic=0.0,translate=0.0,hsv_h=0.0,hsv_s=0.0,hsv_v=0.0
                        )
                    
                    del model
                    torch.cuda.empty_cache()
                    gc.collect()

                    print(f"Completed training for {subfolder.name}/fold_{i}")
        print(f"Completed training for all models in {subfolder.name}")

if __name__ == "__main__":
    base_path = Path("C:\\Users\\Haroun\\Desktop\\YOLO_CROSS_VALIDATION\\combined_kaggletest_yamls1")  #Path(".\\yamls\\")
    train_folds(base_path)
