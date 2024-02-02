import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import csv
from pathlib import Path

def calculate_metrics(mask1, mask2):
    # print(mask1.shape)
    # print(mask2.shape)
    assert mask1.shape == mask2.shape, "Les masques doivent avoir les mêmes dimensions"
    
    # Convertir les masques en booléen au cas où ils ne le sont pas
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    
    # Calculer l'intersection et l'union
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    
    # Calculer le IoU
    iou = np.sum(intersection) / np.sum(union)
    
    # Calculer la précision et le rappel
    true_positive = np.sum(intersection)
    false_positive = np.sum(np.logical_and(np.logical_not(mask1), mask2))
    false_negative = np.sum(np.logical_and(mask1, np.logical_not(mask2)))
    
    # Vérifier si le dénominateur pour la précision est zéro
    if true_positive + false_positive > 0:
        precision = true_positive / (true_positive + false_positive)
    else:
        precision = 0  # Ou np.nan, selon ce qui convient à votre analyse

    # Vérifier si le dénominateur pour le rappel est zéro
    if true_positive + false_negative > 0:
        recall = true_positive / (true_positive + false_negative)
    else:
        recall = 0
    
    # Calculer le coefficient de Dice
    if true_positive + false_positive + false_negative > 0 :
        dice_coefficient = 2 * true_positive / (2 * true_positive + false_positive + false_negative)
    else:
        dice_coefficient = 0
    
    return {"iou": iou, "precision": precision, "recall": recall, "dice_coefficient": dice_coefficient}

path = Path("yolov8x-seg/combined_kaggletest_yamls1")
base_path = Path("C:\\Users\\Haroun\\Desktop\\YOLO_CROSS_VALIDATION\\combined_kaggletest_folded1") 

for i in range(1, 6):
    fold_path = path / f'fold_{i}/train/weights/best.pt'
    model = YOLO(str(fold_path))
    source_path = base_path / f'fold_{i}/test/images'
    elements = model.predict(source=source_path, split="test", device=0, save=True, project=f"predict/yolov8x-seg/combined_kaggle/fold_{i}")

    thresholds = np.arange(0.1, 1.0, 0.1) # Seuils
    metrics_sum = {th: {"iou": 0, "precision": 0, "recall": 0, "dice_coefficient": 0} for th in thresholds}
    element_counts = {th: 170 for th in thresholds} # Il faut le changer avec le bon

    for element in elements:
        head, tail = os.path.split(element.path)
        new_head = head.replace('images', 'masks')
        file_name, ext = os.path.splitext(tail)
        new_tail = f'{file_name}.png'
        new_path = os.path.join(new_head, new_tail)
        grd_truth = cv2.imread(new_path, 0)

        if element.masks is not None:
            for threshold in thresholds:
                mask = element.masks.data.cpu().numpy()
                combined_mask = np.max(mask, axis=0) 
                thresholded_mask = (combined_mask > threshold).astype(int) 
                grd_truth_resized = cv2.resize(grd_truth, (thresholded_mask.shape[1], thresholded_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                results = calculate_metrics(thresholded_mask, grd_truth_resized)
                        # Affichage du masque seuillé
                # plt.imshow(thresholded_mask, cmap='gray')
                # plt.title(f'Masque seuillé à {threshold}')
                # plt.show()

                for metric in metrics_sum[threshold]:
                    metrics_sum[threshold][metric] += results[metric]
                # element_counts[threshold] += 1

    # Calculer la moyenne des métriques pour chaque seuil et écrire dans un fichier CSV
    with open(f'result_yolov8x_combined_kaggle_fold{i}.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        for threshold in thresholds:
            if element_counts[threshold] > 0:
                mean_metrics = {metric: metrics_sum[threshold][metric] / element_counts[threshold] for metric in metrics_sum[threshold]}
                row = [threshold] + [mean_metrics[metric] for metric in mean_metrics]
                writer.writerow(row)