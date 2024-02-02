import pandas as pd
import matplotlib.pyplot as plt

def plot_iou_vs_threshold_no_header(csv_path):
    """ Trace le graphique IoU vs Threshold à partir d'un fichier CSV sans en-tête. """
    df = pd.read_csv(csv_path, header=None, names=['threshold', 'iou', 'precision', 'recall', 'dice'])
    df = df.sort_values(by='threshold')
    plt.figure(figsize=(10, 6))
    plt.plot(df['threshold'], df['iou'], marker='o')
    plt.title('IoU en fonction du Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('IoU')
    plt.grid(True)
    plt.savefig('wnds_iou_yolov8x_combined_kaggle_fold5.png')
    plt.close()

def plot_roc_curve_no_header(csv_path):
    """ Trace la courbe ROC avec une légende dans un encadré à partir d'un fichier CSV sans en-tête. """
    df = pd.read_csv(csv_path, header=None, names=['threshold', 'iou', 'precision', 'recall', 'dice'])
    plt.figure(figsize=(10, 6))
    for i in df.index:
        plt.plot(df['recall'][i], df['precision'][i], marker='o', label=f'Threshold {df["threshold"][i]:.2f}')
    plt.title('ROC Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.legend(loc='best')
    plt.savefig('wnds_roc_curve.png')
    plt.close()


plot_iou_vs_threshold_no_header('result_yolov8x_combined_kaggle_fold5.csv')
#plot_roc_curve_no_header('result_roc_azh.csv')

