import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import os
import json

def evaluate_model(model, test_ds, class_names, output_dir, prefix="model"):
    """
    Evaluasi model dan simpan confusion matrix serta classification report.
    """
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images)
        y_true.extend(np.argmax(labels, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    # Classification Report
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_str = classification_report(y_true, y_pred, target_names=class_names)

    print(report_str)
    with open(os.path.join(output_dir, f'{prefix}_report.txt'), 'w') as f:
        f.write(report_str)

    # Simpan report_dict ke JSON untuk pembandingan nanti
    with open(os.path.join(output_dir, f'{prefix}_metrics.json'), 'w') as f:
        json.dump(report_dict, f)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {prefix}')
    plt.savefig(os.path.join(output_dir, f'{prefix}_confusion_matrix.png'))
    plt.close()
