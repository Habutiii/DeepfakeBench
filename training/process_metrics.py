import csv
import sys
import matplotlib.pyplot as plt
from sklearn import metrics

if len(sys.argv) == 1:
    print("Missing path to csv file!")
    sys.exit()

file_path = sys.argv[1]

preds = []
labels = []

def plot_roc_curve(fpr, tpr, roc_auc):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

with open(file_path, 'r') as file:
    reader = csv.reader(file)
    header = next(reader)

    for row in reader:
        preds.append(float(row[1]))
        labels.append(int(row[2]))

fpr, tpr, thresholds = metrics.roc_curve(labels, preds)
roc_auc = metrics.auc(fpr, tpr)
plot_roc_curve(fpr, tpr, thresholds)
