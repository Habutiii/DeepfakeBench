import csv
import random
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

PROPORTION = 0.1

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

sample_size = min(int(PROPORTION * len(preds)), 1000)

fpr, tpr, _ = metrics.roc_curve(labels, preds)
original_auc = metrics.auc(fpr, tpr)
prediction_class = [1 if i > 0.5 else 0 for i in preds]
correct = sum(a == b for a, b in zip(prediction_class, labels))
original_acc = correct / len(prediction_class)
sampled_auc = []
sampled_acc = []

for _ in range(1000):
    sample_idx = random.sample(range(len(preds)), sample_size)
    sample_preds = [preds[i] for i in sample_idx]
    sample_labels = [labels[i] for i in sample_idx]

    fpr, tpr, _ = metrics.roc_curve(sample_labels, sample_preds)
    sampled_auc.append(metrics.auc(fpr, tpr))

    prediction_class = [1 if i > 0.5 else 0 for i in sample_preds]
    correct = sum(a == b for a, b in zip(prediction_class, sample_labels))
    sampled_acc.append(correct / len(prediction_class))

a = np.array(sampled_auc)
percentile_10 = np.percentile(a, 10)
percentile_90 = np.percentile(a, 90)
std = np.std(a)
plt.hist(sampled_auc, bins=20)
plt.axvline(original_auc, color='navy', linestyle='--', label='AUC for full sample')
plt.axvline(percentile_10, color='red', linestyle='--', label='10th percentile')
plt.axvline(percentile_90, color='brown', linestyle='--', label='90th percentile')
plt.xlabel('Sampled AUC')
plt.ylabel('Number')
plt.title(f'AUCs Using {sample_size} Samples of {len(preds)} Results, std={std:.4f}')
plt.legend()
plt.show()
plt.clf()

a = np.array(sampled_acc)
percentile_10 = np.percentile(a, 10)
percentile_90 = np.percentile(a, 90)
std = np.std(a)
plt.hist(sampled_acc, bins=20)
plt.axvline(original_acc, color='navy', linestyle='--', label='ACC for full sample')
plt.axvline(percentile_10, color='red', linestyle='--', label='10th percentile')
plt.axvline(percentile_90, color='brown', linestyle='--', label='90th percentile')
plt.xlabel('Sampled ACC')
plt.ylabel('Number')
plt.title(f'ACC Using {sample_size} Samples of {len(preds)} Results, std={std:.4f}')
plt.legend()
plt.show()

#plot_roc_curve(fpr, tpr, roc_auc)
