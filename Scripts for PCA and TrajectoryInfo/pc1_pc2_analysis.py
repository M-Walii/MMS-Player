import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

json_dir = '/mnt/c/Users/MuhammadWaleed/Documents/Thesis/Implimentation/AVASAGcorpus-MLsubset91/AVASAGcorpus-MLsubset91/generated/signs/trimmed'

pc1_pc2_ratios = []

for filename in os.listdir(json_dir):
    if filename.endswith('.json'):
        file_path = os.path.join(json_dir, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)
            if 'dominant_hand' in data and 'pc1_pc2_ratio' in data['dominant_hand']:
                pc1_pc2_ratio = data['dominant_hand']['pc1_pc2_ratio']
                pc1_pc2_ratios.append(pc1_pc2_ratio)

pc1_pc2_ratios = np.array(pc1_pc2_ratios)

plt.figure(figsize=(10, 6))
plt.hist(pc1_pc2_ratios, bins=20, color='blue', alpha=0.7)
plt.title('Histogram of PC1 to PC2 Ratio (Dominant Hand)')
plt.xlabel('PC1/PC2 Ratio')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig(f'Histogram.png')
plt.show()

plt.figure(figsize=(10, 6))
sns.kdeplot(pc1_pc2_ratios, color='blue', shade=True)
plt.title('KDE of PC1 to PC2 Ratio (Dominant Hand)')
plt.xlabel('PC1/PC2 Ratio')
plt.ylabel('Density')
plt.grid(True)
plt.savefig(f'KDE.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(range(len(pc1_pc2_ratios)), pc1_pc2_ratios, color='blue', alpha=0.7)
plt.title('Bar Chart of PC1 to PC2 Ratio (Dominant Hand)')
plt.xlabel('Index')
plt.ylabel('PC1/PC2 Ratio')
plt.grid(True)
plt.savefig(f'Bar Chart.png')
plt.show()
