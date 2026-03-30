import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

json_dir = '/mnt/c/Users/MuhammadWaleed/Documents/Thesis/Implimentation/AVASAGcorpus-MLsubset91/AVASAGcorpus-MLsubset91/generated/signs/trimmed'
output_folder = '/mnt/c/Users/MuhammadWaleed/Documents/Thesis/Implimentation/AVASAGcorpus-MLsubset91/AVASAGcorpus-MLsubset91/generated/signs/trimmed//Output_Data/Plots_Histogram_KDE_BarChart'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

pc1_pc2_ratios = []

def plot_pca_for_json(file_path, file_name):
    with open(file_path, 'r') as f:
        data = json.load(f)

    dom_trajectory = np.array(data['dominant_hand']['trajectory'])
    ndom_trajectory = np.array(data['non_dominant_hand']['trajectory'])

    dom_pca_components = np.array([
        data['dominant_hand']['components']['PC1'],
        data['dominant_hand']['components']['PC2'],
        data['dominant_hand']['components']['PC3']
    ])

    ndom_pca_components = np.array([
        data['non_dominant_hand']['components']['PC1'],
        data['non_dominant_hand']['components']['PC2'],
        data['non_dominant_hand']['components']['PC3']
    ])

    dom_explained_variance = data['dominant_hand']['explained_variance']
    ndom_explained_variance = data['non_dominant_hand']['explained_variance']

    fig = plt.figure(figsize=(14, 7))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(dom_trajectory[:, 0], dom_trajectory[:, 1], dom_trajectory[:, 2], 'o-', label="Dominant Hand Trajectory", color='blue')
    ax1.quiver(np.mean(dom_trajectory[:, 0]), np.mean(dom_trajectory[:, 1]), np.mean(dom_trajectory[:, 2]), 
               dom_pca_components[0, 0], dom_pca_components[0, 1], dom_pca_components[0, 2], 
               length=dom_explained_variance[0], color='red', label='Dom PC1')
    ax1.quiver(np.mean(dom_trajectory[:, 0]), np.mean(dom_trajectory[:, 1]), np.mean(dom_trajectory[:, 2]), 
               dom_pca_components[1, 0], dom_pca_components[1, 1], dom_pca_components[1, 2], 
               length=dom_explained_variance[1], color='green', label='Dom PC2')
    ax1.quiver(np.mean(dom_trajectory[:, 0]), np.mean(dom_trajectory[:, 1]), np.mean(dom_trajectory[:, 2]), 
               dom_pca_components[2, 0], dom_pca_components[2, 1], dom_pca_components[2, 2], 
               length=dom_explained_variance[2], color='purple', label='Dom PC3')
    ax1.set_title(f'Dominant Hand PCA - {data["gloss"]}.trajectoryInfo.json')
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis')
    ax1.set_zlabel('Z-axis')
    ax1.legend()

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(ndom_trajectory[:, 0], ndom_trajectory[:, 1], ndom_trajectory[:, 2], 'o-', label="Non-Dominant Hand Trajectory", color='orange')
    ax2.quiver(np.mean(ndom_trajectory[:, 0]), np.mean(ndom_trajectory[:, 1]), np.mean(ndom_trajectory[:, 2]), 
               ndom_pca_components[0, 0], ndom_pca_components[0, 1], ndom_pca_components[0, 2], 
               length=ndom_explained_variance[0], color='red', label='Non-Dom PC1')
    ax2.quiver(np.mean(ndom_trajectory[:, 0]), np.mean(ndom_trajectory[:, 1]), np.mean(ndom_trajectory[:, 2]), 
               ndom_pca_components[1, 0], ndom_pca_components[1, 1], ndom_pca_components[1, 2], 
               length=ndom_explained_variance[1], color='green', label='Non-Dom PC2')
    ax2.quiver(np.mean(ndom_trajectory[:, 0]), np.mean(ndom_trajectory[:, 1]), np.mean(ndom_trajectory[:, 2]), 
               ndom_pca_components[2, 0], ndom_pca_components[2, 1], ndom_pca_components[2, 2], 
               length=ndom_explained_variance[2], color='purple', label='Non-Dom PC3')
    ax2.set_title(f'Non-Dominant Hand PCA - {data["gloss"]}.trajectoryInfo.json')
    ax2.set_xlabel('X-axis')
    ax2.set_ylabel('Y-axis')
    ax2.set_zlabel('Z-axis')
    ax2.legend()

    plt.savefig(os.path.join(output_folder, f'{file_name}_PCA_plot.png'))
    plt.close()

for filename in os.listdir(json_dir):
    if filename.endswith('.json'):
        file_path = os.path.join(json_dir, filename)
        plot_pca_for_json(file_path, filename)

        with open(file_path, 'r') as f:
            data = json.load(f)
            pc1_pc2_ratios.append(data['dominant_hand']['pc1_pc2_ratio'])

plt.figure(figsize=(10, 6))
sns.histplot(pc1_pc2_ratios, kde=True, color='blue', bins=15)
plt.title('Histogram and KDE of Dominant Hand PC1/PC2 Ratios')
plt.xlabel('PC1/PC2 Ratio')
plt.ylabel('Frequency')
plt.savefig(os.path.join(output_folder, 'pc1_pc2_histogram_kde.png'))
plt.close()

plt.figure(figsize=(10, 6))
sns.barplot(x=list(range(len(pc1_pc2_ratios))), y=pc1_pc2_ratios, color='blue')
plt.title('Bar Chart of Dominant Hand PC1/PC2 Ratios')
plt.xlabel('File Index')
plt.ylabel('PC1/PC2 Ratio')
plt.savefig(os.path.join(output_folder, 'pc1_pc2_bar_chart.png'))
plt.close()

plt.figure(figsize=(10, 6))
sns.barplot(x=list(range(len(pc1_pc2_ratios))), y=pc1_pc2_ratios, color='blue')
plt.title('Bar Chart of Dominant Hand PC1/PC2 Ratios')
plt.xlabel('Index')
plt.ylabel('PC1/PC2 Ratio')
plt.xticks(ticks=range(0, len(pc1_pc2_ratios)+1, 50))
plt.savefig(os.path.join(output_folder, 'pc1_pc2_bar_chart.png'))
plt.close()

plt.figure(figsize=(10, 6))
sns.barplot(x=list(range(len(pc1_pc2_ratios))), y=pc1_pc2_ratios, color='blue')
plt.title('Bar Chart of Dominant Hand PC1/PC2 Ratios')
plt.xlabel('Index')
plt.ylabel('PC1/PC2 Ratio')
plt.xticks(ticks=range(0, len(pc1_pc2_ratios)+1, 50))
plt.grid(True, which='both', axis='both')
plt.gca().set_xticks(range(0, len(pc1_pc2_ratios)+1, 50))
plt.savefig(os.path.join(output_folder, 'pc1_pc2_bar_chart.png'))
plt.close()
