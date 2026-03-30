import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

json_dir = '/mnt/c/Users/MuhammadWaleed/Documents/Thesis/Implimentation/AVASAGcorpus-MLsubset91/AVASAGcorpus-MLsubset91/generated/signs/trimmed'
output_dir = os.path.join(json_dir, 'Output_Data/Plots')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def generate_plot(file_name):
    with open(file_name, 'r') as f:
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

    output_plot_path = os.path.join(output_dir, f'{data["gloss"]}_PCA_plot.png')
    plt.savefig(output_plot_path)
    plt.close()

for json_file in os.listdir(json_dir):
    if json_file.endswith('.json'):
        file_path = os.path.join(json_dir, json_file)
        generate_plot(file_path)
