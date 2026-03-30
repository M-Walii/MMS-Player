import json
import numpy as np
from numpy.linalg import norm
import math

file_name = 'INDEX.trajectoryInfo.json'
with open(f'/mnt/c/Users/MuhammadWaleed/Documents/Thesis/Implimentation/AVASAGcorpus-MLsubset91/AVASAGcorpus-MLsubset91/generated/signs/trimmed/{file_name}', 'r') as f:
    data = json.load(f)

dom_trajectory = np.array(data['dominant_hand']['trajectory'])

D = dom_trajectory[-1] - dom_trajectory[0]

D_normalized = D / norm(D)

original_PC1 = np.array(data['dominant_hand']['components']['PC1'])
PC1_normalized = original_PC1 / norm(original_PC1)

dot_product = np.dot(D_normalized, PC1_normalized)

angle = math.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))

if angle > 90:
    adjusted_PC1 = -original_PC1
    flip_status = "Flipping PC1 is necessary."
else:
    adjusted_PC1 = original_PC1
    flip_status = "Flipping PC1 is not necessary."
