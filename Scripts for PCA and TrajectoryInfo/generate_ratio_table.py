import os
import json
import pandas as pd

json_dir = "/mnt/c/Users/MuhammadWaleed/Documents/Thesis/Implimentation/AVASAGcorpus-MLsubset91/AVASAGcorpus-MLsubset91/generated/signs/trimmed"

data_list = []

for filename in os.listdir(json_dir):
    if filename.endswith(".json"):
        file_path = os.path.join(json_dir, filename)

        with open(file_path, 'r') as f:
            data = json.load(f)

        dom_ratio = data['dominant_hand'].get('pc1_pc2_ratio', None)
        ndom_ratio = data['non_dominant_hand'].get('pc1_pc2_ratio', None)

        data_list.append([filename, dom_ratio, ndom_ratio])

df = pd.DataFrame(data_list, columns=["File Name", "Dom Hand PC1/PC2 Ratio", "Non-Dom Hand PC1/PC2 Ratio"])

df_sorted = df.sort_values(by="Dom Hand PC1/PC2 Ratio", ascending=False)

df_sorted.reset_index(drop=True, inplace=True)

output_csv_path = os.path.join(json_dir, "PC1_PC2_Ratios_Sorted.csv")
df_sorted.to_csv(output_csv_path, index=False)
