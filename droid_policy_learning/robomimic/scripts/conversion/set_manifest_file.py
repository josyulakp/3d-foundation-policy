import os
import json
import argparse
import random
import h5py
import robomimic
import args
count = 0
datasets = []
i = 0
parser = argparse.ArgumentParser(description="Process path and language parameters.")
parser.add_argument('--path', type=str, required=True, help="Path to the dataset directory.")
parser.add_argument('--lang', type=str, default=None, help="Language (optional).")
args = parser.parse_args()

path = args.path  # Path from arguments
language = args.lang  # Language from arguments (optional)
for root, dirs, files in os.walk(path):
    if 'failure' in root:
        continue
    for file in files:
        if file == 'trajectory_pcd_new_128_128.h5':
            i += 1              
            if os.path.exists(os.path.join(root, 'lang.txt')):
                with open(os.path.join(root, 'lang.txt'), 'r') as f:
                    lang = f.read()
                datasets.append({'path':os.path.join(root, file), 'lang':lang})
            else:
                language = ""
            
            if lang:
                language = lang
            
            datasets.append({'path':os.path.join(root, file), 'lang':language})

    print('Total Number: ', i)

random.shuffle(datasets)
for i in range(5):
    datasets[i]['train'] = False

with open('./droid_policy_learning/dataset.json', 'w') as out_f:
    out_f.write(json.dumps(datasets, indent=4))