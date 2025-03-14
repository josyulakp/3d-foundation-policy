"""
Add encoded language information to existing droid hdf5 file
"""
import h5py
import os
import json
import random
import torch
import open_clip
from tqdm import tqdm
import argparse

# Initialize model and tokenizer
model, _, preprocess = open_clip.create_model_and_transforms('EVA02-E-14-plus', pretrained='laion2b_s9b_b144k') 
model.eval()
tokenizer = open_clip.get_tokenizer('EVA02-E-14-plus')
model.to('cuda')

import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils

def add_lang(path, raw_lang, args):
    f = h5py.File(path, "a")  # Open HDF5 file in append mode
    
    # Ensure 'lang_fixed' group exists
    if "lang_fixed" not in f["observation"]:
        f["observation"].create_group("lang_fixed")
    
    # Remove existing 'language_distilbert'
    if "language_distilbert" in f["observation/lang_fixed"]:
        del f["observation/lang_fixed/language_distilbert"]
    
    lang_grp = f["observation/lang_fixed"]

    # Get the number of robot states
    H = f["observation/robot_state/cartesian_position"].shape[0]  
    encoded_lang = tokenizer(raw_lang).to('cuda')
    encoded_lang, _ = model.encode_text(encoded_lang)
    encoded_lang = encoded_lang.repeat(H, 1)
    
    # Save language data
    if "language_raw" not in f["observation/lang_fixed"]:
        lang_grp.create_dataset("language_raw", data=[raw_lang]*H)
        lang_grp.create_dataset("language_distilbert", data=encoded_lang.cpu().detach().numpy())
    else:
        lang_grp.create_dataset("language_distilbert", data=encoded_lang.cpu().detach().numpy())  
    
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_file", type=str, help="Manifest file path")
    args = parser.parse_args()

    # Load datasets from manifest
    with open(args.manifest_file, 'r') as file:
        datasets = json.load(file)

    print("Adding language data...")
    success = []
    random.shuffle(datasets)
    for item in tqdm(datasets):
        d, l = item['path'], item['lang']
        d = os.path.expanduser(d)
        try:
            add_lang(d, l, args)
            success.append(item)
        except Exception as e:
            print(f"Failed to add language data to {d}: {e}")
