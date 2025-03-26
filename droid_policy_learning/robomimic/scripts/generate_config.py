import os
import json

import robomimic
from robomimic.config import get_all_registered_configs
import argparse
import time

parser = argparse.ArgumentParser(description="Process dataset, algorithm name and experiment name parameters.")
parser.add_argument('--dataset', type=str, required=True, help="Path to the dataset manifest file.")
parser.add_argument('--algo', type=str, default='mdt', help="Algorithm name")
parser.add_argument('--exp', type=str, default=time.strftime("%Y%m%d_%H%M%S"), help="Experiment name")
args = parser.parse_args()
    

def main():
    # store template config jsons in this directory
    target_dir = os.path.join(robomimic.__path__[0], "logs/config/")
    dataset = json.load(open(args.dataset, 'r'))
    # iterate through registered algorithm config classes
    config = json.load(open(os.path.join(robomimic.__path__[0], "exps/templates/mdt.json"), 'r'))
    config.train.data = dataset
    config.experiment.name = args.exp
    config.train.output_dir = os.path.join(robomimic.__path__[0], "logs", f"{args.algo}_{args.exp}")
    json_path = os.path.join(target_dir, "{}_{}.json".format(args.algo, args.exp))
    config.dump(filename=json_path)


if __name__ == '__main__':
    main()
