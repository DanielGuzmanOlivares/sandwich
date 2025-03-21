#!/usr/bin/env python3
"""
This scripts is used to benchmark the performance of the SANDWICH model using the most common WSD datasets

Author: Daniel Guzman <daniel.guzman@buliltec.com>
Date: 2025-03-19
Version: 1.0
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from sandwich import Sandwich
from sandwich.metrics import eval_f1


parser = argparse.ArgumentParser(description="Benchmark the SANDWICH model")
# add a flag arguments
parser.add_argument(
    "--42D",
    help="Use the 42D dataset for evaluation (see Maru et al. 2022 for more details)",
    action="store_true",
)
parser.add_argument(
    "--softEN",
    help="Use the softEN dataset for evaluation (see Maru et al. 2022 for more details)",
    action="store_true",
)
parser.add_argument(
    "--hardEN",
    help="Use the hardEN dataset for evaluation (see Maru et al. 2022 for more details)",
    action="store_true",
)
parser.add_argument(
    "--semeval2010",
    help="Use the semeval2010 dataset for evaluation (see Maru et al. 2022 for more details)",
    action="store_true",
)

parser.add_argument(
    "--semeval2007",
    help="Use the semeval 2007 dataset, usually used as dev set in Raganato et al. 2017. benchmark.",
    action="store_true",
)
parser.add_argument(
    "--coarse",
    help="Use the coarse dataset in Navigli et al 2007..",
    action="store_true",
)

parser.add_argument(
    "--raganato",
    help="Use the Raganato 2017 benchmark for evaluation",
    action="store_true",
)

# add a positional argument
parser.add_argument(
    "--pos",
    help="Use the n dataset for evaluation (see Maru et al 2022. for more details)",
    action="store_true",
)

# gpu param
parser.add_argument(
    "--gpu",
    help="Use the GPU for the evaluation",
    action="store_true",
)

# batch size
parser.add_argument(
    "--batch_size",
    help="Batch size for the evaluation",
    type=int,
    default=32,
)

parser.add_argument(
    "--all",
    help="Evaluate all datasets",
    action="store_true",
)

if __name__ == "__main__": 

    # parse the arguments
    args = parser.parse_args()

    # files
    file_dict = {
        "42D": "difficult.json",
        "softEN": "softEN.json",
        "hardEN" : "hardEN.json",
        "semeval2010": "s10.json",
        "raganato": "test.json",
        "semeval2007": "dev.json",
        "coarse": "coarse.json",
    }

    # Paths necessary to load resources
    benchmark_dir = Path(__file__).parent / "data" / "benchmarks"
    babelnet_dir = Path(__file__).parent / "data" / "babelnet"
    models_dir = Path(__file__).parent / "models"

    # load the model
    model = Sandwich(
        cross_encoder_nv_path=models_dir / "encoder-nv",
        cross_encoder_v_path=models_dir / "encoder-v",
        definitions_path=babelnet_dir / "definitions.json",
        neighbours_path=babelnet_dir / "neighbours.json",
        device="cuda" if args.gpu else "cpu",
    )

    for key, file in file_dict.items():
        if getattr(args, key) or args.all:
            with open(benchmark_dir / file) as f:
                data = json.load(f)
            pred_dict = {}
            gold_dict = {}
            for k in tqdm(data, total=len(data), desc=f"Disambiguating {key} dataset ..."):
                gold_dict[k] = data[k]["gold"]
                sentence = data[k]['sentence']
                synsets = data[k]['synsets']
                word = data[k]['tokens']
                # add to dict, needs to be a list for the eval_f1 function
                pred_dict[k] = [model.disambiguate(sentence, word, synsets, args.batch_size)] 
            with open(Path(__file__).parent / "results" / file, "w") as f:
                json.dump(pred_dict, f, indent=4)
            results = eval_f1(pred_dict, gold_dict)
            if not args.pos:
                results = results.T.drop(columns=list(set(["n", "v", "a", "r"]) & set(results.columns))).T
            print(results)

