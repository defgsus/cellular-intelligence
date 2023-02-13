import gzip
import json
import os
from io import BytesIO
import argparse
from pathlib import Path
from multiprocessing.pool import Pool
from typing import Tuple, Iterable, List

import numpy as np
import pandas as pd
import scipy.stats
import scipy.signal
import PIL.Image
from tqdm import tqdm
import torch

from src.gen import *
from src.nn.contrastive import ContrastiveEncoder
from src.nn.trainer import CHECKPOINT_PATH
from src.util.image import np_to_pil


MODEL_NAME = "contra-ca-02"
#MODEL_NAME = "contra-ca-03-ch128"

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / MODEL_NAME


class Embedder:

    def __init__(self, latent_dim=64):
        self.latent_dim = latent_dim
        self.model = ContrastiveEncoder(
            #base_channel_size=128,
            latent_dim=self.latent_dim,
        )
        checkpoint = torch.load(CHECKPOINT_PATH / f"{MODEL_NAME}-snapshot.pt")
        self.model.load_state_dict(checkpoint["state_dict"])

    def get_embeddings(
            self,
            generators: List[Generator],
            count: int = 5,
            shape=(32, 32),
    ):
        embeddings = torch.zeros((len(generators), self.latent_dim))
        for idx in range(count):
            states = torch.zeros((len(generators), 1, shape[0], shape[1]))

            for i, gen in enumerate(generators):
                state = torch.Tensor(gen.generate(shape=shape))
                states[i, 0] = state

            with torch.no_grad():
                embeddings += self.model.forward(states)

        embeddings /= count
        return embeddings.numpy()


def run_feature_extraction(
        filename_part: str,
        rules: List[str],
        random_prob: float = .1,
        num_steps: int = 100,
        tqdm_pos: int = 0,
):
    embedder = Embedder()

    (DATA_PATH / f"{filename_part}-rules.json").write_text(json.dumps({
        "rules": rules,
        "latent_dim": embedder.latent_dim,
        "num_steps": num_steps,
    }))
    data_map = np.memmap(
        filename=str(DATA_PATH / f"{filename_part}.memmap"),
        dtype="float32",
        mode="w+",
        shape=(len(rules), embedder.latent_dim),
    )
    batch_size = 128

    for idx in tqdm(range(0, len(rules), batch_size), desc=filename_part, position=tqdm_pos):
        generators = [
            Generator([
                RandomDots(probability=random_prob),
                CARule(rule, count=num_steps, border="wrap"),
            ])
            for rule in rules[idx:idx + batch_size]
        ]
        embeddings = embedder.get_embeddings(generators)
        data_map[idx:idx + embeddings.shape[0]] = embeddings


def _run_feature_extraction_multiproc(args: Tuple[str, List[str], int]):
    run_feature_extraction(args[0], args[1], tqdm_pos=args[2])


def run_feature_extraction_multiproc(filename_part: str, num_processes: int = 10):
    all_rules = list(CARule.iter_automaton_rules())
    rules = [[] for i in range(num_processes)]
    for i, rule in enumerate(all_rules):
        rules[i % len(rules)].append(rule)

    process_args = [
        (f"{filename_part}-proc-{i:03}", r, i)
        for i, r in enumerate(rules)
    ]
    pool = Pool(len(rules))
    pool.map(_run_feature_extraction_multiproc, process_args)


def combine_multiproc_features(
        filename_part: str,
        dtype="float32",
        remove_zeros: bool = True,
):
    filename_part = DATA_PATH / filename_part

    all_data = []
    all_rules = []
    for filename in sorted(filename_part.parent.glob(f"{filename_part.name}-proc-*-rules.json")):
        rules_data = json.loads(Path(filename).read_text())
        filename = f"{str(filename)[:-11]}.memmap"
        data_map = np.memmap(
            filename=filename,
            dtype=dtype,
            mode="r",
            shape=(len(rules_data["rules"]), rules_data["latent_dim"]),
        )
        print(filename, data_map.shape)
        all_data.append(data_map)
        all_rules.extend(rules_data["rules"])

    all_data = np.concatenate(all_data, axis=0)
    all_rules = np.array(all_rules)
    print("combined", all_data.shape, all_rules.shape)

    if remove_zeros:
        zeros = all_data.sum(axis=1) == 0
        all_data = all_data[np.invert(zeros)]
        all_rules = all_rules[np.invert(zeros)]
        print("without zeros", all_data.shape)

    df = pd.DataFrame(all_data, index=all_rules)
    print(df)
    df.to_pickle(f"{filename_part}-df.pkl")


def main():
    global DATA_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--path", nargs="?", type=str, default=str(DATA_PATH),
        help=f"output path, defaults to {DATA_PATH}"
    )
    parser.add_argument(
        "-j", "--processes", nargs="?", type=int, default=4,
        help="number of parallel processes",
    )
    args = parser.parse_args()
    DATA_PATH = Path(args.path)
    os.makedirs(DATA_PATH, exist_ok=True)

    name = f"ca-embeddings-{MODEL_NAME}"
    #run_feature_extraction_multiproc(name, num_processes=args.processes)
    combine_multiproc_features(name)


if __name__ == "__main__":
    main()
