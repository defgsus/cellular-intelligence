import gzip
import json
from io import BytesIO
from pathlib import Path
from multiprocessing.pool import Pool
from typing import Tuple, Iterable, List

import numpy as np
import pandas as pd
import scipy.stats
import scipy.signal
import PIL.Image
from tqdm import tqdm

from src.gen import *
from src.util.image import np_to_pil


def iter_automaton_rules():
    for i in range(2**18):
        r1 = i & (2**9-1)
        r2 = (i >> 9) & (2**9-1)
        r1 = [b for b in range(9) if (r1 >> b) & 1]
        r2 = [b for b in range(9) if (r2 >> b) & 1]
        if r2:
            yield "-".join((
                "".join(str(r) for r in r1),
                "".join(str(r) for r in r1),
            ))


def get_spectrum(
        gen: Generator,
        shape: Tuple[int, int] = (64, 64),
        count: int = 300,
        nfft: int = 1024,
):
    result = dict()

    specs = []
    for idx in range(count):
        state = gen.generate(shape=shape)

        _, spec = scipy.signal.periodogram(state.flatten(), nfft=nfft)
        specs.append(spec[1:].reshape(1, -1))

    specs = np.concatenate([spec.reshape(1, -1) for spec in specs], axis=0)
    spec_mean = specs.mean(axis=0)

    return spec_mean


def run_feature_extraction(
        filename_part: str,
        rules: List[str],
        random_prob: float = .1,
        num_steps: int = 100,
        nfft: int = 1024,
        tqdm_pos: int = 0,
):
    Path(f"{filename_part}-rules.json").write_text(json.dumps({
        "rules": rules,
        "nfft": nfft,
        "num_steps": num_steps,
    }))
    data_map = np.memmap(
        filename=f"{filename_part}.memmap",
        dtype="float32",
        mode="w+",
        shape=(len(rules), nfft // 2),
    )
    for idx, rule in enumerate(tqdm(rules, desc=filename_part, position=tqdm_pos)):
        gen = Generator([
            RandomDots(probability=random_prob),
            CARule(rule, count=num_steps, border="wrap"),
        ])
        spectrum = get_spectrum(gen, nfft=nfft)
        data_map[idx] = spectrum


def _run_feature_extraction_multiproc(args: Tuple[str, List[str], int]):
    run_feature_extraction(args[0], args[1], tqdm_pos=args[2])


def run_feature_extraction_multiproc(filename_part: str, num_processes: int = 10):
    all_rules = list(iter_automaton_rules())
    rules = [[] for i in range(num_processes)]
    for i, rule in enumerate(all_rules):
        rules[i % len(rules)].append(rule)

    process_args = [
        (f"{filename_part}-proc-{i:03}", r, i)
        for i, r in enumerate(rules)
    ]
    pool = Pool(len(rules))
    pool.map(_run_feature_extraction_multiproc, process_args)


def combine_multiproc_features(filename_part: str, dtype="float32"):
    filename_part = Path(filename_part)

    all_data = []
    all_rules = []
    for filename in tqdm(sorted(filename_part.parent.glob(f"{filename_part.name}-proc-*-rules.json"))):
        rules_data = json.loads(Path(filename).read_text())
        filename = f"{str(filename)[:-11]}.memmap"
        data_map = np.memmap(
            filename=filename,
            dtype=dtype,
            mode="r",
            shape=(len(rules_data["rules"]), rules_data["nfft"] // 2),
        )
        print(filename, data_map.shape)
        all_data.append(data_map)
        all_rules.extend(rules_data["rules"])

    all_data = np.concatenate(all_data, axis=0)
    print("combined", all_data.shape, len(all_rules))


def main():
    name = "ca-specs"
    #run_feature_extraction_multiproc(name)
    combine_multiproc_features(name)


if __name__ == "__main__":
    main()
