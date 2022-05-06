import argparse
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--proteinnet", type=str, 
                        help="path to proteinnet dataset to convert to ca distograms")
    parser.add_argument("-d", "--disto", type=str, 
                        help="path to hdf5 distograms")
    parser.add_argument("-f", "--fragment", type=int, choices=[16, 64, 128],
                        help="non-overlapping fragment length",)
    parser.add_argument("-o", "--out", type=str, 
                        help="path to output fragments")

    return parser.parse_args()


def get_record_data(record: str) -> Tuple[str, np.ndarray]:

    """convert sample record to distogram"""

    name = record[0].strip()
    seqlen = len(record[2].strip())
    coords = "".join(record[26:29])

    # N, CA, C
    # start at CA then step 3
    # (n_ca, 3), in angstrom
    coords = np.fromstring(coords, sep="\t").reshape(3, -1)[:, 1::3].T / 100
    disto = np.round(pairwise_distances(coords), 1)

    assert seqlen == len(disto), f"invalid distogram {name}"

    return name, disto


def parse_proteinnet(fn: str, fn_out: str) -> None:

    """get alpha carbon distogram from proteinnet"""

    name = Path(fn).stem
    fo = h5py.File(fn_out, "w")

    with open(fn, "r") as f:
        for line in f:
            if "[ID]" in line:
                # each chunk has fixed number of lines
                record = [f.readline() for i in range(31)]
                name, disto = get_record_data(record)
                print(name)
                fo.create_dataset(f"{name}", data=disto, compression="gzip")

    fo.close()


def disto_to_frags(disto: np.ndarray, frag_len: int) -> np.ndarray:

    """split distogram into arrays of non-overlapping fragments"""

    # skip tail remainder
    n_frags = int(np.floor(disto.shape[0] / frag_len))
    save = []

    # fragments from diagonal
    for i in range(0, n_frags):
        start = i * frag_len
        end = i * frag_len + frag_len
        arr = disto[start:end, start:end]
        save.append(arr)

    # (n_frags, frag_len, frag_len)
    return np.stack(save, axis=0)


def get_frags(disto_path: str, frag_len: int) -> np.ndarray:

    # n_cpus = mp.cpu_count() - 2 if mp.cpu_count() >= 4 else 2
    # pool = mp.Pool(n_cpus)

    f = h5py.File(disto_path, "r")
    keys = list(f.keys())

    # results = []
    # partial_func = partial(disto_to_frags, frag_len=frag_len)
    # jobs = [pool.apply_async(func=partial_func, args=(f[key][...],)) for key in keys]
    # pool.close()
    # for job in tqdm(jobs):
    #     results.append(job.get())

    results = [
        disto_to_frags(f[key][...], frag_len)
        for key in tqdm(keys)
        if len(f[key][...]) > frag_len
    ]
    f.close()

    return np.concatenate(results, axis=0)


if __name__ == "__main__":

    # 1. proteinnet records to distograms
    # >>> python preprocess.py -p <path_to_proteinnet> -d <path_to_output_distograms>
    # >>> python preprocess.py -p ../data/training_30 -d ../data/training_30_disto.hdf5

    # 2. distograms to fragments
    # >>> python preprocess.py -d <path_to_distograms> -f <fragment_size> -o <path_to_output_fragments>

    args = parse_args()

    if args.proteinnet and args.disto:
        parse_proteinnet(args.proteinnet, args.disto)

    # if (args.disto and args.fragment is None) or (args.fragment and args.disto is None):
    #    raise Exception('Must select fragment length with --fragment and disto path with --disto')

    if args.disto and args.fragment and args.out:
        results = get_frags(args.disto, args.fragment)
        with h5py.File(args.out, "w") as f:
            f.create_dataset("arr", data=results, compression="gzip")
