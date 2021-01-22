from pathlib import Path
from sklearn.metrics import pairwise_distances
from functools import partial
from tqdm.auto import tqdm
import multiprocessing as mp
import argparse
import numpy as np
import h5py

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--proteinnet', type=str,
                        help='path to proteinnet dataset to convert to ca distrograms')
    parser.add_argument('-d', '--distro', type=str,
                        help='path to hdf5 distrograms')
    parser.add_argument('-f', '--fragment', type=int, choices = [16, 64, 128],
                        help='non-overlapping fragment length')

    return parser.parse_args()


def get_record_data(record):

    '''convert sample record to distrogram'''

    name = record[0].strip()
    seqlen = len(record[2].strip())
    coords = ''.join(record[26:29])

    # N, CA, C
    # start at CA then step 3
    # (n_ca, 3)
    coords = np.fromstring(coords, sep='\t').reshape(3, -1)[:, 1::3].T / 100
    distro = np.round(pairwise_distances(coords), 1)

    assert seqlen == len(distro), f'invalid distrogram {name}'

    return name, distro


def parse_proteinnet(fn):

    '''get alpha carbon distrogram from proteinnet'''

    name = fn.split('/')[-1]
    fo = h5py.File(f'./data/{name}_distrograms.hdf5', 'w')

    with open(fn, 'r') as f:
        for line in f:
            if '[ID]' in line:
                record = [f.readline() for i in range(31)]
                name, distro = get_record_data(record)
                print(name)
                fo.create_dataset(f'{name}', data=distro, compression='gzip')

    fo.close()


def distro_to_frags(distro, frag_len):

    '''split distrogram into arrays of non-overlapping fragments'''

    # skip tail remainder
    n_frags = int(np.floor(distro.shape[0] / frag_len))
    save = []

    # fragments from diagonal
    for i in range(0, n_frags):
        start = i*frag_len
        end = i*frag_len + frag_len
        arr = distro[start:end, start:end]
        save.append(arr)

    # (n_frags, frag_len, frag_len)
    return np.stack(save, axis=0)


def get_frags(distro_path, frag_len):

    # n_cpus = mp.cpu_count() - 2 if mp.cpu_count() >= 4 else 2
    # pool = mp.Pool(n_cpus)

    f = h5py.File(distro_path, 'r')
    keys = list(f.keys())

    # results = []
    # partial_func = partial(distro_to_frags, frag_len=frag_len)
    # jobs = [pool.apply_async(func=partial_func, args=(f[key][...],)) for key in keys]
    # pool.close()
    # for job in tqdm(jobs):
    #     results.append(job.get())

    results = [distro_to_frags(f[key][...], frag_len) for key in tqdm(keys) if len(f[key][...]) > frag_len]

    f.close()

    return np.concatenate(results, axis=0)


if __name__ == '__main__':

    args = parse_args()

    if args.proteinnet:
         parse_proteinnet(args.proteinnet)

    if (args.distro and args.fragment is None) or (args.fragment and args.distro is None):
        raise Exception('Must select fragment length with --fragment and distro path with --distro')

    if args.distro and args.fragment:

        name = args.distro.split('/')[-1].split('_distrograms.')[0]
        save_name = f'./data/{name}_{args.fragment}.hdf5'

        results = get_frags(args.distro, args.fragment)

        with h5py.File(save_name, 'w') as f:
            f.create_dataset('arr', data=results, compression='gzip')



