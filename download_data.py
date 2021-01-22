import requests
from pathlib import Path
import logging
from tqdm.auto import tqdm

def download_pdb(pdb):

    save_path = Path(f'./pdbs/{pdb}.pdb')

    if save_path.exists():
        logging.info(f'skipping {pdb}')
        return

    logging.info(f'downloading {pdb}')
    url = f'https://files.rcsb.org/download/{pdb}.pdb'
    r = requests.get(url)
    r.raise_for_status()
    save_path.write_text(r.text)


def download_pdbs(fn):

    f = Path(fn).read_text().splitlines()
    for line in tqdm(f):
        pdb = line.strip()
        download_pdb(pdb)


if __name__ == '__main__':

    #logging.basicConfig(filename='./logs/download.log', level=logging.INFO)
    #logging.getLogger().addHandler(logging.StreamHandler())

    # skip, using proteinnet instead of full rcsb split from paper
    #download_pdbs('data/train_ids.txt')
    #download_pdbs('data/test_ids.txt')
    pass
