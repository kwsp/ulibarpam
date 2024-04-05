from pathlib import Path

from tqdm import tqdm
from ulibarpam import IOParams, ReconParams
from ulibarpam import get_num_scans, load_scans
from ulibarpam import recon_one_scan
from ulibarpam import write_images
import numpy as np

import click


ioparams = IOParams.default()
params = ReconParams.default()


def recon_bin(
    ioparams: IOParams,
    recon_params: ReconParams,
    fname,
    start_i=0,
    n_scans=0,
    savedir=Path("images"),
):
    savedir.mkdir(exist_ok=True)

    num_scans_all = get_num_scans(fname)
    if n_scans < 1:
        n_scans = num_scans_all - start_i
    assert start_i + n_scans <= num_scans_all

    for i in tqdm(range(start_i, start_i + n_scans)):
        print(i)
        rf = load_scans(fname, i)
        rf = rf - np.mean(rf, axis=0)

        flip = bool(i % 2)
        rect, radial = recon_one_scan(rf, ioparams, recon_params, flip)
        write_images(savedir / f"rect_{i:03}.png", rect)
        write_images(savedir / f"radial_{i:03}.png", radial)


@click.command()
@click.argument("fname")
@click.option(
    "--nscans",
    default=-1,
    help="Number of Scans to recon. By default recon all scans available. A small number is useful for testing.",
)
def main(fname, nscans: int):
    fname = Path(fname)
    savedir = fname.parent / fname.stem
    savedir.mkdir(exist_ok=True)

    recon_bin(ioparams, params, fname, n_scans=nscans, savedir=savedir)


if __name__ == "__main__":
    main()
