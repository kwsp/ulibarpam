from pathlib import Path
import json

from tqdm import tqdm
from ulibarpam import IOParams, ReconParams
from ulibarpam import get_num_scans, load_scans, estimate_aline_background
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

    num_scans_all = get_num_scans(fname) - 1
    if num_scans_all < 1:
        print(f"No scans available in {fname}")
        return

    if n_scans < 1:
        n_scans = num_scans_all - start_i
    assert start_i + n_scans <= num_scans_all

    all_meta = {}
    print("Estimating A-line background")
    rf_background = estimate_aline_background(fname, 50_000)
    for i in tqdm(range(start_i, start_i + n_scans)):
        rf = load_scans(fname, i)
        rf = rf - rf_background

        flip = bool(i % 2)
        rect, radial, meta = recon_one_scan(rf, ioparams, recon_params, flip)
        write_images(savedir / f"rect_{i:03}.png", rect)
        write_images(savedir / f"radial_{i:03}.png", radial)

        all_meta[i] = meta

    with open(savedir / "meta.json", "w") as fp:
        json.dump(all_meta, fp)


def recon_bin_multiproc(
    ioparams: IOParams,
    recon_params: ReconParams,
    fname,
    start_i=0,
    n_scans=0,
    savedir=Path("images"),
    num_workers=4,
):
    savedir.mkdir(exist_ok=True)

    num_scans_all = get_num_scans(fname) - 1
    if num_scans_all < 1:
        print(f"No scans available in {fname}")
        return

    if n_scans < 1:
        n_scans = num_scans_all - start_i
    assert start_i + n_scans <= num_scans_all

    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(
        num_workers,
        initializer=ReconWorker.init_worker,
        initargs=(fname, savedir, ioparams, recon_params),
    ) as executor:

        futures = executor.map(
            ReconWorker.process_task, range(start_i, start_i + n_scans)
        )

        all_meta = {}
        for i, meta in tqdm(futures, total=n_scans):
            all_meta[i] = meta

        import json

        with open(savedir / "meta.json", "w") as fp:
            json.dump(all_meta, fp)

    # worker = ReconWorker(fname, savedir, ioparams, recon_params)
    # for i in tqdm(range(start_i, start_i + n_scans)):
    # worker.recon(i)


class ReconWorker:
    def __init__(self, fname, savedir, ioparams: IOParams, recon_params: ReconParams):
        self.fname = fname
        self.savedir = savedir
        self.ioparams = ioparams
        self.recon_params = recon_params

        self.rf_background = estimate_aline_background(fname, 50_000)

    def recon(self, i):
        rf = load_scans(self.fname, i)
        rf = rf - self.rf_background

        flip = bool(i % 2)
        rect, radial, meta = recon_one_scan(rf, ioparams, self.recon_params, flip)
        write_images(self.savedir / f"rect_{i:03}.png", rect)
        write_images(self.savedir / f"radial_{i:03}.png", radial)
        return i, meta

    @staticmethod
    def init_worker(fname, savedir, ioparams, recon_params):
        global worker
        worker = ReconWorker(fname, savedir, ioparams, recon_params)

    @staticmethod
    def process_task(i):
        global worker
        return worker.recon(i)


@click.command()
@click.argument("fname")
@click.option(
    "--nscans",
    default=-1,
    help="Number of Scans to recon. By default recon all scans available. A small number is useful for testing.",
)
@click.option("--workers", default=1, help="Number of workers.")
def main(fname, nscans: int, workers: int):
    fname = Path(fname)
    savedir = fname.parent / fname.stem
    savedir.mkdir(exist_ok=True)

    if workers == 1:
        recon_bin(ioparams, params, fname, n_scans=nscans, savedir=savedir)
    else:
        recon_bin_multiproc(
            ioparams,
            params,
            fname,
            n_scans=nscans,
            savedir=savedir,
            num_workers=workers,
        )


if __name__ == "__main__":
    main()
