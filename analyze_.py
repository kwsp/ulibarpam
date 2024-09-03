# %%
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from dataclasses import dataclass


# %%
@dataclass
class INPUT:
    path: Path
    normal_oclock: tuple[int, int]
    tumor_oclock: tuple[int, int]


def load_float_bin(path, dtype=np.float32):
    buf = np.fromfile(path, dtype)
    nrows = int(buf.size / 1000)
    return buf.reshape((1000, nrows))


def oclock_to_ascans(*args, nAscans=1000):
    return [round(nAscans * (arg / 12)) for arg in args]


# %%
DESKTOP = Path.home() / "DESKTOP"


def get_patient(pid="206"):
    root = DESKTOP / pid
    paths = [p for p in root.glob("*") if p.is_dir()]
    return root, paths


# %%
# p = DESKTOP / "invivo_12042019_206_converted094057_109"
# USenv = load_float_bin(p / "USenv.bin")
# PAenv = load_float_bin(p / "PAenv.bin")

# normal_range_oclock = (5, 10)
# tumor_range_oclock = (0, 2)


# roi = USenv[slice(*oclock_to_ascans(*normal_range_oclock))]
# print(roi.shape)
# plt.imshow(roi)


import spectral_fit
import importlib

importlib.reload(spectral_fit)
SpectralFit = spectral_fit.SpectralFit


def run_analysis(
    pid: str,
    oclock: tuple[int, int],
    suffix: str,
    debug=False,
):
    root, paths = get_patient(pid)
    if debug:
        paths = paths[:1]

    spectralPA = SpectralFit()
    spectralUS = SpectralFit()

    _slice = slice(*oclock_to_ascans(*oclock))

    avg_slopesPA = np.zeros(len(paths))
    avg_slopesUS = np.zeros(len(paths))
    for i, p in enumerate(paths):
        PAraw = load_float_bin(p / "PArf.bin")
        PAroi = PAraw[_slice, 500:-500]
        PAroi = cv2.medianBlur(PAroi, 3)

        USraw = load_float_bin(p / "USrf.bin")
        USroi = USraw[_slice, 500:-500]

        spectralPA.fit(PAroi)
        avg_slopesPA[i] = spectralPA.get_avg_slope()

        spectralUS.fit(USroi)
        avg_slopesUS[i] = spectralUS.get_avg_slope()

        if debug:
            f, ax = plt.subplots(2, 2, figsize=(6, 6))
            f.suptitle(f"Spectral fit analysis\n{p.stem}")

            # ax[0][0].imshow(PAroi, extent=[-1, 1, -1, 1])
            # ax[0][1].imshow(USroi, extent=[-1, 1, -1, 1])
            ax[0][0].plot(PAroi[0])
            ax[0][0].set_title("PA signal")
            ax[0][1].plot(USroi[0])
            ax[0][1].set_title("US signal")

            spectralPA.plot(ax[1][0])
            ax[1][0].set_title("PA spectrum")
            spectralUS.plot(ax[1][1])
            ax[1][1].set_title("US spectrum")

            f.tight_layout()

            f.savefig(root / f"spectral_fit_analysis_sample_{suffix}.png", dpi=200)

    fpath = root / f"avg_slopes_{Path().stem}_{suffix}.pkl"

    obj = {"avg_slopesPA": avg_slopesPA, "avg_slopesUS": avg_slopesUS}
    import pickle

    with open(fpath, "wb") as fp:
        pickle.dump(obj, fp)

    print(f"Saved slopes to {fpath}")
    msg = (
        f"PA: {avg_slopesPA.mean()} +- {avg_slopesPA.std()};  "
        f"US: {avg_slopesUS.mean()} +- {avg_slopesUS.std()};  "
    )
    print(msg)

    return obj


params = (("226", (5, 7), "tumor"),)
run_analysis(*param, debug=True)

# %%
RESULTS = {}

# %%
params = [
    ("206", (5, 10), "normal"),
    ("206", (0, 2), "tumor"),
    ("234", (0, 2), "tumor"),
    ("234", (6, 9), "normal"),
    ("212", (10, 12), "tumor"),
    ("227", (1.5, 2.5), "tumor"),
    ("227", (7, 10), "normal"),
    ("210", (9, 10), "tumor"),
    ("210", (2, 4), "normal"),
    ("226", (5, 7), "tumor"),
    ("226", (9, 12), "normal"),
]

for param in params:
    RESULTS[(param[0], param[2])] = run_analysis(*param)


# %%
def gen_hist_x(n):
    return (np.random.random(n) - 0.5) / 10


pidss = [
    # ("206", "234"),
    ("206",),
    ("212", "227"),
    ("210", "226"),
]
xticks = ("TRG 3", "TRG 2", "TRG 0")
f, ax = plt.subplots()

boxplot_y = []
boxplot_pos = []
for i, pids in enumerate(pidss):
    ally = []
    for pid in pids:
        y = RESULTS[(pid, "tumor")]["avg_slopesPA"]
        y = np.abs(y)
        ally.append(y)
        x = gen_hist_x(len(y)) + i
        label = f"{pid} ({xticks[i]}, n={len(y)})"
        ax.scatter(x, y, label=label, marker=".")

    ally = np.concatenate(ally)
    boxplot_y.append(ally)
    boxplot_pos.append(i)

i = len(pidss)
xticks += ("normal",)
ally = []
for pid in ("206", "234", "227", "210", "226"):
    y = RESULTS[(pid, "normal")]["avg_slopesPA"]
    y = np.abs(y)
    ally.append(y)
    x = gen_hist_x(len(y)) + i
    ax.scatter(x, y, marker=".", color="k")

ally = np.concatenate(ally)
boxplot_y.append(ally)
boxplot_pos.append(i)

ax.boxplot(boxplot_y, positions=boxplot_pos)

title = "PA Spectral fit of MVD cohort\n" "No significant difference found"
ax.set_title(title)
ax.set_xticks(range(len(xticks)), xticks)
ax.set_ylabel("Spectral fit slope")
ax.legend()
f.savefig("MVD_PA_spectral_fit.png", dpi=200)

# %%

# %%

# %%

# %%
(np.random.random(n) - 0.5) / 10
# %%

# %%
"""
TODO

Quantify spectrum - linear fit
"""
import spectral_fit
import importlib

importlib.reload(spectral_fit)
SpectralFit = spectral_fit.SpectralFit

spectral_fit = SpectralFit()

# %%
coefs = spectral_fit.fit(roi)
slope = spectral_fit.get_avg_slope()
print(slope)

spectral_fit.plot()

# %%
