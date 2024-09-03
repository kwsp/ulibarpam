# %%
from pathlib import Path
from tqdm import tqdm
import numpy as np


def loadOldBin(path: Path):
    rf = np.fromfile(path, dtype=">d")
    return rf.reshape((1000, int(rf.size / 1000)))


def get_sequence_pairs(session_path: Path):
    files = sorted(p for p in session_path.glob("*") if p.suffix == ".bin")
    assert len(files) % 2 == 0
    pairs = []
    for i in range(0, len(files), 2):
        pa, us = files[i], files[i + 1]
        assert pa.stem.endswith("PA")
        assert us.stem.endswith("US")
        assert pa.stem[:6] == us.stem[:6]
        pairs.append((pa, us))
    return pairs


def get_new_bin_name(session_path: Path, seq=None):
    newRoot = Path("converted")
    newRoot.mkdir(exist_ok=True)

    newDir = newRoot / session_path.name
    newDir.mkdir(exist_ok=True)

    # Use the first sequence time as the new bin name
    if seq is None:
        pairs = get_sequence_pairs(session_path)
        seq = pairs[0][0].stem[:6]
    return newDir / ("converted" + seq + ".bin")


def load_old_pa_us_pair(pa: Path, us: Path):
    def load_multiscan(p: Path, samples_per_line: int):
        rf = np.memmap(p, dtype=np.double)
        nscans = int(rf.size / 1000 / samples_per_line)
        return rf.reshape((nscans, 1000, samples_per_line))

    return load_multiscan(pa, 3200), load_multiscan(us, 6400)


# %%
from tqdm import tqdm

rfSize = 2**13

PAoffset = 250
PAsize = 3200 - PAoffset

USoffset = 500
USstart = 2732
USsize = 6400 - USoffset

print(f"{rfSize=} {PAoffset=} {PAsize=} {USoffset=} {USstart=} {USsize=}")
print(8192 / 3)


def combine_US_PA(USbin: np.ndarray, PAbin: np.ndarray):
    rf = np.zeros((1000, rfSize), dtype=USbin.dtype)

    rf[:, :PAsize] = PAbin[:, PAoffset : PAoffset + PAsize]
    USsize_ = min(USsize, rfSize - USstart)
    rf[:, USstart : USstart + USsize_] = USbin[:, USoffset : USoffset + USsize_]

    # Convert from double to uint16
    rfNew = ((rf / 4 + 0.5) * (2**16)).astype(np.uint16)
    return rfNew


# %%
500 / (180e6) * 1540

# %%
Fs = 180e6
SoundSpeed = 1500


PAsamplesSave = 2730 - 100
USsamplesSave = 5460 + 100 + 2
PAtruncate = 300
UStruncate = 500
totalSamplesSave = PAsamplesSave + USsamplesSave
totalSamplesAcquire = totalSamplesSave + PAtruncate + UStruncate

# PAsamplesSave = 2730
# USsamplesSave = 5460 + 2
# totalSamplesSave = PAsamplesSave + USsamplesSave
# PAtruncate = 0
# UStruncate = 0


def make(PAsamplesSave, PAtruncate, USsamplesSave, UStruncate):
    totalSave = PAsamplesSave + USsamplesSave
    assert totalSave == 8192

    PAdepthTotal = (PAsamplesSave + PAtruncate) / Fs * SoundSpeed
    USdepthTotal = (USsamplesSave + UStruncate) / Fs * SoundSpeed / 2

    PAdepthTruncated = (PAtruncate) / Fs * SoundSpeed
    USdepthTruncated = (UStruncate) / Fs * SoundSpeed / 2

    print(f"Saving {PAsamplesSave=} {USsamplesSave=} ")
    print(f"{totalSamplesSave=}, {totalSamplesAcquire=}")
    print(f"PA imaging depth {PAdepthTotal}, truncated {PAdepthTruncated}")
    print(f"US imaging depth {USdepthTotal}, truncated {USdepthTruncated}")


make(PAsamplesSave, PAtruncate, USsamplesSave, UStruncate)

# %%
2730 * 2

# %%


def convert_old_PA_US_session(session_path):
    # One output sequence
    # out = get_new_bin_name(session_path)
    # with open(out, "wb") as fp:
    #     pairs = get_sequence_pairs(session_path)
    #     # pa, us = pairs[0]
    #     for pa, us in tqdm(pairs):
    #         PA, US = load_old_pa_us_pair(pa, us)
    #         assert PA.shape[0] == US.shape[0]
    #         for i in range(PA.shape[0]):
    #             combine_US_PA(US[i], PA[i]).tofile(fp)

    # Multiple output sequence
    pairs = get_sequence_pairs(session_path)
    # pa, us = pairs[0]
    for pa, us in pairs:
        seq = pa.stem[:6]
        out = get_new_bin_name(session_path, seq)
        with open(out, "wb") as fp:
            print(f"[{session_path.stem}] Converting {seq}...")
            PA, US = load_old_pa_us_pair(pa, us)
            assert PA.shape[0] == US.shape[0]
            for i in tqdm(range(PA.shape[0])):
                combine_US_PA(US[i], PA[i]).tofile(fp)


# session_path = Path("E:/Data/ARPAM_Rectum/in vivo data/invivo_20220406_235")
session_path = Path("E:/Data/ARPAM_Rectum/in vivo data/invivo_20220405_234")

convert_old_PA_US_session(session_path)


# %%
with open("tmp.bin", "wb") as fp:
    for US, PA in list_scans_in_sequence(old_sessions[0] / "135427"):
        combine_US_PA(loadOldBin(US), loadOldBin(PA)).tofile(fp)

# rfNew = combine_US_PA(USbin, PAbin)
# rfNew.tofile("tmp.bin")
