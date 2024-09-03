# %%
from pathlib import Path
import numpy as np


def list_sequences_in_session(session: Path):
    sequences = [
        p
        for p in session.glob("*")
        if p.is_dir() and p.name.isdigit() and len(p.name) == 6
    ]
    return sorted(sequences, key=lambda p: p.name)


def list_scans_in_sequence(sequence: Path):
    for i in range(1, 13):
        US = sequence / f"NormalUS{i}.bin"
        PA = sequence / f"NormalPA{i}.bin"
        if US.exists() and PA.exists():
            yield US, PA
        else:
            break


## Check each sequence has 6 pairs of US and PA
def check_sequence(sequence: Path):
    for US, PA in list_scans_in_sequence(sequence):
        assert US.exists()
        assert PA.exists()


def loadOldBin(path: Path):
    rf = np.fromfile(path, dtype=">d")
    return rf.reshape((1000, int(rf.size / 1000)))


def check_session(session_path: Path):
    for sequence in list_sequences_in_session(session_path):
        try:
            check_sequence(sequence)
        except Exception as e:
            print(f"Check sequence for {sequence} failed! Continuing...")
            raise e


def getNewBinName(session_path: Path):
    newRoot = Path("converted")
    newRoot.mkdir(exist_ok=True)

    newDir = newRoot / session_path.name
    newDir.mkdir(exist_ok=True)

    # Use the first sequence time as the new bin name
    sequence = list_sequences_in_session(session_path)[0]
    return newDir / ("converted" + sequence.name + ".bin")


# %%
from tqdm import tqdm

rfSize = 2**13

PAoffset = 500
PAsize = 3200 - PAoffset

USoffset = 1000
USstart = 2732
USSize = 6400 - USoffset


def combine_US_PA(USbin: np.ndarray, PAbin: np.ndarray):
    rf = np.zeros((1000, rfSize))

    rf[:, :PAsize] = PAbin[:, PAoffset : PAoffset + PAsize]
    USsize_ = min(USSize, rfSize - USstart)
    rf[:, USstart : USstart + USsize_] = USbin[:, USoffset : USoffset + USsize_]

    # Convert from double to uint16
    rfNew = ((rf / 4 + 0.5) * (2**16)).astype(np.uint16)
    return rfNew


def convert_old_session(session_path: Path):
    check_session(session_path)

    newBinName = getNewBinName(session_path)
    print(f"Converting {session_path} to {newBinName}")

    with open(newBinName, "wb") as fp:
        sequences = list_sequences_in_session(session_path)
        for sequence in tqdm(sequences):
            for US, PA in list_scans_in_sequence(sequence):
                USbin = loadOldBin(US)
                PAbin = loadOldBin(PA)

                rfNew = combine_US_PA(USbin, PAbin)

                rfNew.tofile(fp)


PAsize, USSize, PAsize + USSize

# %%
oldInvivo = session_path = Path("E:/Data/ARPAM_Rectum/in vivo data")
old_sessions = list(oldInvivo.glob("in_vivo*")) + list(oldInvivo.glob("invivo*"))

# %%
for session_path in old_sessions:
    try:
        convert_old_session(session_path)
    except Exception as e:
        print(e)
        print(f"Failed to convert {session_path}! Skipping...")


# %%
with open("tmp.bin", "wb") as fp:
    for US, PA in list_scans_in_sequence(old_sessions[0] / "135427"):
        combine_US_PA(loadOldBin(US), loadOldBin(PA)).tofile(fp)

# rfNew = combine_US_PA(USbin, PAbin)
# rfNew.tofile("tmp.bin")

# %%
2800 + 5460

# %%
