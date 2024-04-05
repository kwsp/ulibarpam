from dataclasses import dataclass
from pathlib import Path
from scipy import signal
import numpy as np
import cv2


### Constants
RF_ALINE_SIZE = 2**13
"Size of A-line is 8192, defined in DAQ"


### IO
@dataclass
class IOParams:
    rf_size_PA: int
    rf_size_spacer: int
    rf_size_US: int
    "No. of samples in US RF. Must be twice rf_size_PA"

    offset_US: int
    offset_PA: int
    "PA Aline offset (in addition to offset_US to match the US location"

    @staticmethod
    def default():
        "Config for Sitai's late 2023 to 2024 Labview data"
        return IOParams(
            rf_size_PA=2650,
            rf_size_spacer=87,
            rf_size_US=5300,
            offset_US=350,
            offset_PA=215,
        )


def get_num_scans(fname, alines_per_bscan=1000):
    with open(fname, "rb") as fp:
        fp.seek(0, 2)  # seek to end
        fsize = fp.tell()

    num_scans = fsize // (RF_ALINE_SIZE * alines_per_bscan * 2)
    return num_scans


def load_scans(fname, i: int, nscans=1, alines_per_bscan=1000):
    """
    Load `nscans` from the `i`th scan position
    """
    assert nscans >= 1

    with open(fname, "rb") as fp:
        scan_size = RF_ALINE_SIZE * alines_per_bscan * 2
        fp.seek(1 + scan_size * i)
        raw = fp.read(RF_ALINE_SIZE * alines_per_bscan * nscans * 2)

    buf = np.frombuffer(raw, np.uint16)
    if nscans == 1:
        return buf.reshape((alines_per_bscan, RF_ALINE_SIZE))
    return buf.reshape((nscans, alines_per_bscan, RF_ALINE_SIZE))


def split_rf_USPA(rf: np.ndarray, ioparams: IOParams):
    # Get raw RF data for PA and US
    rf_PA = rf[:, : ioparams.rf_size_PA]
    _US_start = ioparams.rf_size_PA + ioparams.rf_size_spacer
    _US_end = _US_start + ioparams.rf_size_US
    rf_US = rf[:, _US_start:_US_end]

    # apply offsets
    rf_US = np.roll(rf_US, ioparams.offset_US, axis=-1)
    rf_US[:, : ioparams.offset_US] = 0

    offset_PA = ioparams.offset_US // 2 + ioparams.offset_PA
    rf_PA = np.roll(rf_PA, offset_PA, axis=-1)
    rf_PA[:, :offset_PA] = 0
    return rf_PA, rf_US


### Recon
@dataclass
class ReconParams:
    filter_PA: tuple[list, list]
    filter_US: tuple[list, list]

    noise_floor_PA: int
    noise_floor_US: int

    desired_dynamic_range_PA: int
    desired_dynamic_range_US: int

    aline_rotation_offset: int
    "When flipped, how many A-lines need to be shifted to account for rotation offset"

    @staticmethod
    def default():
        recon_params = ReconParams(
            filter_PA=([0, 0.03, 0.035, 0.2, 0.22, 1], [0, 0, 1, 1, 0, 0]),
            filter_US=([0, 0.1, 0.3, 1], [0, 1, 1, 0]),
            noise_floor_PA=250,
            noise_floor_US=200,
            desired_dynamic_range_PA=40,
            desired_dynamic_range_US=48,
            aline_rotation_offset=24,
        )
        return recon_params


def recon_one_scan_(
    rf_PA: np.ndarray, rf_US: np.ndarray, params: ReconParams, flip=False
):
    # Compute filter kernels
    kernel_PA = signal.firwin2(65, *params.filter_PA)
    kernel_US = signal.firwin2(65, *params.filter_US)

    # Recon PA
    PA_env = recon(rf_PA, kernel_PA)
    PA = 255 * log_compress(
        PA_env, params.noise_floor_PA, params.desired_dynamic_range_PA
    )

    # Recon US
    US_env = recon(rf_US, kernel_US)
    US = 255 * log_compress(
        US_env, params.noise_floor_US, params.desired_dynamic_range_US
    )

    if flip:
        US = np.flipud(US)
        PA = np.flipud(PA)

        # shift A scans to account for rotation mismatch
        rotate_offset = params.aline_rotation_offset
        US = np.roll(US, rotate_offset, 0)
        PA = np.roll(PA, rotate_offset, 0)

    # Scan conversion
    PA_rect = make_rectangular(PA).astype(np.uint8)
    PA_radial = make_radial(PA).astype(np.uint8)

    US_rect = make_rectangular(US).astype(np.uint8)
    US_radial = make_radial(US).astype(np.uint8)

    all_rect = np.hstack((PA_rect, US_rect))
    all_radial = np.hstack((PA_radial, US_radial))

    return all_rect, all_radial


def recon_one_scan(rf: np.ndarray, ioparams: IOParams, params: ReconParams, flip=False):
    rf_PA, rf_US = split_rf_USPA(rf, ioparams)
    return recon_one_scan_(rf_PA, rf_US, params, flip)


def write_images(path: str | Path, PAUS_img):
    ncols = PAUS_img.shape[1] // 2
    PA = PAUS_img[:, :ncols]
    US = PAUS_img[:, ncols:]
    img = np.empty((PAUS_img.shape[0], ncols * 3, 3))

    img[:, :ncols] = PA[:, :, np.newaxis]
    img[:, ncols : ncols * 2] = US[:, :, np.newaxis]
    img[:, ncols * 2 :] = make_overlay(US, PA)

    cv2.imwrite(str(path), img)


### Recon helpers


def apply_fir_filt(x: np.ndarray, kernel: np.ndarray):
    x_filt = np.empty_like(x, dtype=np.double)
    for i in range(x.shape[0]):
        x_filt[i] = np.convolve(x[i], kernel, "same")
    return x_filt


def recon(rf: np.ndarray, kernel: np.ndarray):
    rf_filt = apply_fir_filt(rf, kernel)
    rf_env = np.abs(signal.hilbert(rf_filt))
    return rf_env


def log_compress(x, noise_floor, desired_dynamic_range_dB=45):
    # The noise floor should be constant for a given system and is predetermined
    # The desired DR after compression is used to scale

    # Determine the peak signal value.
    peak_level = np.max(x)
    dynamic_range_dB = 20 * np.log10(peak_level / noise_floor)
    print(f"Dynamic range: {dynamic_range_dB} dB")

    # Apply log compression
    x_log = 20 * np.log10(x) - 20 * np.log10(noise_floor)
    # Normalize to the desired DR and cut off values below 0
    x_log = np.clip(x_log, 0, desired_dynamic_range_dB)
    # Scale to the range [0, 1]
    x_log /= desired_dynamic_range_dB
    return x_log


### Scan conversion


def make_rectangular(img):
    img = cv2.resize(img, (1000, 1000))
    return img.T


def make_radial(img):
    """
    Input image has alines for rows
    """
    h, w = img.shape
    r = min(h, w)
    img = cv2.resize(img, (r * 2, r * 2))
    img = cv2.warpPolar(img, img.shape[:2], (r, r), r, cv2.WARP_INVERSE_MAP)
    img = cv2.resize(img, (r, r))
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def grey2heatmap(grey: np.ndarray) -> np.ndarray:
    "Convert PA image from grey to BGR heatmap"
    return cv2.applyColorMap(grey, cv2.COLORMAP_HOT)


def make_overlay_(PAUS):
    ncols = PAUS.shape[1] // 2
    PA = PAUS[:, :ncols]
    US = PAUS[:, ncols:]
    return make_overlay(US, PA)


def make_overlay(US, PA):
    # Convert US to BGR
    img = cv2.cvtColor(US, cv2.COLOR_GRAY2BGR)

    # Convert PA to BGR (if not already)
    if len(PA.shape) == 2:
        PA = grey2heatmap(PA)
    logic_a = PA.sum(2) > 10
    img[logic_a] = PA[logic_a]
    return img
