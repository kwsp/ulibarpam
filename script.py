# %%
import matplotlib.pyplot as plt

from ulibarpam import IOParams, load_scans


path = "F:/tmp/20231025 ex vivo 249/152449PAUS.bin"
i = 311
aline_i = 0

rf_all = load_scans(path, i)
rf_all_sub = rf_all - rf_all.mean(axis=0)
aline = rf_all[aline_i]
aline_sub = rf_all_sub[aline_i]

ioparams = IOParams.default()

# %%
f, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(aline)
ax[0].set_title("Aline RF (Raw)")
ax[0].set_xlabel("Samples")
ax[0].set(xlim=[0, len(aline)], ylim=[0, None])

ax[0].axvline(ioparams.rf_size_PA, c="orange")
ax[0].axvline(ioparams.rf_size_PA + ioparams.rf_size_spacer, c="orange")
ax[0].axvline(
    ioparams.rf_size_PA + ioparams.rf_size_spacer + ioparams.rf_size_US, c="orange"
)

ax[1].plot(aline_sub)
ax[1].set_title("Aline RF (Background subtracted)")
ax[1].set_xlabel("Samples")
ax[1].set(xlim=[0, len(aline_sub)], ylim=[0, None])

ax[1].axvline(ioparams.rf_size_PA, c="orange")
ax[1].axvline(ioparams.rf_size_PA + ioparams.rf_size_spacer, c="orange")
ax[1].axvline(
    ioparams.rf_size_PA + ioparams.rf_size_spacer + ioparams.rf_size_US, c="orange"
)

f.tight_layout()

# %%
from scipy import signal
from ulibarpam import debug_recon, split_rf_USPA, ReconParams

debug_US = True
debug_PA = True

rf_PA, rf_US = split_rf_USPA(aline_sub, ioparams)
params = ReconParams.default()

kernel_PA = signal.firwin2(65, *params.filter_PA)
kernel_US = signal.firwin2(65, *params.filter_US)
debug_recon(rf_PA, kernel_PA, params.noise_floor_PA, params.desired_dynamic_range_PA)
plt.gcf().suptitle("PA")
plt.gcf().tight_layout()

debug_recon(rf_US, kernel_US, params.noise_floor_US, params.desired_dynamic_range_US)
plt.gcf().suptitle("US")
plt.gcf().tight_layout()

# %%
%load_ext autoreload
%autoreload 2

# %%
from ulibarpam import recon_one_scan_
import ulibarpam


rf_PA, rf_US = split_rf_USPA(rf_all_sub, ioparams)
# env = ulibarpam.recon(rf_US, kernel_US)
# env = np.abs(signal.hilbert(rf_US))
all_rect, all_radial, meta = recon_one_scan_(rf_PA, rf_US, params, flip=bool(i % 2))
# img, db = ulibarpam.log_compress(env, params.noise_floor_PA, params.desired_dynamic_range_PA)

f, ax = plt.subplots()
ax.imshow(all_rect, "gray", extent=[0, 1, 0, 1])

# %%
plt.imshow(all_radial)

# %%
all_rect

# %%
all_radial.max()

# %%
params.noise_floor_PA

# %%
import numpy as np
import ulibarpam

tmp, db = ulibarpam.log_compress(np.abs(signal.hilbert(rf_US)), 200, 40)

plt.imshow(tmp.T, "gray", extent=[0, 1, 0, 1])


# %%
from pprint import pprint

params.desired_dynamic_range_PA = 45
params.desired_dynamic_range_US = 45
pprint(params)

# %%
meta
