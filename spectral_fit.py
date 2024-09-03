import numpy as np
import matplotlib.pyplot as plt


class SpectralFit:
    def __init__(self, fs=200e6, freq_band=(12.5e6, 27.5e6)):
        self.fs = fs
        self.freq_band = freq_band

        self._spectrum = None
        self._Fx = None

    @property
    def nyquist(self):
        return self.fs / 2

    def make_Fx(self, N):
        # Assume input is rfft, i.e. length == nyquist
        return np.linspace(0, self.nyquist, N)

    def freq2idx(self, freq, N):
        # assume N is rfft length
        return int(freq / self.nyquist * N)

    def spectrum_in_band(self, roi, plot=False):
        spectrum = np.abs(np.fft.rfft(roi, axis=-1))
        N = spectrum.shape[-1]
        Fx = self.make_Fx(N)

        lower_i = self.freq2idx(self.freq_band[0], len(Fx))
        upper_i = self.freq2idx(self.freq_band[1], len(Fx))
        s = slice(lower_i, upper_i)
        i = 0
        if plot:
            plt.plot(Fx, spectrum[i])
            plt.plot(Fx[s], spectrum[i, s])

        return Fx[s], spectrum[:, s]

    def fit(self, roi):
        """
        Fit a linear least squares model to the last axis of data
        """
        Fx, spectrum = self.spectrum_in_band(roi)
        Nlines = spectrum.shape[0]

        coefs = np.zeros((Nlines, 2))
        for i in range(Nlines):
            coefs[i] = np.polyfit(Fx, spectrum[i], 1)

        self._Fx = Fx
        self._spectrum = spectrum
        self._coefs = coefs

        return coefs

    def get_avg_slope(self):
        return self._coefs[:, 0].mean()

    def plot(self, ax=None):
        if self._spectrum is None:
            print("Please run get_spectral_fit first")
            return

        Fx = self._Fx
        ROI = self._spectrum
        coefs = self._coefs

        i = int(ROI.shape[0] / 2)
        assert len(self.freq_band) == 2
        f_lower, f_upper = self.freq_band

        if ax is None:
            _, ax = plt.subplots()

        ax.plot(Fx / 1e6, ROI[i], label=f"Spectrum")
        ax.set_xlim(f_lower / 1e6, f_upper / 1e6)
        ax.set_ylim(0, None)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Spectrum")

        coef = coefs[i]
        # label = f"y = {coef[0]:.2E}*x  + {coef[1]:.2E}"
        label = f"Slope = {coef[0]:.2E}"
        ax.plot(Fx / 1e6, np.polyval(coef, Fx), label=label)
        ax.legend()
