import numpy as np


class GaussianBroadener:
def __init__(self, fwhm=20.0, normalize=True, padding=100):
self.fwhm = fwhm
self.normalize = normalize
self.padding = padding


def broaden(self, freqs, intensities):
freqs = np.array(freqs)
intensities = np.array(intensities)


# frequency grid
grid_min = freqs.min() - self.padding
grid_max = freqs.max() + self.padding
grid = np.linspace(grid_min, grid_max, 3000)


sigma = self.fwhm / (2 * np.sqrt(2 * np.log(2)))


spectrum = np.zeros_like(grid)


for f, I in zip(freqs, intensities):
gauss = np.exp(-(grid - f) ** 2 / (2 * sigma**2))
if self.normalize:
gauss /= gauss.max() if gauss.max() != 0 else 1
spectrum += I * gauss


return grid, spectrum
