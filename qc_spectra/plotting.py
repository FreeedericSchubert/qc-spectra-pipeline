import matplotlib.pyplot as plt

class IRPlotter:
def plot(self, grid, spectrum, title="IR Spectrum"):
plt.figure(figsize=(10, 4))
plt.plot(grid, spectrum)
plt.gca().invert_xaxis() # conventional IR axis direction
plt.title(title)
plt.xlabel("Wavenumber (cm^-1)")
plt.ylabel("Intensity (a.u.)")
plt.tight_layout()
plt.show()
