from pathlib import Path


class SpectrumPipeline:
def __init__(self, parser, broadener, plotter=None):
self.parser = parser
self.broadener = broadener
self.plotter = plotter


def process_folder(self, folder):
folder = Path(folder)


for sub in folder.iterdir():
if not sub.is_dir():
continue


outs = list(sub.glob("*.out"))
if not outs:
continue


out_file = outs[0]
name = out_file.stem


print(f"Processing: {name}")


freqs, ints = self.parser.parse(out_file)
grid, spectrum = self.broadener.broaden(freqs, ints)


if self.plotter:
self.plotter.plot(grid, spectrum, title=name)
