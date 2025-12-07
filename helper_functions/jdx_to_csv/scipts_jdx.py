import numpy as np
import re
import csv

# -------------------------
#  Convert T → A
# -------------------------
def transmittance_to_absorbance(y):
    y = np.array(y, dtype=float)

    # detect percent transmittance
    if np.max(y) > 2:
        y = y / 100.0

    # avoid log10(0)
    y = np.clip(y, 1e-9, 1.0)

    return -np.log10(y)


# -------------------------
#  Convert micrometers → cm^-1
# -------------------------
def micrometer_to_wavenumber(um):
    um = np.array(um, dtype=float)
    return 10000.0 / um  # 1 cm = 10000 µm


# -------------------------
#  Load JDX File
# -------------------------
def load_jdx(filepath):
    with open(filepath, "r") as f:
        raw = f.read()

    # extract the XYDATA section
    match = re.search(r"##XYDATA=\(X\+\+\(Y\.\.Y\)\)(.*)", raw, re.S)
    if not match:
        raise ValueError("XYDATA section not found")

    data_block = match.group(1)

    x_vals = []
    y_vals = []

    for line in data_block.strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue

        x0 = float(parts[0])
        ys = [float(v) for v in parts[1:]]

        # x increments from X++ rule: linear spacing via DELTAX
        for i, y in enumerate(ys):
            x_vals.append(x0 + i * 0.0)  # placeholder, replaced below
            y_vals.append(y)

    # clean arrays
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)

    # get FIRSTX and DELTAX from header
    firstx = float(re.search(r"##FIRSTX=([0-9\.]+)", raw).group(1))
    deltax = float(re.search(r"##DELTAX=([0-9\.]+)", raw).group(1))

    # reconstruct true x scale
    x_vals = firstx + np.arange(len(x_vals)) * deltax

    # convert micrometers → wavenumber
    x_wavenumber = micrometer_to_wavenumber(x_vals)

    return x_wavenumber, y_vals


# -------------------------
#  Save CSV
# -------------------------
def save_csv(x, y, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Wavenumber_cm^-1", "Absorbance"])

        for xx, yy in zip(x, y):
            writer.writerow([xx, yy])


# -------------------------
#  FULL PIPELINE
# -------------------------
def convert_jdx_to_csv(jdx_file, csv_file):
    x, y = load_jdx(jdx_file)

    # convert transmittance → absorbance
    y_abs = transmittance_to_absorbance(y)

    # sort by descending wavenumber (typical IR)
    order = np.argsort(-x)
    x = x[order]
    y_abs = y_abs[order]

    save_csv(x, y_abs, csv_file)
    print(f"Saved CSV: {csv_file}")


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    convert_jdx_to_csv("Acetone_IR_Exp.jdx", "Acetone_IR_Exp.csv")
