import re

def parse_orca_ir(path):
    freqs = []
    intensities = []

    in_ir_section = False
    pattern = re.compile(r"\s*\d+:\s*([0-9.]+)\s+[0-9.Ee+-]+\s+([0-9.]+)")

    with open(path, "r") as f:
        for line in f:
            # Enter IR SPECTRUM section
            if "IR SPECTRUM" in line:
                in_ir_section = True
                continue

            # Leave IR section if it ends
            if in_ir_section and line.strip().startswith("----"):
                continue 
            if in_ir_section and line.strip() == "":
                break  

            # Extract values only if inside IR block
            if in_ir_section:
                m = pattern.search(line)
                if m:
                    freq = float(m.group(1))
                    inten = float(m.group(2))
                    freqs.append(freq)
                    intensities.append(inten)

    return freqs, intensities
