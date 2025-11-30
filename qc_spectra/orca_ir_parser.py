import re

class Parser:
    def __init__(self):
        self.ir_pattern = re.compile(
            r"\s*\d+:\s*([0-9.]+)\s+[0-9.Ee+-]+\s+([0-9.]+)"
        )

    def parse_ir(self, path):
        freqs = []
        intensities = []
        in_ir_section = False

        with open(path, "r") as f:
            for line in f:

                # Start IR section
                if "IR SPECTRUM" in line:
                    in_ir_section = True
                    continue

                # Exit IR section
                if in_ir_section and line.strip() == "":
                    break
                if in_ir_section and line.strip().startswith("----"):
                    continue

                # Extract values
                if in_ir_section:
                    m = self.ir_pattern.search(line)
                    if m:
                        freqs.append(float(m.group(1)))
                        intensities.append(float(m.group(2)))

        return freqs, intensities
