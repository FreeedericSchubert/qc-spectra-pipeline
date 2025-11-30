import subprocess
import time
from pathlib import Path
import csv

class OrcaRunner:
    """
    Run ORCA on a batch of input files and maintain a master CSV log.
    """

    def __init__(self, orca_exe: Path, main_folder: Path, log_file_name="master_log.csv"):
        self.orca_exe = orca_exe
        self.main_folder = main_folder
        self.master_log_file = main_folder / log_file_name
        self.main_folder.mkdir(parents=True, exist_ok=True)

        # Create log CSV if it doesn't exist
        if not self.master_log_file.exists():
            with open(self.master_log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Run_Folder", "Input_File", "Output_File", "Runtime_sec", "Status"])

    def run_batch(self):
        """Loop over subfolders and run ORCA on each .inp file."""
        with open(self.master_log_file, "a", newline="") as log_csv:
            log_writer = csv.writer(log_csv)

            for run_folder in self.main_folder.iterdir():
                if not run_folder.is_dir():
                    continue

                inp_files = list(run_folder.glob("*.inp"))
                if not inp_files:
                    print(f"No .inp file in {run_folder.name}, skipping...")
                    log_writer.writerow([run_folder.name, "", "", 0, "No input"])
                    continue

                inp_file = inp_files[0]  # assume one input per folder
                output_file = run_folder / f"{inp_file.stem}.out"

                print(f"Running ORCA for {run_folder.name} ...")
                start_time = time.time()
                status = "Success"

                try:
                    with open(output_file, "w") as f_out:
                        subprocess.run(
                            [str(self.orca_exe), str(inp_file)],
                            check=True,
                            stdout=f_out,
                            stderr=subprocess.STDOUT
                        )
                except subprocess.CalledProcessError:
                    print(f"ORCA failed for {run_folder.name}")
                    status = "Failed"
                except Exception as e:
                    print(f"Unexpected error in {run_folder.name}: {e}")
                    status = "Error"

                elapsed = time.time() - start_time
                print(f"{run_folder.name} finished in {elapsed:.2f} sec\n")

                # Log the run
                log_writer.writerow([run_folder.name, str(inp_file), str(output_file), round(elapsed, 2), status])
