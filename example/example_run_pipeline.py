from pathlib import Path
from orca_pipeline.input_generator import InputGenerator
from orca_pipeline.orca_runner import OrcaRunner
from orca_pipeline.parser import OrcaParser
from orca_pipeline.fingerprinting import Fingerprinting
from orca_pipeline.logger import Logger

# paths
sdf_file = Path("examples/example_data/small_dataset.sdf")
output_dir = Path("examples/output_demo")

# initialize components
logger = Logger(output_dir / "master_log.csv")
generator = InputGenerator(output_dir=output_dir, logger=logger)
runner = OrcaRunner(orca_exe=Path("path/to/orca.exe"),
                    main_folder=output_dir,
                    logger=logger,
                    n_threads_orca=1,
                    n_workers=2)
parser = OrcaParser()
fpgen = Fingerprinting()

# 1. Generate ORCA inputs
for idx, mol in enumerate(generator.load_sdf(sdf_file), 1):
    generator.generate_ir_input(mol, mol_idx=idx)

# 2. Run ORCA jobs
runner.run_batch()

# 3. Parse ORCA outputs
for mol_dir in output_dir.iterdir():
    out_file = next(mol_dir.glob("*.out"), None)
    if out_file:
        peaks = parser.parse_ir(out_file)
        print(f"{mol_dir.name}: peaks={peaks}")

# 4. Generate Morgan fingerprints
for mol_dir in output_dir.iterdir():
    mol_file = next(mol_dir.glob("*.mol"), None)  # or use xyz
    if mol_file:
        mol = generator.load_mol(mol_file)
        fp = fpgen.Morgan_FP(mol)
        print(f"{mol_dir.name}: fingerprint={fp")
