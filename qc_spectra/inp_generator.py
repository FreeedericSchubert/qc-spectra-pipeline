import os
import re
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolfiles

class InputGenerator:
    """
    General QC input generator.
    Provides common methods for generating 3D coordinates and saving XYZ files.
    Spectroscopy-specific input files are generated via dedicated methods.
    """

    def __init__(self, output_dir="orca_inputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def sanitize(smiles):
        """Make SMILES safe to use as a folder/file name."""
        return re.sub(r'[^A-Za-z0-9._-]+', '_', smiles)

    def generate_xyz(self, mol, seed=123):
        """
        Generate a 3D conformer for the molecule and return coordinates as string.
        """
        molH = Chem.AddHs(mol)
        AllChem.EmbedMolecule(molH, randomSeed=seed)
        AllChem.MMFFOptimizeMolecule(molH)

        xyz_block = rdmolfiles.MolToXYZBlock(molH)
        xyz_lines = xyz_block.splitlines()
        coords_only = "\n".join(xyz_lines[2:])  # skip atom count + comment
        return molH, coords_only

    def generate_ir_input(self, mol, header="! PBE def2-SVP Opt Freq", charge=0, mult=1, seed=123):
        """
        Generate ORCA input for IR calculation.
        """
        smiles = Chem.MolToSmiles(mol)
        safe_name = self.sanitize(smiles)

        # Generate XYZ coordinates
        molH, coords_only = self.generate_xyz(mol, seed=seed)

        # Create folder for this molecule
        mol_folder = os.path.join(self.output_dir, safe_name)
        os.makedirs(mol_folder, exist_ok=True)

        # Build ORCA input text
        inp_text = f"{header}\n\n* xyz {charge} {mult}\n{coords_only}\n*\n"
        inp_path = os.path.join(mol_folder, f"{safe_name}.inp")

        # Write the file
        with open(inp_path, "w") as f:
            f.write(inp_text)

        return inp_path
