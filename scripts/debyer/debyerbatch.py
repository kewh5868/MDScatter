import os
import subprocess
import xarray as xr
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

## Custom Imports
from conversion.pdbhandler import PDBFileHandler, Atom

def calculate_number_density(atom_count, box_dimensions):
    volume = np.prod(box_dimensions)
    rho_0 = atom_count / volume
    return rho_0

class DebyerBatch:
    def __init__(self, input_dir, output_dir, filename_prefix, mode='rPDF', from_value=0.5, to_value=30, step_value=0.01, ro_value=0.05, cutoff_value=None, bounding_box=[50, 50, 50]):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.filename_prefix = filename_prefix
        self.mode = mode
        self.from_value = from_value
        self.to_value = to_value
        self.step_value = step_value
        self.ro_value = ro_value
        self.cutoff_value = cutoff_value

        # Unpack bounding_box into pbca, pbcb, and pbcc
        if len(bounding_box) != 3:
            raise ValueError("Bounding box must be a list of three values.")
        self.pbca, self.pbcb, self.pbcc = bounding_box

        # Determine the smallest bounding box dimension
        smallest_box_dimension = min(self.pbca, self.pbcb, self.pbcc)

        # Check if to_value exceeds half the smallest bounding box distance
        max_distance_allowed = 0.5 * smallest_box_dimension
        if self.to_value > max_distance_allowed:
            # Adjust to_value and inform the user
            print(f"Warning: The requested 'to_value' of {self.to_value} Å exceeds the Nyquist limit for the smallest "
                  f"bounding box dimension. Adjusting 'to_value' to {max_distance_allowed:.2f} Å, which is half the "
                  f"smallest box width ({smallest_box_dimension} Å).\n"
                  "This is necessary because distances greater than half the box width cannot be accurately computed "
                  "due to atom self-scattering inaccuracies.")
            self.to_value = max_distance_allowed

    ## Main Methods Section
        ## Notes on methods to add:
            ## add a method prune_pdb files based on residue names and residue input list. 
                ## User provides a list of residue names to preserve, copies of the PDB files are made in a new folder where all atoms with residues NOT found in the list are removed.
                ## Class maintains reference to this folder, and user can calculate partial PDF, rPDF, or RDF with this value.
                ## number densities need to be recalculated for the new box.
            ## add method to run debyer calculation on a pruned set of pdb files.
            ## add loader method for pdb files.
            ## add the shape function baseline estimate that uses g(r) to estimate baseline for conversion to R(r)

    def run_debyer(self):
        valid_modes = {'RDF': '--RDF', 'PDF': '--PDF', 'rPDF': '--rPDF'}
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid mode '{self.mode}'. Choose from 'RDF', 'PDF', or 'rPDF'.")

        # Determine the folder name based on the mode
        mode_folder_map = {'RDF': 'calcrdf', 'PDF': 'calcpdf', 'rPDF': 'calcrpdf'}
        mode_folder = mode_folder_map[self.mode]

        # Create the output directory for the mode-specific data
        output_subdir = os.path.join(self.output_dir, mode_folder)
        os.makedirs(output_subdir, exist_ok=True)

        # Create a subfolder inside the mode-specific folder using filename_prefix
        prefix_subdir = os.path.join(output_subdir, self.filename_prefix)
        os.makedirs(prefix_subdir, exist_ok=True)

        if self.cutoff_value is not None and self.to_value > self.cutoff_value:
            print(f"Warning: to_value ({self.to_value}) exceeds cutoff_value ({self.cutoff_value}). Setting to_value to cutoff_value.")
            self.to_value = self.cutoff_value

        xyz_files = sorted([f for f in os.listdir(self.input_dir) if f.endswith('.xyz')])
        total_files = len(xyz_files)

        if total_files == 0:
            print("No .xyz files found in the input directory.")
            return

        for i, file_name in tqdm(enumerate(xyz_files, start=1), desc=f"Calculating {self.mode} using Debyer", total=total_files, ncols=100, unit="file"):
            input_file_path = os.path.join(self.input_dir, file_name)
            output_file_name = f"{self.filename_prefix}_{i:04d}.txt"
            output_file_path = os.path.join(prefix_subdir, output_file_name)

            # Construct Debyer command
            debyer_command = [
                "debyer",
                valid_modes[self.mode],
                f"--pbc-a={self.pbca}",
                f"--pbc-b={self.pbcb}",
                f"--pbc-c={self.pbcc}",
                f"--from={self.from_value}",
                f"--to={self.to_value}",
                f"--step={self.step_value}",
                "--weight=x",
                "--partials",
                f"--ro={self.ro_value}",
                f"--output={output_file_path}",
                input_file_path
            ]

            if self.cutoff_value is not None:
                debyer_command.insert(2, f"--cutoff={self.cutoff_value}")

            result = subprocess.run(debyer_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                print(f"Error processing {file_name}:\n{result.stderr}")

    @staticmethod
    def run_single_debyer(input_file, output_file, mode='PDF', from_value=0.5, to_value=30, step_value=0.01, ro_value=0.05, cutoff_value=None, pbca=0, pbcb=0, pbcc=0):
        """
        Run Debyer to calculate a specified distribution function (RDF, PDF, rPDF) for a single .xyz file.

        Parameters:
            input_file (str): Path to the input .xyz file.
            output_file (str): Path to the output file where results will be saved.
            mode (str): Type of distribution function to calculate ('RDF', 'PDF', 'rPDF'). Default is 'PDF'.
            from_value (float): Starting value for the calculation range. Default is 0.5.
            to_value (float): Ending value for the calculation range. Default is 30.
            step_value (float): Step size for the calculation range. Default is 0.01.
            ro_value (float): Numeric density value. Default is 0.05.
            cutoff_value (float, optional): Cutoff distance for the calculation. If None, cutoff is not applied.

        Raises:
            ValueError: If an invalid mode is provided.
        """

        # Validate mode
        valid_modes = {'RDF': '--RDF', 'PDF': '--PDF', 'rPDF': '--rPDF'}
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Choose from 'RDF', 'PDF', or 'rPDF'.")

        # Adjust to_value if cutoff_value is provided and to_value exceeds it
        if cutoff_value is not None and to_value > cutoff_value:
            print(f"Warning: to_value ({to_value}) exceeds cutoff_value ({cutoff_value}). Setting to_value to cutoff_value.")
            to_value = cutoff_value

        # Construct Debyer command
        debyer_command = [
            "debyer",
            valid_modes[mode],
            f"--pbc-a={pbca}",             # PBC box length in x direction
            f"--pbc-b={pbcb}",             # PBC box length in y direction
            f"--pbc-c={pbcc}",             # PBC box length in z direction
            f"--from={from_value}",
            f"--to={to_value}",
            f"--step={step_value}",
            "--weight=x",
            "--partials",
            f"--ro={ro_value}",
            f"--output={output_file}",
            input_file
        ]

        # Add cutoff_value to command if provided
        if cutoff_value is not None:
            debyer_command.insert(2, f"--cutoff={cutoff_value}")

        # Run Debyer command
        result = subprocess.run(debyer_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Check for errors in execution
        if result.returncode != 0:
            print(f"Error processing {input_file}:\n{result.stderr}")
        else:
            print(f"Successfully processed {input_file} -> {output_file}")

    ## Bounding Box & Number Density Estimates
    ## Notes for code to add:
        ## modify pbca, pbcb, pbcc inputs to be bounding_box list that is unpacked into pbca, pbcb, pbcc values
            ## if the user does not pass bounding box dimensions, the program will assess the box dimension based on if the input file is .xyz or .pdb.
            ## if the input file is .pdb, the code will get the bounding box dimension from the header of the .pdb
            ## if the input file is .xyz, the code will estimate the bounding box with the estimate_bounding_box method
    def estimate_bounding_box(self, xyz_file):
        coordinates = []
        with open(xyz_file, 'r') as file:
            lines = file.readlines()
            for line in lines[2:]:
                parts = line.split()
                if len(parts) == 4:
                    coordinates.append([float(parts[1]), float(parts[2]), float(parts[3])])
        coordinates = np.array(coordinates)
        min_coords = np.min(coordinates, axis=0)
        max_coords = np.max(coordinates, axis=0)
        box_dimensions = max_coords - min_coords
        return min_coords, max_coords, box_dimensions

    ## Data Average Methods
    def load_gr_files(self):
        datasets = []
        for file_name in sorted(os.listdir(self.output_dir)):
            if file_name.endswith('.txt'):
                file_path = os.path.join(self.output_dir, file_name)
                try:
                    data_array = self.load_gr_file(file_path)
                    datasets.append(data_array)
                except Exception as e:
                    print(f"Error loading file {file_path}: {e}")
        if not datasets:
            raise ValueError("No valid data files found to concatenate.")
        combined_dataset = xr.concat(datasets, dim='file')
        return combined_dataset

    def load_gr_file(self, file_path):
        data = pd.read_csv(file_path, delim_whitespace=True, comment='#')
        with open(file_path) as f:
            for line in f:
                if line.startswith('# sum'):
                    columns = line.split()[1:]
                    break
        data.columns = ['r_A'] + columns
        data_array = data.set_index('r_A').to_xarray()
        data_array = data_array.assign_coords(r_A=data['r_A'])
        data_array['r_A'].attrs['units'] = 'angstroms'
        for column in columns:
            data_array[column].attrs['units'] = '1/angstrom^2'
        return data_array

    def average_datasets(self, dataset):
        avg_dataset = dataset.mean(dim='file')
        return avg_dataset

    def save_averaged_dataset(self, avg_dataset, output_file):
        df = avg_dataset.to_dataframe().reset_index()
        df.to_csv(output_file, sep=' ', index=False)

    ## Plotting Methods
    def plot_averaged_GR(self, avg_dataset, include_partials=False):
        plt.figure(figsize=(10, 6))
        r_A = avg_dataset['r_A']
        plt.plot(r_A, avg_dataset['sum'], label='avg', color='black')
        if include_partials:
            for column in avg_dataset.data_vars:
                if column != 'sum':
                    plt.plot(r_A, avg_dataset[column], label=column)
        plt.xlabel('r (Å)')
        plt.ylabel('G(r) (1/Å²)')
        plt.title('Averaged Pair Distribution Function (PDF)')
        plt.legend()
        plt.show()

    def plot_averaged_rr(self, avg_dataset, include_partials=False):
        plt.figure(figsize=(10, 6))
        r_A = avg_dataset['r_A']
        plt.plot(r_A, avg_dataset['sum'], label='avg', color='black')
        if include_partials:
            for column in avg_dataset.data_vars:
                if column != 'sum':
                    plt.plot(r_A, avg_dataset[column], label=column)
        plt.xlabel('r (Å)')
        plt.ylabel('R(r) (1/Å)')
        plt.title('Averaged Radial Distribution Function (RDF)')
        plt.legend()
        plt.show()

    def plot_averaged_gr(self, avg_dataset, include_partials=False):
        plt.figure(figsize=(10, 6))
        r_A = avg_dataset['r_A']
        plt.plot(r_A, avg_dataset['sum'], label='avg', color='black')
        if include_partials:
            for column in avg_dataset.data_vars:
                if column != 'sum':
                    plt.plot(r_A, avg_dataset[column], label=column)
        plt.xlabel('r (Å)')
        plt.ylabel('g(r)')
        plt.title('Averaged PDF g(r)')
        plt.legend()
        plt.show()

    def plot_all_gr_traces(self, include_partials=False):
        plt.figure(figsize=(12, 8))
        for file_name in sorted(os.listdir(self.output_dir)):
            if file_name.endswith('.txt'):
                file_path = os.path.join(self.output_dir, file_name)
                try:
                    gr_data = self.load_gr_file(file_path)
                    plt.plot(gr_data['r_A'], gr_data['sum'], label=file_name, lw=1)
                    if include_partials:
                        for column in gr_data.data_vars:
                            if column != 'sum':
                                plt.plot(gr_data['r_A'], gr_data[column], linestyle='--', lw=0.5)
                except Exception as e:
                    print(f"Error plotting file {file_path}: {e}")
        plt.xlabel('r (Å)')
        plt.ylabel('G(r) (1/Å²)')
        plt.title('G(r) Traces from Folder')
        plt.yscale('log')
        plt.xscale('log')
        plt.legend(loc='upper right', fontsize='small', ncol=2)
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.show()

    ## Conversion Methods, R(r) -> g(r), G(r), G(r) -> R(r), g(r) -> R(r), G(r)
        ## Notes to add code:
            ## add methods to convert from R(r) -> g(r), G(r)
            ## add methods to convert from G(r) -> R(r), g(r)
            ## add methods to convert from g(r) -> R(r), G(r)

    def extract_first_two_columns(self, file_path):
        base_name, ext = os.path.splitext(file_path)
        output_file = f"{base_name}_extracted{ext}"
        with open(file_path, 'r') as file:
            lines = file.readlines()
        header = lines[0].split()
        first_two_columns = [header[0], header[1]]
        data = []
        for line in lines[1:]:
            values = line.split()
            data.append([values[0], values[1]])
        with open(output_file, 'w') as file:
            file.write(f"# {first_two_columns[0]} {first_two_columns[1]}\n")
            for row in data:
                file.write(f"{row[0]} {row[1]}\n")
        print(f"Extracted columns saved to {output_file}")

    def convert_gr_to_Gr(input_file, ro_value, r_min=0.0, r_max=None):
        """
        Converts g(r) to G(r), with options to interpolate g(r), G(r), or neither.
        Outputs a file with columns: interpolated r_A and corresponding G(r)
        
        Parameters:
        input_file (str): Path to the input file containing g(r) data.
        rho_naught (float): Number density of scatterers in atoms/Å³.
        r_min (float, optional): Minimum r value to include in the output data.
        r_max (float, optional): Maximum r value to include in the output data.
        r_step (float, optional): Step size for resampling the data over r.
        """
        # Read the input file, skipping lines that start with '#'
        data = np.loadtxt(input_file, comments='#')
        r_data = data[:, 0]
        g_r_data = data[:, 1]

        # Apply r_min and r_max to the original data
        if r_min is not None:
            mask = r_data >= r_min
            r_data = r_data[mask]
            g_r_data = g_r_data[mask]
        else:
            r_min = r_data[0]

        if r_max is not None:
            mask = r_data <= r_max
            r_data = r_data[mask]
            g_r_data = g_r_data[mask]
        else:
            r_max = r_data[-1]

        # No interpolation; use original data
        G_r_new = 4 * np.pi * r_data * ro_value * (g_r_data - 1)
        r_new = r_data

        # Create output file name with '_converted' suffix
        output_file = input_file.replace('.txt', '_converted.gr')

        # Prepare the header
        header = (
            "# r_A\tG(r)\n"
            f"# rho_naught = {ro_value} atoms/Å³\n"
        )

        # Save the result to the new file
        np.savetxt(output_file, np.column_stack((r_new, G_r_new)), header=header, comments='', fmt='%.6e')

        print(f"Data saved to {output_file}")

class prunePDB:
    def __init__(self, input_folder, allowed_residues):
        """
        Initialize the prunePDB class with the input folder and allowed residue list.

        Parameters:
        - input_folder (str): The folder containing PDB files.
        - allowed_residues (list): List of residue names to retain in the PDB files.
        """
        self.input_folder = input_folder
        self.allowed_residues = allowed_residues
        
        # Create a suffix based on allowed residues
        self.residue_suffix = "_pruned_" + "_".join(allowed_residues)
        
        # Create the output folder for pruned PDB files, appending allowed residues to the name
        self.pruned_folder = os.path.join(input_folder, f"pruned_pdbs{self.residue_suffix}")
        os.makedirs(self.pruned_folder, exist_ok=True)

        # Create the output folder for converted XYZ files
        self.xyz_folder = os.path.join(input_folder, f"converted_xyzs{self.residue_suffix}")
        os.makedirs(self.xyz_folder, exist_ok=True)

    def prune_pdb(self, pdb_file):
        """
        Prune atoms from the PDB file that do not match the allowed residues.

        Parameters:
        - pdb_file (str): Path to the PDB file to prune.

        Returns:
        - pruned_file (str): Path to the pruned PDB file.
        """
        # Load the PDB file using PDBFileHandler
        pdb_handler = PDBFileHandler(pdb_file, self.allowed_residues, [])

        # Collect the atoms that belong to allowed residues
        pruned_atoms = [atom for atom in pdb_handler.core_atoms if atom.residue_name in self.allowed_residues]
        
        # Generate the output PDB file name with the residue suffix
        pdb_filename = os.path.splitext(os.path.basename(pdb_file))[0] + self.residue_suffix + ".pdb"
        pruned_file = os.path.join(self.pruned_folder, pdb_filename)

        # Write the pruned atoms to the output PDB file
        pdb_handler.write_pdb_file(pruned_file, pruned_atoms)

        return pruned_file

    def convert_pdb_to_xyz(self, pdb_file):
        """
        Convert a pruned PDB file to an XYZ file format.

        Parameters:
        - pdb_file (str): Path to the pruned PDB file to convert.

        Returns:
        - xyz_file (str): Path to the generated XYZ file.
        """
        # Generate the output XYZ file name with the residue suffix
        xyz_filename = os.path.splitext(os.path.basename(pdb_file))[0] + self.residue_suffix + ".xyz"
        xyz_file = os.path.join(self.xyz_folder, xyz_filename)

        # Read the pruned PDB file using PDBFileHandler
        pdb_handler = PDBFileHandler(pdb_file, self.allowed_residues, [])

        # Use the provided Atom class to convert to XYZ
        atom_details = pdb_handler.get_atom_details()

        # Write the XYZ file format
        with open(xyz_file, 'w') as xyz_out:
            xyz_out.write(f"{len(atom_details)}\n")  # Write the number of atoms
            xyz_out.write(f"Converted from {pdb_file}\n")  # XYZ file comment
            for atom_id, atom_name, element, coords in atom_details:
                x, y, z = coords
                xyz_out.write(f"{element} {x:.3f} {y:.3f} {z:.3f}\n")

        return xyz_file

    def process_all_files(self):
        """
        Process all PDB files in the input folder by pruning and converting to XYZ format.
        """
        pdb_files = [f for f in os.listdir(self.input_folder) if f.endswith(".pdb")]

        if not pdb_files:
            print("No PDB files found in the input folder.")
            return

        # Use tqdm to track the overall progress
        with tqdm(total=len(pdb_files), desc="Processing PDB files", ncols=100) as pbar:
            for pdb_file in pdb_files:
                pdb_path = os.path.join(self.input_folder, pdb_file)

                # Prune the PDB file
                pruned_pdb = self.prune_pdb(pdb_path)

                # Convert the pruned PDB to XYZ format
                self.convert_pdb_to_xyz(pruned_pdb)

                # Update progress bar
                pbar.update(1)

        print("All files processed successfully.")
