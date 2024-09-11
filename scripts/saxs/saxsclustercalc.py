## Custom Imports
from pdbhandler import PDBFileHandler
from pdbhandler import Atom
from clusternetwork import ClusterNetwork

import sys, os, re, ast
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull
import xraydb
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

class SAXSClusterCalculation:
    def __init__(self, structurePath, expPath=None, QRange=None, solvFormula=None, solvDensity=None, solvElectronDensity=None, core_residue_names=None, shell_residue_names=None, pdb_save_folderPath=None, q_extent=None):
        """
        Initialization of the SAXSClusterCalculation class.
        
        Parameters:
        - structurePath (str): The path to the XYZ or PDB file containing atomic coordinates and elements.
        - expPath (str): Path to experimental data file for QRange determination (optional).
        - QRange (list of float): List of qmin, qmax, step for generating a q-range (optional).
        - solvFormula (str): Chemical formula of the solvent (e.g., 'C3H7NO') (optional).
        - solvDensity (float): Solvent density in g/mL (required if solvFormula is provided).
        - solvElectronDensity (float): Pre-calculated electron density in e/Å³ (optional but can validate against calculated density).
        - core_residue_names (list of str): List of core residue names in the structure (e.g., ['PBI']).
        - shell_residue_names (list of str): List of shell residue names (representing solvent) (e.g., ['DMS']).
        - pdb_save_folderPath (str): Path to folder where timestamped PDB output folders will be generated.
        - q_extent (list of float): Optional qmin, qmax, and downsampling factor to restrict the experimental data q-range (optional).

        Class Variables:
        - self.ptable (dict): Periodic table loaded from 'ptable.txt' with element symbols as keys and atomic numbers as values.
        - self.atomic_masses (dict): Atomic masses loaded from 'atomic_masses.txt' with element symbols as keys and atomic masses as values.
        - self.structurePath (str): The structure file path for the input PDB or XYZ file.
        - self.expPath (str): Path to experimental data file (if provided).
        - self.QRange (list): List of qmin, qmax, and step size for generating a q-range (if provided).
        - self.solvFormula (str): Solvent chemical formula (if provided).
        - self.solvDensity (float): Solvent density in g/mL (if provided).
        - self.solvElectronDensity (float): Provided electron density (if provided).
        - self.core_residue_names (list): List of core residue names for PDB or XYZ.
        - self.shell_residue_names (list): List of shell residue names (solvent molecules).
        - self.pdb_save_folderPath (str): Folder path where PDB outputs will be saved.
        - self.pdb_save_folder (str): Folder path with a unique timestamp for saving PDB outputs.
        - self.q_values (numpy array): Generated or loaded q-range used for the SAXS calculations.
        - self.pdb_handler (PDBFileHandler object): PDB handler object used for managing core and shell atoms.
        - self.coordinates (numpy array): Array of atomic coordinates from the structure file.
        - self.elements (numpy array): Array of atomic elements from the structure file.
        """

        # Load periodic table and atomic masses
        self.ptable = self.load_tabledata('ptable.txt')
        self.atomic_masses = self.load_tabledata('atomic_masses.txt')
        
        self.structurePath = structurePath
        self.expPath = expPath
        self.QRange = QRange
        self.solvFormula = solvFormula
        self.solvDensity = solvDensity
        self.solvElectronDensity = solvElectronDensity
        self.core_residue_names = core_residue_names
        self.shell_residue_names = shell_residue_names
        self.pdb_save_folderPath = pdb_save_folderPath
        self.q_extent = q_extent  # Store the optional q_extent

        
        self.q_values = None  # To store the calculated q-range
        self.data = None  # To store experimental data loaded from a file
        
        # Setup the PDB save folder with a timestamp
        self.setup_pdb_save_folder()

        # Load the structure file (PDB or XYZ)
        self.load_structure()

        # Set the q-range, either from the experimental data or user-provided range
        self.set_q_range()

    ## -- Initialization Methods
    def load_tabledata(self, filename):
        """
        Load data from a specified file, assuming the file contains a dictionary definition.
        """
        full_path = os.path.join(os.getcwd(), filename)
        try:
            with open(full_path, 'r') as file:
                data = file.read()
            data_dict = data.split('=', 1)[1].strip()
            table = ast.literal_eval(data_dict)
            return table
        except FileNotFoundError:
            print(f"File not found: {full_path}")
            return {}
        except SyntaxError as e:
            print(f"Error reading the data: {e}")
            return {}

    def setup_pdb_save_folder(self):
        """
        Create a timestamped folder in the designated pdb_save_folderPath to store PDB outputs.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.pdb_save_folder = os.path.join(self.pdb_save_folderPath, f'clusters_{timestamp}')
        os.makedirs(self.pdb_save_folder, exist_ok=True)

    ## -- Structure Loading Methods
    def load_structure(self):
        """
        Load PDB or XYZ file with core and shell residue names.
        """
        if self.structurePath.endswith('.pdb'):
            ## Load greater structure for cluster analysis using the PDBFileHandler
            self.pdb_handler = PDBFileHandler(self.structurePath, self.core_residue_names, self.shell_residue_names)
        elif self.structurePath.endswith('.xyz'):
            self.coordinates, self.elements = self.loadXYZ(self.structurePath)
        else:
            raise ValueError("Unsupported structure file format. Only PDB and XYZ are supported.")

    def loadXYZ(self, xyzPath):
        """
        Load atomic coordinates and elements from an XYZ file.
        """
        with open(xyzPath, 'r') as file:
            lines = file.readlines()
        atom_data = [line.split() for line in lines[2:] if len(line.split()) == 4]
        elements, coords = zip(*[(parts[0], np.array(list(map(float, parts[1:])))) for parts in atom_data])
        return np.array(coords), np.array(elements)

    def load_pdb_file(self, pdb_filepath):
        """
        Load atomic positions and elements from a PDB file using the Atom class.
        
        Parameters:
        - pdb_filepath (str): The path to the PDB file.
        
        Returns:
        - positions (numpy array): Atomic positions in the PDB file (shape: [num_atoms, 3]).
        - elements (list): List of element symbols for each atom in the PDB file.
        """
        positions = []
        elements = []

        # Read the PDB file and parse atom information
        with open(pdb_filepath, 'r') as file:
            for line in file:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    atom = self.parse_atom_line(line)
                    positions.append(list(atom.coordinates))  # Access coordinates as a tuple
                    elements.append(atom.element)

        return np.array(positions), elements

    def parse_atom_line(self, line):
        """
        Parses a line of a PDB file and returns an Atom object.
        
        Parameters:
        - line (str): A line from a PDB file.
        
        Returns:
        - atom (Atom): An Atom object.
        """
        atom_id = int(line[6:11].strip())
        atom_name = line[12:16].strip()
        residue_name = line[17:20].strip()
        residue_number = int(line[22:26].strip())
        x = float(line[30:38].strip())
        y = float(line[38:46].strip())
        z = float(line[46:54].strip())
        element = line[76:78].strip()
        return Atom(atom_id, atom_name, residue_name, residue_number, x, y, z, element)

    ## -- Q-Range Setup Methods
    def load_experimental_data(self, file_path):
        """
        Load SAXS experimental data from a text file. The file should have a header row starting with #, followed by q, I(Q), and error data.
        """
        # Read the file, skip the comment line (header), and name columns appropriately
        self.data = pd.read_csv(file_path, delim_whitespace=True, comment='#', names=['q', 'I(Q)', 'error'])
        print(f"Data loaded from {file_path}")

    def set_q_range(self):
        """
        Set the q-range using either the experimental data or user-specified QRange.
        If q_extent is provided, restrict the q-range to [qmin, qmax], with optional downsampling.
        """
        if self.expPath:
            # Load q-range from experimental data file
            self.load_experimental_data(self.expPath)

            # Extract q values from the experimental data
            q_values = self.data['q'].values

            if self.q_extent:
                qmin, qmax, downsampling = self.q_extent

                # Filter q-values based on qmin and qmax
                q_values = q_values[(q_values >= qmin) & (q_values <= qmax)]

                # Apply downsampling if downsampling > 0
                if downsampling > 0:
                    num_q_vals = len(q_values)
                    downsample_indices = np.round(np.linspace(0, num_q_vals - 1, num=int(num_q_vals // downsampling))).astype(int)
                    
                    # Ensure that qmin and qmax are included in the selected q-values
                    downsample_indices[0] = 0
                    downsample_indices[-1] = num_q_vals - 1

                    q_values = q_values[downsample_indices]

            self.q_values = q_values
            print(f"Q-range set from experimental data, restricted to {self.q_extent[:2] if self.q_extent else 'full range'}, with downsampling factor {self.q_extent[2] if self.q_extent else 'none'}.")

        elif self.QRange:
            # Generate q-range from user-provided QRange [qmin, qmax, step]
            self.q_values = np.arange(self.QRange[0], self.QRange[1] + self.QRange[2], self.QRange[2])
            print(f"Q-range set from user-provided QRange: {self.QRange}.")

        else:
            raise ValueError("Either experimental data path or QRange must be provided.")
        
    ## -- Solvent Electron Density Calculation Methods
    def calcEDensitySolv(self):
        """
        Calculate the electron density from a molecular formula and mass density, or use the input value.
        Raise an error if there's a mismatch but continue the calculation.
        """
        if self.solvElectronDensity and self.solvFormula:
            calculated_edens = self._calculate_edensity_from_formula(self.solvFormula, self.solvDensity)
            if not np.isclose(calculated_edens, self.solvElectronDensity, atol=1e-3):
                raise ValueError(f"Mismatch between provided and calculated electron densities: {self.solvElectronDensity} vs {calculated_edens}")
            return self.solvElectronDensity
        elif self.solvElectronDensity:
            return self.solvElectronDensity
        else:
            return self._calculate_edensity_from_formula(self.solvFormula, self.solvDensity)

    def _calculate_edensity_from_formula(self, molec_formula, mass_density):
        """
        Helper method to calculate electron density from molecular formula and mass density.
        """
        pattern = r'([A-Z][a-z]?)(\d*)'
        tot_electrons = 0
        tot_molarmass = 0 
        for element, count in re.findall(pattern, molec_formula):
            count = int(count) if count else 1
            atomic_number = self.ptable[element]
            molarmass = self.atomic_masses[element]

            if atomic_number:
                tot_electrons += atomic_number * count
            else:
                raise ValueError(f'Element {element} not found in ptable dictionary.')
            if molarmass:
                tot_molarmass += molarmass * count
            else:
                raise ValueError(f'Element {element} not found in molar mass dictionary.')
                
        molecular_volume = (tot_molarmass/mass_density)*(1e24/6.02e23)
        electron_dens = tot_electrons/molecular_volume
        return electron_dens

    ## -- Execute Cluster Extraction & Data Loading Methods
    def runClusterExtraction(self, node_elements=None, linker_elements=None, terminator_elements=None, segment_cutoff=None, target_elements=None, neighbor_elements=None, distance_thresholds=None):
        """
        Extract clusters, assign cluster IDs, and save PDBs to the timestamped folder.
        
        Parameters:
        - node_elements (list): Elements in the main cluster to consider as nodes. Defaults to ['Pb'].
        - linker_elements (list): Elements linking the nodes in the cluster. Defaults to ['Pb', 'I'].
        - terminator_elements (list): Elements terminating the cluster. Defaults to ['I'].
        - segment_cutoff (float): Cutoff distance for defining a cluster. Defaults to 3.8.
        - target_elements (list): Elements in the main cluster to consider for coordination. Defaults to ['Pb'].
        - neighbor_elements (list): Neighboring elements in the shell residues. Defaults to ['O'].
        - distance_thresholds (dict): Coordination thresholds. Defaults to {('Pb', 'O'): 3}.
        
        Outputs:
        - Dictionary where keys are three-character cluster names, and values are filenames in the format {originalfilename}_{i:03}.pdb.
        """
        
        # Default values if not provided
        node_elements = node_elements if node_elements is not None else ['Pb']
        linker_elements = linker_elements if linker_elements is not None else ['Pb', 'I']
        terminator_elements = terminator_elements if terminator_elements is not None else ['I']
        segment_cutoff = segment_cutoff if segment_cutoff is not None else 3.8
        target_elements = target_elements if target_elements is not None else ['Pb']
        neighbor_elements = neighbor_elements if neighbor_elements is not None else ['O']
        distance_thresholds = distance_thresholds if distance_thresholds is not None else {('Pb', 'O'): 3}

        # Output 'Cluster Finder Settings' to the user
        print("=== Cluster Finder Settings ===")
        print(f"Node Elements: {node_elements}")
        print(f"Linker Elements: {linker_elements}")
        print(f"Terminator Elements: {terminator_elements}")
        print(f"Segment Cutoff: {segment_cutoff}\n")

        # Output 'Solvent Finder Settings' to the user
        print("=== Solvent Finder Settings ===")
        print(f"Target Elements: {target_elements}")
        print(f"Neighbor Elements: {neighbor_elements}")
        print(f"Distance Thresholds: {distance_thresholds}\n")

        # Initialize ClusterNetwork with relevant parameters
        cluster_network = ClusterNetwork(
            self.pdb_handler.core_atoms,
            self.pdb_handler.shell_atoms,
            node_elements,
            linker_elements,
            terminator_elements,
            segment_cutoff,
            self.core_residue_names,
            self.shell_residue_names
        )

        # Write the cluster PDB files with the coordinated shell residues
        cluster_network.write_cluster_pdb_files_with_coordinated_shell(
            self.pdb_handler,
            self.pdb_save_folder,
            target_elements,
            neighbor_elements,
            distance_thresholds,
            self.shell_residue_names
        )

        # Generate the list of PDB files in the pdb_save_folder after they are written
        cluster_pdbs = [f for f in os.listdir(self.pdb_save_folder) if f.endswith('.pdb')]

        if len(cluster_pdbs) == 0:
            raise ValueError("No PDB files were found in the save folder. Ensure that cluster extraction worked correctly.")

        # Dictionary to hold cluster_id mappings (3-char names -> filenames)
        originalfilename = os.path.splitext(os.path.basename(self.structurePath))[0]
        self.cluster_id_dict = {}

        for cluster_pdb in cluster_pdbs:
            # Extract the 3-character cluster ID before the '.pdb'
            cluster_id_name = os.path.basename(cluster_pdb)[-7:-4]  # This extracts 'AAA' from '..._AAA.pdb'
            cluster_3char_name = f"{cluster_id_name}"  # Use the extracted 3-char cluster name
            
            # Move the cluster PDB file (if needed) and update the cluster_id_dict
            cluster_pdb_path = os.path.join(self.pdb_save_folder, cluster_pdb)
            self.cluster_id_dict[cluster_3char_name] = cluster_pdb_path

        print(f"Cluster ID Dictionary: {self.cluster_id_dict}")

        # Load all PDB clusters into memory
        self.load_all_pdb_clusters()

    def load_all_pdb_clusters(self):
        """
        Load all PDB cluster files into memory, store positions and atoms in dictionaries,
        and parse through the dataset to obtain a unique set of element names.
        """
        # Dictionaries to store positions and elements for each cluster
        self.cluster_positions = {}
        self.cluster_elements = {}

        # List to collect all unique element types across clusters
        all_elements = []

        # Loop over each cluster in the cluster_id_dict and load the corresponding PDB file
        for cluster_3char_name, pdb_filename in self.cluster_id_dict.items():
            pdb_filepath = os.path.join(self.pdb_save_folder, pdb_filename)

            # Load atomic positions and elements from the PDB file
            positions, elements = self.load_pdb_file(pdb_filepath)

            # Store positions and elements in the dictionaries
            self.cluster_positions[cluster_3char_name] = np.array(positions)
            self.cluster_elements[cluster_3char_name] = np.array(elements)

            # Add elements to the list to get unique elements later
            all_elements.extend(elements)

        # Get unique elements across all clusters and set as a class attribute
        self.unique_elements = np.unique(all_elements)
        print(f"Unique elements across all clusters: {self.unique_elements}")

    ## -- Cluster Volume Approximation Methods
    def calculate_cluster_volume(self, pdb_handler):
        """
        Calculate the convex hull volume of the atom coordinates in the cluster.
        """
        all_atoms = pdb_handler.core_atoms + pdb_handler.shell_atoms
        if len(all_atoms) < 4:
            print(f"Not enough atoms to calculate Convex Hull. Returning 0 volume.")
            return 0.0

        points = np.array([atom.coordinates for atom in all_atoms])
        hull = ConvexHull(points)
        return hull.volume

    def calculate_radius_of_gyration(self, positions, elements):
        """
        Calculate the radius of gyration (Rg) for a cluster of atoms and estimate volume.
        
        Parameters:
        - positions (list): List of atom positions (x, y, z) for the cluster.
        - elements (list): List of element symbols for the atoms in the cluster.
        
        Returns:
        - volume (float): Estimated volume based on the radius of gyration.
        """
        # Convert positions to a numpy array and calculate the center of mass
        positions = np.array(positions)
        electron_counts = np.array([self.ptable[element] for element in elements])
        center_of_mass = np.average(positions, axis=0, weights=electron_counts)

        # Calculate the radius of gyration
        squared_distances = np.sum(electron_counts * np.sum((positions - center_of_mass) ** 2, axis=1))
        total_electrons = np.sum(electron_counts)
        radius_of_gyration = np.sqrt(squared_distances / total_electrons)

        # Estimate the volume assuming a spherical distribution of atoms
        volume = (4/3) * np.pi * (radius_of_gyration ** 3)
        
        return volume
    
    def calculate_cluster_volumes_and_electrons(self, method="convex_hull"):
        """
        Iterate over the cluster .pdb files and calculate the volume of each cluster.
        
        Parameters:
        - method (str): Specify the volume calculation method, either 'convex_hull' or 'radius_of_gyration'.
        
        Returns:
        - cluster_volumes (dict): Dictionary with cluster_id as key and volume as value.
        - cluster_electrons (dict): Dictionary with cluster_id as key and total number of electrons as value.
        """
        cluster_volumes = {}
        cluster_electrons = {}
        
        # Loop over each cluster in the cluster_id_dict
        for cluster_id, pdb_filename in self.cluster_id_dict.items():
            pdb_filepath = os.path.join(self.pdb_save_folder, pdb_filename)

            # Load atomic positions and elements from the PDB file
            positions, elements = self.load_pdb_file(pdb_filepath)
            
            # Calculate the total number of electrons in the cluster
            total_electrons = sum(self.ptable[element] for element in elements)
            
            # Calculate the volume based on the chosen method
            if method == "convex_hull":
                if len(positions) < 4:
                    print(f"Not enough atoms to calculate Convex Hull for {cluster_id}. Returning 0 volume.")
                    volume = 0.0
                else:
                    hull = ConvexHull(positions)
                    volume = hull.volume
            elif method == "radius_of_gyration":
                volume = self.calculate_radius_of_gyration(positions, elements)
            else:
                raise ValueError("Invalid method. Choose either 'convex_hull' or 'radius_of_gyration'.")

            # Store the volume and total electrons in the dictionaries
            cluster_volumes[cluster_id] = volume
            cluster_electrons[cluster_id] = total_electrons

        return cluster_volumes, cluster_electrons
    
    def calc_vf(self, xyz_path, mass_conc):
        """
        Calculate the unitless volume fraction for a given solution of molecules defined in XYZ file.
        """
        with open(xyz_path, 'r') as file:
            lines = file.readlines()
        atom_data = [line.split() for line in lines[2:] if len(line.split()) == 4]
        symbols, coords = zip(*[(parts[0], np.array(list(map(float, parts[1:])))) for parts in atom_data])

        coords = np.array(coords)
        if len(coords) > 3:
            hull = ConvexHull(coords)
            molecular_volume = hull.volume
        else:
            molecular_volume = 0
            if len(coords) == 1:
                print('Insufficient atoms to create hull, approximating atom as sphere with radius 1.5Å')
                molecular_volume = (4/3)*np.pi*1.5**3
            else:
                max_distance = np.max(pdist(coords, metric='euclidean'))
                molecular_volume = max_distance*np.pi*3**2

        molec_mm = sum(self.atomic_masses[symbol] for symbol in symbols)
        num_molec_per_ml = ((molec_mm/6.02e23)**-1)*mass_conc/1000
        vol_fract = num_molec_per_ml*molecular_volume*1e-24

        return vol_fract
    
    ## -- Debye Scattering Computational Methods
    def compute_f0_dict(self):
        """
        Compute the f₀(q) dictionary for each unique element in the class attribute unique_elements.
        """
        # Ensure unique_elements has been set
        if not hasattr(self, 'unique_elements'):
            raise ValueError("Unique elements have not been set. Run load_all_pdb_clusters first.")

        # Check if q_values are available
        if self.q_values is None:
            raise ValueError("Q-range has not been set.")

        # Compute f₀(q) for each unique element across all q-values in the range
        self.f0_dictionary = {
            element: np.array([xraydb.f0(element, q/(4 * np.pi))[0] for q in self.q_values])
            for element in self.unique_elements
        }

        print(f"f₀(q) dictionary computed for unique elements: {list(self.f0_dictionary.keys())}")
        return self.f0_dictionary

    def sq_with_f0(self, pos, elements, f0_scales, qs):
        """
        Calculate the scattering profile using the Debye equation with atomic scattering factors.

        Input
        pos = scatterer positions in 3D cartesian coordinates (nx3 array)
        elements = 1D array of strings of the element symbols for each scatterer
        f0_scales = 1D array of scaling factors for f0 based on solvent electron density contrast
        qs = list of q values to evaluate scattering intensity at
        """
        nbins = len(qs)
        sq = np.zeros(nbins)
        
        # Calculate the pairwise distances between atoms
        rij_matrix = squareform(pdist(pos, metric='euclidean'))

        # Identify unique elements and precompute f0 for each element and each q value
        unique_elements = np.unique(elements)
        f0_dict = {element: np.array([xraydb.f0(element, q/(4 * np.pi))[0] for q in qs]) for element in unique_elements}

        # Map precomputed f0 values to the elements array
        f0_q_elements = np.array([f0_dict[element] for element in elements])

        # Ensure f0_scales has the correct shape (it should match f0_q_elements in shape)
        if f0_scales.ndim == 1:
            f0_scales = f0_scales[:, np.newaxis]  # Make sure it's 2D, matching (62, 100)
        
        # Apply the f0 scales to the f0_q_elements
        f0_q_elements *= f0_scales

        for i, q_val in enumerate(tqdm(qs)):
            f0_q = f0_q_elements[:, i]  # Atomic scattering factors for this q value
            
            rij_q = rij_matrix * q_val  # Pre-multiply rij by q_val to avoid repetitive computation
            
            # Compute sin(rij * q_val) / (rij * q_val) for all rij elements (this includes rij and rji)
            sinc_rij_q = np.sinc(rij_q / np.pi)
            
            # Compute contributions to S(q) for all pairs of points including self-interaction
            sq[i] += np.sum(np.outer(f0_q, f0_q) * sinc_rij_q)

        return sq

    def calculate_all_sqs(self):
        """
        Iteratively calculate S(q) for all clusters using the f₀(q) dictionary and store
        the results in a dictionary linked to the three-character cluster ID name.
        
        Returns:
        - sq_dict: Dictionary where keys are the 3-character cluster IDs and values are the corresponding S(q) arrays.
        """
        # Ensure that the f₀(q) dictionary has been computed
        if not hasattr(self, 'f0_dictionary') or self.f0_dictionary is None:
            print("f₀(q) dictionary has not been computed yet. Computing it now...")
            self.compute_f0_dict()

        # Ensure that q_values are available
        if self.q_values is None:
            raise ValueError("Q-range has not been set.")
        
        # Dictionary to store S(q) for each cluster
        self.sq_dict = {}

        # Iterate over all clusters and calculate S(q) for each
        for cluster_3char_name, cluster_filename in self.cluster_id_dict.items():
            print(f"Calculating S(q) for cluster {cluster_3char_name}...")

            # Get positions and elements for the current cluster
            pos = self.cluster_positions[cluster_3char_name]
            elements = self.cluster_elements[cluster_3char_name]

            # Map the elements to the corresponding f₀(q) values from the f₀(q) dictionary
            f0_scales = np.array([self.f0_dictionary[element] for element in elements])

            # Calculate S(q) for this cluster
            sq = self.sq_with_f0(pos, elements, f0_scales, self.q_values)

            # Store the S(q) result in the dictionary using the three-character cluster ID name
            self.sq_dict[cluster_3char_name] = sq

        print("S(q) calculation completed for all clusters.")
        return self.sq_dict

    ## -- Plotting & Saving Methods
    def plot_sq_traces(self, loglog=True, cmap='viridis'):
        """
        Plot traces of q vs. S(q) for all cluster IDs using a Viridis color map by default.
        The plot will be in log-log scale by default.

        Parameters:
        - loglog (bool): Whether to plot both axes in log-log scale. Default is True.
        - cmap (str): Color map to use for plotting. Default is 'viridis'.
        """
        # Check if the S(q) data has been computed
        if not hasattr(self, 'sq_dict') or self.sq_dict is None:
            raise ValueError("S(q) data has not been computed. Run calculate_all_sqs first.")

        # Get the number of clusters
        cluster_ids = list(self.sq_dict.keys())
        num_clusters = len(cluster_ids)

        # Create the colormap based on the number of clusters
        colors = plt.cm.get_cmap(cmap, num_clusters)

        # Initialize the plot
        plt.figure(figsize=(10, 8))

        # Loop over each cluster to plot q vs. S(q)
        for i, cluster_id in enumerate(cluster_ids):
            # Get the S(q) values and q values for the current cluster
            sq = self.sq_dict[cluster_id]
            q_values = self.q_values

            # Plot the trace for the current cluster, color based on the Viridis color map
            if loglog:
                plt.loglog(q_values, sq, label=cluster_id, color=colors(i))
            else:
                plt.plot(q_values, sq, label=cluster_id, color=colors(i))

        # Set labels and title
        plt.xlabel('q')
        plt.ylabel('S(q)')
        plt.title('q vs. S(q) for all Clusters')
        
        # Add a legend
        plt.legend(title="Cluster IDs")

        # Show the plot
        plt.show()

    def saveResults(self):
        """
        Save scattering profiles or I(Q) to text files based on user input.
        """
        pass  # Implementation for saving I(Q) or S(Q) profiles to .txt files
