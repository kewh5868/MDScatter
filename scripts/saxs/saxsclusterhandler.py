import os, ast
import numpy as np
import pandas as pd
import xraydb
from scipy.spatial.distance import pdist, squareform

## Custom Imports
from conversion.pdbhandler import PDBFileHandler, Atom
from cluster.clusterbatchanalyzer import ClusterBatchAnalyzer
from cluster.clusternetwork import ClusterNetwork


class SAXSClusterHandler:
    def __init__(self,
                 analyzer: ClusterBatchAnalyzer, 
                 QRange=None, 
                 expPath=None, 
                 q_extent=None):
        """
        Initialization of the SAXSClusterHandler class.

        Input Parameters:
        - QRange (list of float): List of qmin, qmax, step for generating a q-range (optional).
        - expPath (str): Path to experimental data file for QRange determination (optional).
        - q_extent (list of float): Optional qmin, qmax, and downsampling factor to restrict the experimental data q-range (optional).
        
        Class Variables:
        - self.ptable (dict): Periodic table loaded from 'ptable.txt' with element symbols as keys and atomic numbers as values.
        - self.atomic_masses (dict): Atomic masses loaded from 'atomic_masses.txt' with element symbols as keys and atomic masses as values.
        - self.QRange (list): List of qmin, qmax, and step size for generating a q-range (if provided).
        - self.expPath (str): Path to experimental data file (if provided).
        - self.q_extent (list of float): Optional qmin, qmax, and downsampling factor to restrict the experimental data q-range (optional).
        """
    
        # Initialize the passed ClusterBatchAnalyzer
        self.analyzer = analyzer
        self.solution_electron_density_A3 = None # This will be set fomr the analyzer BulkVolume attribute 'bulk_volume' attribute

        ## Assert that the clusterbatch analyzer that was passed is valid and has valid PDB paths.
        self._assertvalid_ClusterBatchAnalyzer()
        self._assertvalid_PDBPaths()
        self._assertvalid_BulkVolume()

        # Load Periodic Table & Atomic Masses
        self.ptable = self.load_tabledata('ptable.txt')
        self.atomic_masses = self.load_tabledata('atomic_masses.txt')

        # Q-Range Setup
        self.QRange = QRange
        self.expPath = expPath
        self.q_extent = q_extent
        self.q_values = None  # To store the calculated q-range in set_q_range()
        
        ## Set the q-range, either from the experimental data or user-provided range
        self.set_q_range()

        ## PDB Coordinate & Element Information
        # self.cluster_id_dict = {} # create a dictionary of unique cluster identifiers
        self.cluster_coords = {} # create a dictionary of coordinates for each cluster
        self.cluster_elements = {} # create a dictionary of elements for each cluster coordinate
        self.cluster_volumes = {} # create a dictionary of volumes for each cluster
        self.cluster_electron_density = {} # create a dictionary of electron densities for each cluster
        self.cluster_delta_rho = {} # create a dictionary of electron density contrasts for each cluster
        
        ## f0 Dictionary Definition
        self.f0_unique_atoms = None # list of unique elements to build the f0 dictionary 
        self.f0_dictionary = {} # dictionary of f0 data arrays
    
    def _assertvalid_ClusterBatchAnalyzer(self):
        ## Ensure that the loaded clusterbatchanalyzer object is valid.
        ## Is there a valid pdb_directory with .xyz or .pdb files?
        ## Is there a valid coordination_stats_per_size attribute (not None)
        ## Is there a valid self.cluster_size_distribution attribute?
        ## Is there a valid self.cluster_data attribute?

        ## Remind of cluster_data format:
        # self.cluster_data.append({
        # 'pdb_file': pdb_file,
        # 'cluster_size': cluster_size,
        # 'coordination_stats': coordination_stats,
        # 'volume': cluster_volume,
        # 'charge': cluster_charge,
        # 'electron_density': electron_density,
        # 'delta_rho': delta_rho})

        return 
    
    def _assertvalid_PDBPaths(self):
        ## Use the cluster_data variable from the ClusterBatchAnalyzer to extract the 'pdb_file' filepaths.
        ## Validate these filepaths exist
        return
    
    def _assertvalid_BulkVolume(self):
        ## Check analyzer.bulk_volume.electrons_info is not None
        ## Check analyzer.bulk_volume.solution_electron_density_A3 has been calculated. 
            ## If not, run analyzer.bulk_volume.calculate_solution_electron_density() method
        ## Set self.solution_electron_density_A3 = analyzer.bulk_volume.solution_electron_density_A3
        return
    
    def _load_tabledata(self, filename):
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

    def _loadclusterXYZ(self, xyz_filepath):
        """
        Load atomic coordinates and elements from an XYZ file.
        """
        with open(xyz_filepath, 'r') as file:
            lines = file.readlines()
        atom_data = [line.split() for line in lines[2:] if len(line.split()) == 4]
        elements, coords = zip(*[(parts[0], np.array(list(map(float, parts[1:])))) for parts in atom_data])

        # returns the coordinates for atoms and their corresponding elements.
        return np.array(coords), np.array(elements)
    
    def _loadclusterPDB(self, pdb_filepath):
        """
        Load atomic positions and elements from a PDB file using the Atom class.
        
        Parameters:
        - pdb_filepath (str): The path to the PDB file.
        
        Returns:
        - positions (numpy array): Atomic positions in the PDB file (shape: [num_atoms, 3]).
        - elements (list): List of element symbols for each atom in the PDB file.
        """

        def parse_atom_line(line):
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
        
        positions = []
        elements = []

        # Read the PDB file and parse atom information
        with open(pdb_filepath, 'r') as file:
            for line in file:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    atom = parse_atom_line(line)
                    positions.append(list(atom.coordinates))  # Access coordinates as a tuple
                    elements.append(atom.element)

        ## update this to have the sameoutput format and variable names as _loadclusterXYZ
        return np.array(positions), np.array(elements)
    
    def set_q_range(self):
        """
        Set the q-range using either the experimental data or user-specified QRange.
        If q_extent is provided, restrict the q-range to [qmin, qmax], with optional downsampling.
        """

        def load_experimental_data(self, file_path):
            """
            Load SAXS experimental data from a text file. The file should have a header row starting with #, followed by q, I(Q), and error data.
            """
            # Read the file, skip the comment line (header), and name columns appropriately
            self.data = pd.read_csv(file_path, delim_whitespace=True, comment='#', names=['q', 'I(Q)', 'error'])
            print(f"Data loaded from {file_path}")

        if self.expPath:
            # Load q-range from experimental data file
            load_experimental_data(self.expPath)

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
    
    def _loadall_pdb_clusters(self):
        """
        Load all PDB cluster files into memory, store positions and atoms in dictionaries,
        and parse through the dataset to obtain a unique set of element names.
        """
        all_elements = []
        for index, pdb_filepath in self.analyzer.cluster_data[index]['pdb_file']:
            
            # extract filename from the pdb path for cluster_id_name key
            cluster_id_name = os.path.splitext(os.path.basename(pdb_filepath))
            coords, elements = self.load_pdb_file(pdb_filepath)

            # append cluster coordinates and atoms to reference dictionaries
            self.cluster_coords[cluster_id_name] = np.array(coords)
            self.cluster_elements[cluster_id_name] = np.array(elements)
            self.cluster_volumes[cluster_id_name] = self.analyzer.cluster_data[index]['volume']
            self.cluster_electron_density[cluster_id_name] = self.analyzer.cluster_data[index]['electron_density']
            self.cluster_delta_rho[cluster_id_name] = self.analyzer.cluster_data[index]['delta_rho']

            all_elements.extend(elements)

        ## Create a list of unique elements across all PDB files.
        self.f0_unique_atoms = np.unique(all_elements)
        return
        
    def _compute_f0_dict(self):
        """
        Compute the f₀(q) dictionary for each unique element in the class attribute unique_elements.
        """
        # Ensure unique_elements has been set
        if not hasattr(self, 'f0_unique_atoms'):
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
        # Update f0_dictionary to custom import, include anomalous scattering factors
        return self.f0_dictionary
    
    def _compute_f0_qelements_wsf(self, cluster_id_name):
        """
        Compute the temporary scaled f₀(q) dictionary for electron density contrast, given the cluster volume, coordinates, and cluster electron density.
        Weak scatterers are filtered, indicated by '_wsf' in the method name.
        """

        if not hasattr(self, 'f0_dictionary'):
            raise ValueError("f₀(q) dictionary must be computed before f0_scales can be computed for a given cluster.")
    
        if not hasattr(self, 'cluster_coords'):
            raise ValueError("PDB files must be loaded before scaled f₀(q) dictionary can be computed.")
        
        cluster_n_atoms = len(self.cluster_coords[cluster_id_name]) # set the number of atoms to the length of the number of elements in the cluster
        cluster_coords = self.cluster_coords[cluster_id_name]
        cluster_elements = self.cluster_elements[cluster_id_name]
        cluster_volume = self.cluster_volumes[cluster_id_name] # set the volume from the cluster volume
        # cluster_electron_density = self.cluster_electron_density[cluster_id_name] # set the electron density from the cluster electron density

        # CALCULATE SCALE FACTORS
        ## Take the electron density of the solution, distribute the contrast density evenly over the atoms in the cluster volume.
        scale_factor = self.solution_electron_density_A3 * cluster_volume / cluster_n_atoms

        ## Calculate the electron scale factor for each element in the cluster elements list
        scaled_electrons = np.array([self.ptable[element] - scale_factor for element in cluster_elements])

        # FILTERING WEAK SCATTERERS
        weak_scatterer_mask = scaled_electrons >=1 # filter scatterers with weak or negligible contrast < 1
        
        ## Obtain a new set of scatterer coordinates and elements
        debye_coords = np.array(cluster_coords)[weak_scatterer_mask]
        debye_elements = np.array(cluster_elements)[weak_scatterer_mask]

        # SETUP F0(Q) SCALING FOR THE COMPUTATION
        ## Obtain a new set of f0 scaling factors for the f0_dictionary
        f0_scales = scaled_electrons[weak_scatterer_mask]/ [self.ptable[element] for element in debye_elements]
        f0_scales = np.asarray(f0_scales)

        ## Generate the f0_qelements array that maps f0_dictionary to the 
        f0_qelements = np.array([self.f0_dictionary[element] for element in cluster_elements])

        # Ensure f0_scales has the correct shape (it should match f0_q_elements in shape)
        if f0_scales.ndim == 1:
            f0_scales = f0_scales[:, np.newaxis]

        f0_qelements *= f0_scales

        return debye_coords, debye_elements, f0_qelements

    def _compute_media_f0avg(self):
        ## compute the favg
        return
    
    def _compute_media_contrast(self):
        ## for a given atom fi, average media favg, q, and rc value, compute the media contrast term
        return
    