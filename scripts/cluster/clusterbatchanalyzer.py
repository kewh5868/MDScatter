import os, shutil, mendeleev, xraydb
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.spatial import ConvexHull
from scipy.linalg import eigh
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from tqdm.notebook import tqdm  # Import for Jupyter Notebook visual progress bar
from mendeleev import element  # To fetch ionic radii
from datetime import datetime
from collections import defaultdict
from typing import Optional, Dict

# Custom Method Imports
from conversion.pdbhandler import PDBFileHandler
from cluster.radiusofgyration import RadiusOfGyrationCalculator
from tools.bulkvolume import BulkVolume, BulkVolumeParams

class ClusterBatchAnalyzer:
    def __init__(self, 
                 pdb_directory, 
                 target_elements, 
                 neighbor_elements, 
                 distance_thresholds, 
                 charges, 
                 core_residue_names=['PBI'], 
                 shell_residue_names=['DMS'], 
                 volume_method: Optional[str] = None,
                 bulk_volume_params: Optional[BulkVolumeParams] = None, 
                 copy_no_target_files=False):
        
        # Cluster Search Variables
        self.target_elements = target_elements
        self.neighbor_elements = neighbor_elements
        self.distance_thresholds = distance_thresholds
        self.core_residue_names = core_residue_names
        self.shell_residue_names = shell_residue_names

        # Statistics Information
        self.cluster_data = []
        self.cluster_size_distribution = defaultdict(list)
        self.cluster_coordination_stats = {}
        self.charges = charges

        # File Management Information
        self.pdb_directory = pdb_directory
        self.pdb_files = self._load_pdb_files()
        self.copy_no_target_files = copy_no_target_files
        self.no_target_atoms_files = []

        self.sorted_folder = None  # Attribute to store the sorted folder path
        self.no_node_elements_folder = None  # Attribute for storing path for files with no node elements

        # Volume Methods Initialization
        self.volume_method = volume_method
        self.bulk_volume_params = bulk_volume_params
        self.bulk_volume = None

        self.coordination_stats_per_size = None
        
        ## Ionic Radius Calculation Method
        if self.volume_method == 'ionic_radius':
            # Only build the ionic radius lookup table if required
            self.radius_lookup, _ = self.build_ionic_radius_lookup()
        
        ## Bulk Volume Calculation Method
        if self.volume_method == 'bulk_volume':
            self._initialize_bulk_volume()
            
    ## -- Supporting Methods
    def _load_pdb_files(self):
        return [os.path.join(self.pdb_directory, f) for f in os.listdir(self.pdb_directory) if f.endswith('.pdb')]

    def get_atomic_number(self, element):
        """
        Returns the atomic number of a given element.

        Parameters:
        - element: str, chemical symbol of the element.
        
        Returns:
        - atomic_number: int, atomic number of the element.
        """
        periodic_table = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
            'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
            'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 
            'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 
            'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
            'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 
            'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58,
            'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 
            'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74,
            'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 
            'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
            'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98,
            'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 
            'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112, 
            'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
        }
        return periodic_table[element]
    
    @staticmethod
    def determine_safe_thread_count(task_type='cpu', max_factor=2):
        ''' Evaluate the number of threads available for an io-bound or cpu-bound task. '''
        num_cores = os.cpu_count() or 1  # Fallback to 1 if os.cpu_count() returns None

        if task_type == 'cpu':
            # For CPU-bound tasks: use a minimum of 1 and a maximum of num_cores
            thread_count = max(1, num_cores - 1)
        elif task_type == 'io':
            # For I/O-bound tasks: consider using more threads
            thread_count = max(1, num_cores * max_factor)
        else:
            raise ValueError("task_type must be 'cpu' or 'io'")

        return thread_count
    
    def generate_ascii_table(self, multiplicity_counts, title="Multiplicity Counts"):
        """
        Generate and print an ASCII table from the given multiplicity counts.

        Parameters:
        - multiplicity_counts (dict): A nested dictionary {element: {multiplicity: count}}
        - title (str): Title of the table
        """
        import pandas as pd

        # Find all unique multiplicities
        multiplicities = set()
        for counts in multiplicity_counts.values():
            multiplicities.update(counts.keys())
        multiplicities = sorted(multiplicities)

        # Create a DataFrame
        df = pd.DataFrame(index=sorted(multiplicity_counts.keys()), columns=multiplicities).fillna(0)

        # Fill the DataFrame
        for element, counts in multiplicity_counts.items():
            for multiplicity, count in counts.items():
                df.loc[element, multiplicity] = count

        # Convert DataFrame to ASCII table
        print(f"\n{title}")
        print(df.to_string())

    ## -- File Sorting Methods
    def sort_pdb_files_by_node_count(self, node_elements):
        """
        Sort PDB files into subfolders based on the count of node elements in each file.
        The sorted subfolders are placed in a new directory with 'sorted_' as a prefix to the original folder name.
        
        Parameters:
        - node_elements (list): List of node elements to count in each PDB file (e.g., ['Pb']).
        """
        # Create a new folder named 'sorted_<original_folder_name>'
        original_folder_name = os.path.basename(self.pdb_directory)
        self.sorted_folder = os.path.join(os.path.dirname(self.pdb_directory), f'sorted_{original_folder_name}')
        os.makedirs(self.sorted_folder, exist_ok=True)

        # Folder to handle cases with no node elements (ac000)
        self.no_node_elements_folder = os.path.join(self.sorted_folder, f'node_no_element_ac000')
        os.makedirs(self.no_node_elements_folder, exist_ok=True)

        # Loop through each PDB file and count node elements
        for pdb_file in self.pdb_files:
            pdb_handler = PDBFileHandler(pdb_file, core_residue_names=self.core_residue_names, 
                                         shell_residue_names=self.shell_residue_names)

            # Count the node elements in the PDB file
            node_count = defaultdict(int)
            for atom in pdb_handler.core_atoms:
                if atom.element in node_elements:
                    node_count[atom.element] += 1

            if node_count:  # If node elements are found
                for node_element, count in node_count.items():
                    # Format the folder name based on the node element and its count (e.g., node_Pb_ac005)
                    folder_name = f'node_{node_element}_ac{count:03}'
                    folder_path = os.path.join(self.sorted_folder, folder_name)
                    os.makedirs(folder_path, exist_ok=True)

                    # Copy the PDB file to the appropriate subfolder
                    output_file_path = os.path.join(folder_path, os.path.basename(pdb_file))
                    shutil.copy2(pdb_file, output_file_path)

                    print(f"Copied {pdb_file} to {output_file_path}")
            else:  # If no node elements are found, place in 'node_no_element_ac000' folder
                output_file_path = os.path.join(self.no_node_elements_folder, os.path.basename(pdb_file))
                shutil.copy2(pdb_file, output_file_path)
                print(f"Copied {pdb_file} to {self.no_node_elements_folder}")

        print(f"Sorted PDB files are stored in: {self.sorted_folder}")

    ## -- Cluster Analysis Methods
    def analyze_clusters(self, shape_type='sphere', output_folder='no_target_atoms', copy_no_target_files=False):
        all_cluster_sizes = []
        coordination_stats_per_size = defaultdict(lambda: defaultdict(list))
        no_target_atoms_count = 0
        electron_lookup = {}  # Reusable electron lookup table

        if copy_no_target_files:
            os.makedirs(output_folder, exist_ok=True)

        num_threads = self.determine_safe_thread_count(task_type='cpu')

        ## Setup processing for a single cluster file
        def process_pdb_file(pdb_file):
            try:
                pdb_handler = PDBFileHandler(pdb_file, core_residue_names=self.core_residue_names, 
                                            shell_residue_names=self.shell_residue_names)

                target_atoms = [atom for atom in pdb_handler.core_atoms if atom.element in self.target_elements]

                if not target_atoms:
                    self.no_target_atoms_files.append(pdb_file)
                    return None, None, None, None, None

                cluster_size = len(target_atoms)
                coordination_stats, _ = self.calculate_coordination_numbers(pdb_handler, target_atoms)

                # Initialize cluster_charge
                cluster_charge = None

                if self.volume_method == 'ionic_radius':
                    cluster_volume = self.estimate_total_molecular_volume(pdb_handler)
                elif self.volume_method == 'radius_of_gyration':
                    atom_charges = [self.charges[atom.element][0] for atom in target_atoms]
                    rg_calculator = RadiusOfGyrationCalculator(
                        atom_positions=[atom.coordinates for atom in target_atoms],
                        atom_elements=[atom.element for atom in target_atoms],
                        atom_charges=atom_charges,
                        electron_lookup=electron_lookup  # Pass the reusable lookup table
                    )
                    if shape_type == 'sphere':
                        cluster_volume = rg_calculator.calculate_volume(method='sphere')
                    elif shape_type == 'ellipsoid':
                        cluster_volume, Rgx, Rgy, Rgz = rg_calculator.calculate_volume(method='ellipsoid')
                        # Make sure to include Rgx, Rgy, and Rgz in the return statement below if you need to store them
                    else:
                        raise ValueError(f"Unknown shape type: {shape_type}")
                elif self.volume_method =='bulk_volume':
                    # cluster_volume = self.calculate_pdb_volume(pdb_handler)
                    cluster_volume, electron_density, delta_rho = self.calculate_pdb_volume_and_electron_density(pdb_handler)
                else:
                    raise ValueError(f"Unknown volume method: {self.volume_method}")

                # Calculate cluster charge
                cluster_charge = self.calculate_cluster_charge(pdb_handler)

                return pdb_file, cluster_size, coordination_stats, cluster_volume, cluster_charge, electron_density, delta_rho
            except Exception as e:
                print(f"Error processing file {pdb_file}: {e}")
                return None, None, None, None, None

        ## Here is where we batch process all of the clsuter information.
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {executor.submit(process_pdb_file, pdb_file): pdb_file for pdb_file in self.pdb_files}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing PDB files", ncols=100):
                result = future.result()
                if result[0] is not None:
                    pdb_file, cluster_size, coordination_stats, cluster_volume, cluster_charge, electron_density, delta_rho = result
                    all_cluster_sizes.append(cluster_size)
                    for pair, (avg_coord, _) in coordination_stats.items():
                        coordination_stats_per_size[cluster_size][pair].append(avg_coord)

                    self.cluster_data.append({
                        'pdb_file': pdb_file,
                        'cluster_size': cluster_size,
                        'coordination_stats': coordination_stats,
                        'volume': cluster_volume,
                        'charge': cluster_charge,
                        'electron_density': electron_density,
                        'delta_rho': delta_rho
                    })
                    self.cluster_size_distribution[cluster_size].append(cluster_volume)
                else:
                    no_target_atoms_count += 1

        if self.copy_no_target_files:
            for pdb_file in self.no_target_atoms_files:
                shutil.copy(pdb_file, output_folder)

        print(f"Number of files without target atoms: {no_target_atoms_count}")

        self.generate_statistics()

        ## Plotting the output of the statistical distribution
        self.plot_cluster_size_distribution(all_cluster_sizes)
        self.plot_coordination_histogram(coordination_stats_per_size)
        if self.volume_method == 'ionic_radius':
            self.plot_average_volume_vs_cluster_size()
        elif self.volume_method == 'radius_of_gyration':
            self.plot_average_volume_vs_cluster_size_rg()
        elif self.volume_method == 'bulk_volume':
            self.plot_average_volume_vs_cluster_size()
        
        ## NOTE: Need to update this to generalize box_size input or to grab this dynamically.
        self.plot_volume_percentage_of_scatterers(box_size_angstroms=53.4, num_boxes=250)
        self.plot_phi_Vc_vs_cluster_size()

        self.coordination_stats_per_size = coordination_stats_per_size

        return coordination_stats_per_size

    def generate_statistics(self):
        """
        Generate statistics for cluster size distribution, calculating the average and standard deviation of volumes.
        """
        cluster_size_distribution = defaultdict(list)
        
        for data in self.cluster_data:
            cluster_size = data['cluster_size']
            cluster_volume = data['volume']
            
            cluster_size_distribution[cluster_size].append(cluster_volume)
        
        average_volumes = {}
        for size, volumes in cluster_size_distribution.items():
            average_volumes[size] = np.mean(volumes)
        
        self.average_volumes_per_size = average_volumes
        self.cluster_size_distribution = cluster_size_distribution

    ## -- Coordination Number Calculation
    def calculate_coordination_numbers(self, pdb_handler, target_atoms):
        """
        Calculates the coordination numbers and their standard deviations for each atom pair type.

        Parameters:
        - pdb_handler: PDBFileHandler object containing all atoms.
        - target_atoms: List of target atoms to calculate coordination numbers for.

        Returns:
        - coordination_stats: Dictionary containing average and standard deviation of coordination numbers for each atom pair.
        """
        coordination_numbers = defaultdict(list)
        
        for atom in target_atoms:
            counts = {neighbor: 0 for neighbor in self.neighbor_elements}
            for other_atom in pdb_handler.core_atoms + pdb_handler.shell_atoms:
                if other_atom.element in self.neighbor_elements:
                    pair = (atom.element, other_atom.element)
                    if pair in self.distance_thresholds:
                        if self.are_connected(atom, other_atom, self.distance_thresholds[pair]):
                            counts[other_atom.element] += 1
            total_coordination = 0
            for neighbor, count in counts.items():
                coordination_numbers[(atom.element, neighbor)].append(count)
                total_coordination += count
        
        # Calculate mean and standard deviation for each atom pair type
        coordination_stats = {}
        for pair, counts in coordination_numbers.items():
            avg = np.mean(counts)
            std = np.std(counts)
            coordination_stats[pair] = (avg, std)

        return coordination_stats, None

    def are_connected(self, atom1, atom2, threshold):
        distance = np.linalg.norm(np.array(atom1.coordinates) - np.array(atom2.coordinates))
        return distance <= threshold

    def print_coordination_numbers(self, coordination_stats_per_size):
        for size, stats in coordination_stats_per_size.items():
            print(f"Cluster Size: {size}")
            total = 0
            for pair, avg_values in stats.items():
                avg = np.mean(avg_values)
                print(f"  {pair[0]} - {pair[1]}: Avg = {avg:.2f}")
                total += avg
            print(f"  Total Coordination Number: {total:.2f}\n")

    ## -- Coordination Number Calculation by Cluster Size
    def calculate_coordination_numbers_for_atom(self, pdb_handler, target_atom, neighbor_elements, distance_thresholds=None):
        """
        Calculate the coordination number for a single atom within a PDB file,
        including tracking which neighbor atoms are coordinated to the target atom.

        Parameters:
        - pdb_handler (PDBFileHandler): The PDB file handler containing atom data.
        - target_atom (Atom): The target atom for which the coordination number is being calculated.
        - neighbor_elements (list): List of neighbor elements to calculate coordination with.
        - distance_thresholds (dict, optional): Dictionary of distance thresholds for atom pairs, e.g., {('Pb', 'I'): 3.6}.

        Returns:
        - coordination_stats (dict): Dictionary containing the coordination stats for the target atom.
        - neighbor_atom_ids (set): Set of neighbor atom IDs coordinated with the target atom.
        - counting_stats (dict): A dictionary tracking how many times each neighbor atom was counted, keyed by atom ID.
        """
        # Initialize coordination number tracking
        coordination_numbers = defaultdict(int)
        counting_stats = defaultdict(lambda: {'count': 0, 'element': None})  # Track counts per neighbor Atom ID
        neighbor_atom_ids = set()  # Track neighbor atom IDs coordinated with the target atom

        for other_atom in pdb_handler.core_atoms + pdb_handler.shell_atoms:
            if other_atom.element in neighbor_elements:
                pair = (target_atom.element, other_atom.element)
                if distance_thresholds and pair in distance_thresholds:
                    threshold = distance_thresholds[pair]
                    if not self.are_connected(target_atom, other_atom, threshold):
                        continue

                # Double counting is always enabled
                coordination_numbers[other_atom.element] += 1
                counting_stats[other_atom.atom_id]['count'] += 1
                counting_stats[other_atom.atom_id]['element'] = other_atom.element
                neighbor_atom_ids.add(other_atom.atom_id)

        total_coordination = sum(coordination_numbers.values())

        # Store coordination stats
        coordination_stats = {}
        for neighbor in neighbor_elements:
            count = coordination_numbers.get(neighbor, 0)
            coordination_stats[(target_atom.element, neighbor)] = (count, 0)  # No std deviation for single atom

        # Return the coordination stats, neighbor atom IDs, and counting stats per neighbor atom
        return coordination_stats, neighbor_atom_ids, counting_stats

    def calculate_coordination_stats_by_subfolder(self, sorted_pdb_folder=None, target_elements=None, neighbor_elements=None, distance_thresholds=None):
        """
        Calculate coordination statistics for all target elements coordinated by neighbor elements in each subfolder,
        track how many times each neighbor atom was counted in each PDB file, and identify sharing patterns.

        Parameters:
        - sorted_pdb_folder (str, optional): Path to the sorted PDB folder. If not provided,
        the method will use the class attribute `self.sorted_pdb_folder`.
        - target_elements (list, required): List of target elements to calculate coordination for.
        - neighbor_elements (list, required): List of neighbor elements to calculate coordination with.
        - distance_thresholds (dict, required): Dictionary with distance thresholds for each atom pair (e.g., {('Pb', 'I'): 3.6}).

        Returns:
        - subfolder_coordination_stats (dict): Dictionary with subfolder names as keys and coordination stats as values.
        - sharing_patterns (dict): Dictionary containing counts of sharing patterns across all PDB files.
        """
        if target_elements is None or neighbor_elements is None or distance_thresholds is None:
            raise ValueError("target_elements, neighbor_elements, and distance_thresholds must all be provided.")

        if sorted_pdb_folder:
            self.sorted_pdb_folder = sorted_pdb_folder
        elif not hasattr(self, 'sorted_pdb_folder') or not os.path.exists(self.sorted_pdb_folder):
            raise ValueError("Sorted PDB folder not found. Please run sort_pdb_files_by_node_count first or provide a sorted path.")

        from collections import defaultdict
        import os
        from tqdm import tqdm

        subfolder_coordination_stats = {}
        self.cluster_coordination_stats = {}  # Initialize or reset the attribute
        self.coordination_details = defaultdict(list)
        self.per_file_neighbor_counts = defaultdict(dict)  # Store per-PDB-file neighbor counts
        self.per_folder_multiplicity_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # Per-folder counts
        self.overall_multiplicity_counts = defaultdict(lambda: defaultdict(int))  # Overall counts across all folders
        sharing_patterns = defaultdict(int)  # Counts of sharing patterns across all PDB files
        self.sharing_pattern_instances = []  # Initialize or reset the attribute to store instance details

        subfolders = os.listdir(self.sorted_pdb_folder)
        overall_progress_bar = tqdm(total=len(subfolders), desc="Processing subfolders", ncols=100)

        for subfolder in subfolders:
            subfolder_path = os.path.join(self.sorted_pdb_folder, subfolder)
            if os.path.isdir(subfolder_path):
                pdb_files = [os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path) if f.endswith('.pdb')]
                subfolder_stats = defaultdict(list)
                cluster_stats = defaultdict(list)  # Collect coordination numbers for this cluster size

                for pdb_file in pdb_files:
                    pdb_handler = PDBFileHandler(pdb_file, core_residue_names=self.core_residue_names,
                                                shell_residue_names=self.shell_residue_names)

                    # Create a mapping from atom IDs to atom objects
                    pdb_handler.atom_id_map = {}
                    for atom in pdb_handler.core_atoms + pdb_handler.shell_atoms:
                        pdb_handler.atom_id_map[atom.atom_id] = atom

                    target_atoms = [atom for atom in pdb_handler.core_atoms if atom.element in target_elements]

                    if target_atoms:
                        file_neighbor_counts = defaultdict(lambda: {'count': 0, 'element': None})
                        neighbor_to_targets = defaultdict(set)  # Mapping from neighbor atom ID to set of target atom IDs
                        target_to_neighbors = defaultdict(set)  # Mapping from target atom ID to set of neighbor atom IDs

                        for target_atom in target_atoms:
                            coordination_stats, neighbor_atom_ids, counting_stats = self.calculate_coordination_numbers_for_atom(
                                pdb_handler, target_atom, neighbor_elements, distance_thresholds
                            )

                            for pair, (coordination_number, _) in coordination_stats.items():
                                subfolder_stats[pair].append(coordination_number)
                                # Collect coordination numbers for cluster stats
                                cluster_stats[pair].append(coordination_number)

                            # Update mappings
                            target_to_neighbors[target_atom.atom_id] = neighbor_atom_ids
                            for neighbor_atom_id in neighbor_atom_ids:
                                neighbor_to_targets[neighbor_atom_id].add(target_atom.atom_id)

                            # Update file_neighbor_counts with counts from this target atom
                            for neighbor_atom_id, stats in counting_stats.items():
                                file_neighbor_counts[neighbor_atom_id]['count'] += stats['count']
                                file_neighbor_counts[neighbor_atom_id]['element'] = stats['element']

                            self.coordination_details[pdb_file].append({
                                'target_atom_id': target_atom.atom_id,
                                'target_atom_element': target_atom.element,
                                'coordination_stats': coordination_stats,
                                'neighbor_atom_ids': neighbor_atom_ids,
                                'counting_stats': counting_stats
                            })

                        # After processing all target atoms, store the per-PDB-file neighbor counts
                        self.per_file_neighbor_counts[pdb_file] = file_neighbor_counts

                        # Aggregate counts per neighbor element and multiplicity for the current PDB file
                        for neighbor_atom_id, stats in file_neighbor_counts.items():
                            element = stats['element']
                            multiplicity = stats['count']
                            # Update per-folder counts
                            self.per_folder_multiplicity_counts[subfolder][element][multiplicity] += 1
                            # Update overall counts
                            self.overall_multiplicity_counts[element][multiplicity] += 1

                        # Identify sharing patterns
                        # Group neighbor atoms by the set of target atoms they are coordinated with
                        neighbor_groups = defaultdict(list)
                        for neighbor_atom_id, target_atom_ids_set in neighbor_to_targets.items():
                            key = frozenset(target_atom_ids_set)
                            neighbor_groups[key].append(neighbor_atom_id)

                        for target_atom_ids_set, neighbor_atom_ids in neighbor_groups.items():
                            num_targets = len(target_atom_ids_set)
                            num_neighbors = len(neighbor_atom_ids)

                            if num_targets > 0:
                                # Get the element names
                                target_elements_involved = set()
                                for target_atom_id in target_atom_ids_set:
                                    target_atom = pdb_handler.atom_id_map[target_atom_id]
                                    target_elements_involved.add(target_atom.element)

                                neighbor_elements_involved = set()
                                for neighbor_atom_id in neighbor_atom_ids:
                                    neighbor_atom = pdb_handler.atom_id_map[neighbor_atom_id]
                                    neighbor_elements_involved.add(neighbor_atom.element)

                                # Assuming only one target element and one neighbor element involved
                                if len(target_elements_involved) == 1 and len(neighbor_elements_involved) == 1:
                                    target_element = next(iter(target_elements_involved))
                                    neighbor_element = next(iter(neighbor_elements_involved))
                                else:
                                    # Handle cases where multiple elements are involved
                                    target_element = ','.join(sorted(target_elements_involved))
                                    neighbor_element = ','.join(sorted(neighbor_elements_involved))

                                pattern = (num_targets, target_element, num_neighbors, neighbor_element)
                                sharing_patterns[pattern] += 1

                                # Record instance details
                                instance = {
                                    'num_targets': num_targets,
                                    'target_atom_ids': list(target_atom_ids_set),
                                    'num_neighbors': num_neighbors,
                                    'neighbor_atom_ids': neighbor_atom_ids,
                                    'target_element': target_element,
                                    'neighbor_element': neighbor_element,
                                    'pattern': pattern,
                                    'pdb_file': pdb_file  # Optional: Include the source PDB file
                                }
                                self.sharing_pattern_instances.append(instance)

                # Store the coordination stats for this subfolder
                subfolder_coordination_stats[subfolder] = subfolder_stats
                self.cluster_coordination_stats[subfolder] = cluster_stats  # Store cluster stats

            overall_progress_bar.update(1)
        overall_progress_bar.close()

        return subfolder_coordination_stats, sharing_patterns

    ## -- Cluster Charge Calculation
    def calculate_cluster_charge(self, pdb_handler):
        """
        Calculates the total charge of the cluster by summing the charges of all atoms.
        
        Parameters:
        - pdb_handler: PDBFileHandler object containing all atoms.
        
        Returns:
        - total_charge: The total charge of the cluster.
        """
        total_charge = 0.0
        
        # Sum the charges for core atoms
        for atom in pdb_handler.core_atoms:
            charge, _ = self.charges.get(atom.element, (0, 0))  # Get the charge part of the tuple, default to 0 if not found
            total_charge += charge
        
        # Sum the charges for neighboring atoms
        for atom in pdb_handler.shell_atoms:
            charge, _ = self.charges.get(atom.element, (0, 0))  # Get the charge part of the tuple, default to 0 if not found
            total_charge += charge
        
        # print(f"Total charge of the cluster: {total_charge}")
        return total_charge

    ## -- Volume Calculation Methods -- ##
    # - Volume Method 0: Bulk Solvent Volume Calculation
    def _initialize_bulk_volume(self):
        if self.bulk_volume_params is None:
            raise ValueError(
                "bulk_volume_params must be provided when volume_method is 'bulk_volume'."
            )
        
        # Validate that all required keys are present
        required_keys = [
            'mass_percent_solute',
            'density_solution',
            'density_neat_solvent',
            'molar_mass_solvent',
            'molar_mass_solute',
            'ionic_radii',
            'stoichiometry',
            'atomic_masses',
            'solute_residues',
            'solvent_name',
            # 'electrons_info' is optional and handled separately
        ]
        
        missing_keys = [key for key in required_keys if key not in self.bulk_volume_params]
        if missing_keys:
            raise KeyError(f"Missing keys in bulk_volume_params: {missing_keys}")

        # Instantiate BulkVolume with the provided parameters, including electrons_info if present
        self.bulk_volume = BulkVolume(**self.bulk_volume_params)
        print("BulkVolume has been successfully initialized.")

        # Estimate volumes
        self.bulk_volume.estimate_volumes()
        print("Volumes estimated and stored as .volumes attribute in BulkVolume class instance.")

        # If electrons_info is provided, add electrons per unit
        if self.bulk_volume.electrons_info is not None:
            self.bulk_volume.add_electrons_per_unit()
            print("Electrons per unit have been added to volumes.")
        else:
            print("No electrons_info provided; skipping add_electrons_per_unit().")

        # Calculate solution electron density
        try:
            self.bulk_volume.calculate_solution_electron_density()
            print(f"Calculated solution electron density: {self.bulk_volume.solution_electron_density_A3:.4e} electrons/Å³")
        except ValueError as e:
            print(f"Error calculating solution electron density: {e}")

    def calculate_pdb_volume(self, pdb_handler: PDBFileHandler) -> float:
        """
        Calculates the total volume of the PDB structure using BulkVolume and PDBFileHandler.
        
        Parameters:
        - pdb_handler (PDBFileHandler): An instance of PDBFileHandler with parsed PDB data.
        
        Returns:
        - total_volume (float): Total estimated volume in cubic angstroms (Å³).
        """
        if self.volume_method != 'bulk_volume':
            raise ValueError("Volume estimation method is not set to 'bulk_volume'.")

        if self.bulk_volume is None:
            raise ValueError("BulkVolume instance is not initialized.")

        # Generate the volume lookup table if not already done
        if not self.bulk_volume.volumes:
            self.bulk_volume.generate_volume_lookup(pdb_handler)
            raise ValueError("BulkVolume does not contain a valid .volumes attribute.")

        total_volume = 0.0

        # 1. Calculate Solvent Volume
        solvent_count = pdb_handler.count_solvent_molecules()
        solvent_name = self.bulk_volume.solvent_name  # e.g., 'DMS'
        solvent_info = self.bulk_volume.volumes.get(solvent_name, {}).get('Solvent', {})
        solvent_volume_per_molecule = solvent_info.get('Volume per Molecule', 0.0)
        solvent_total_volume = solvent_volume_per_molecule * solvent_count
        total_volume += solvent_total_volume
        print(f"Solvent Count: {solvent_count}, Volume per Molecule: {solvent_volume_per_molecule:.3f} Å³, "
              f"Total Solvent Volume: {solvent_total_volume:.3f} Å³")

        # 2. Calculate Solute Molecules Volume (if applicable)
        # If solute molecules have 'Volume per Molecule' in lookup, handle them here
        # This part is optional based on your BulkVolume setup
        # For demonstration, we'll assume solute molecules do not have a separate volume

        # 3. Calculate Solute Atoms Volume
        solute_atom_counts = pdb_handler.count_solute_atoms()
        solute_residue_names = set(self.bulk_volume.solute_residues.values())

        for solute_residue in solute_residue_names:
            for element, count in solute_atom_counts.items():
                atom_info = self.bulk_volume.volumes.get(solute_residue, {}).get(element, {})
                volume_per_atom = atom_info.get('Volume per Atom', 0.0)
                if volume_per_atom > 0.0:
                    atom_total_volume = volume_per_atom * count
                    total_volume += atom_total_volume
                    print(f"Solute Atom Type: {element}, Count: {count}, Volume per Atom: {volume_per_atom:.3f} Å³, "
                          f"Total Atom Volume: {atom_total_volume:.3f} Å³")
                else:
                    print(f"Warning: Volume information for solute atom '{element}' not found in lookup.")

        print(f"Total Estimated Volume: {total_volume:.3f} Å³")
        return total_volume

    def calculate_pdb_volume_and_electron_density(self, pdb_handler: PDBFileHandler) -> (float, float, float):
        """
        Calculates the total volume and electron density of the PDB structure using BulkVolume and PDBFileHandler.
        Also calculates the difference between the solution electron density and the PDB electron density (delta_rho).
        
        Parameters:
        - pdb_handler (PDBFileHandler): An instance of PDBFileHandler with parsed PDB data.
        
        Returns:
        - total_volume (float): Total estimated volume in cubic angstroms (Å³).
        - electron_density (float): Electron density of the PDB structure in electrons per cubic angstrom (electrons/Å³).
        - delta_rho (float): Difference between the solution electron density and the PDB electron density (electrons/Å³).
        """
        if self.volume_method != 'bulk_volume':
            raise ValueError("Volume estimation method is not set to 'bulk_volume'.")

        if self.bulk_volume is None:
            raise ValueError("BulkVolume instance is not initialized.")

        # Ensure volumes are estimated
        if not hasattr(self.bulk_volume, 'volumes') or not self.bulk_volume.volumes:
            # Estimate volumes and populate self.bulk_volume.volumes
            self.bulk_volume.estimate_volumes()
            if not self.bulk_volume.volumes:
                raise ValueError("BulkVolume does not contain a valid .volumes attribute.")

        # Ensure electrons_info is provided
        if self.bulk_volume.electrons_info is None:
            raise ValueError("Electrons information (electrons_info) must be provided before calculating electron density.")

        # Add electrons per unit if not already added
        if not any(
            'Electrons per Atom' in comp or 'Electrons per Molecule' in comp
            for res in self.bulk_volume.volumes.values()
            for comp in res.values()
        ):
            self.bulk_volume.add_electrons_per_unit()
            print("Electrons per unit have been added to volumes.")

        # Ensure solution electron density is calculated
        if self.bulk_volume.solution_electron_density_A3 is None:
            self.bulk_volume.calculate_solution_electron_density()
            if self.bulk_volume.solution_electron_density_A3 is None:
                raise ValueError("Failed to calculate solution electron density.")

        total_volume = 0.0
        total_electrons = 0.0

        # 1. Calculate Solvent Volume and Electrons
        solvent_count = pdb_handler.count_solvent_molecules()
        solvent_name = self.bulk_volume.solvent_name  # e.g., 'DMS'
        solvent_info = self.bulk_volume.volumes.get(solvent_name, {}).get('Solvent', {})
        solvent_volume_per_molecule = solvent_info.get('Volume per Molecule', 0.0)
        electrons_per_molecule = solvent_info.get('Electrons per Molecule', 0)
        if solvent_volume_per_molecule > 0 and electrons_per_molecule > 0:
            solvent_total_volume = solvent_volume_per_molecule * solvent_count
            solvent_total_electrons = electrons_per_molecule * solvent_count
            total_volume += solvent_total_volume
            total_electrons += solvent_total_electrons
            print(f"Solvent Count: {solvent_count}, Volume per Molecule: {solvent_volume_per_molecule:.3f} Å³, "
                f"Total Solvent Volume: {solvent_total_volume:.3f} Å³, "
                f"Electrons per Molecule: {electrons_per_molecule}, "
                f"Total Solvent Electrons: {solvent_total_electrons}")
        else:
            raise ValueError("Missing volume or electron data for solvent.")

        # 2. Calculate Solute Atoms Volume and Electrons
        solute_atom_counts = pdb_handler.count_solute_atoms()
        solute_residue_names = set(self.bulk_volume.solute_residues.values())

        for solute_residue in solute_residue_names:
            for element, count in solute_atom_counts.items():
                # Map element to the format used in BulkVolume, if necessary
                mapped_element = element
                atom_info = self.bulk_volume.volumes.get(solute_residue, {}).get(mapped_element, {})
                volume_per_atom = atom_info.get('Volume per Atom', 0.0)
                electrons_per_atom = atom_info.get('Electrons per Atom', 0)
                if volume_per_atom > 0.0 and electrons_per_atom > 0:
                    atom_total_volume = volume_per_atom * count
                    atom_total_electrons = electrons_per_atom * count
                    total_volume += atom_total_volume
                    total_electrons += atom_total_electrons
                    print(f"Solute Atom Type: {mapped_element}, Count: {count}, Volume per Atom: {volume_per_atom:.3f} Å³, "
                        f"Total Atom Volume: {atom_total_volume:.3f} Å³, "
                        f"Electrons per Atom: {electrons_per_atom}, "
                        f"Total Atom Electrons: {atom_total_electrons}")
                else:
                    raise ValueError(f"Volume or electron information for solute atom '{mapped_element}' not found in lookup.")

        print(f"\nTotal Estimated Volume: {total_volume:.3f} Å³")
        print(f"Total Electrons: {total_electrons}")

        # Calculate electron density
        if total_volume > 0:
            electron_density = total_electrons / total_volume  # electrons/Å³
            print(f"Electron Density of the PDB Structure: {electron_density:.4e} electrons/Å³")
        else:
            raise ValueError("Total volume is zero. Cannot calculate electron density.")

        # Calculate delta_rho
        solution_electron_density = self.bulk_volume.solution_electron_density_A3  # electrons/Å³
        delta_rho = electron_density - solution_electron_density
        print(f"Solution Electron Density: {solution_electron_density:.4e} electrons/Å³")
        print(f"Electron Density Contrast (Delta Rho): {delta_rho:.4e} electrons/Å³")

        return total_volume, electron_density, delta_rho

    # - Volume Method 1: Calculate Radius of Gyration for Volume Method
    def calculate_radius_of_gyration(self, atom_positions, electron_counts):
        """
        Calculate the radius of gyration (Rg) for a cluster of atoms.

        Parameters:
        - atom_positions: List of (x, y, z) tuples representing atomic positions.
        - electron_counts: List of electron counts corresponding to the atomic positions.

        Returns:
        - radius_of_gyration: The calculated radius of gyration.
        """
        # Calculate the center of mass using electron counts as weights
        electron_counts = np.array(electron_counts)
        positions = np.array(atom_positions)
        center_of_mass = np.average(positions, axis=0, weights=electron_counts)

        # Calculate the radius of gyration
        squared_distances = np.sum(electron_counts * np.sum((positions - center_of_mass) ** 2, axis=1))
        total_electrons = np.sum(electron_counts)
        radius_of_gyration = np.sqrt(squared_distances / total_electrons)

        return radius_of_gyration

    def get_electron_counts(self, atoms):
        """
        Calculate the electron counts for each atom based on the element and formal charge.

        Parameters:
        - atoms: List of atom objects with element and charge information.

        Returns:
        - electron_counts: List of electron counts for each atom.
        """
        electron_counts = []
        for atom in atoms:
            elem_info = element(atom.element)
            total_electrons = elem_info.electrons - self.charges.get(atom.element, (0, 0))[0]
            electron_counts.append(total_electrons)
        return electron_counts

    def calculate_volume_using_rg(self, pdb_handler, shape_type='sphere'):
        """
        Calculate the volume of the cluster using the radius of gyration.
        
        :param pdb_handler: The PDBFileHandler object for the current PDB file.
        :param shape_type: 'sphere' or 'ellipsoid' to choose the volume calculation method.
        :return: The calculated volume.
        """
        # Load data into the RadiusOfGyrationCalculator
        self.rg_calculator.load_from_pdb(pdb_handler, self.charges)
        
        # Calculate volume based on the specified shape type
        if shape_type == 'sphere':
            return self.rg_calculator.calculate_volume(method='sphere')
        elif shape_type == 'ellipsoid':
            return self.rg_calculator.calculate_volume(method='ellipsoid')
        else:
            raise ValueError(f"Unknown shape type: {shape_type}")

    def estimate_volume_using_rg(self, pdb_handler):
        """
        Estimate the molecular volume using the radius of gyration (Rg).

        Parameters:
        - pdb_handler: PDBFileHandler object containing all atoms.

        Returns:
        - volume: Estimated volume based on Rg in cubic angstroms.
        """
        atom_positions = [atom.coordinates for atom in pdb_handler.core_atoms + pdb_handler.shell_atoms]
        electron_counts = self.get_electron_counts(pdb_handler.core_atoms + pdb_handler.shell_atoms)

        rg = self.calculate_radius_of_gyration(atom_positions, electron_counts)
        volume = (4/3) * np.pi * (rg ** 3)  # Volume estimation using Rg

        return volume
    
    # - Volume Method 2: Ionic Radius Volume Approximation Method
    # Note: Add a method to base the ionic radius dynamically on the coordination for select atoms.
    def build_ionic_radius_lookup(self):
        """
        Builds a lookup table for ionic radii, electron count, and volume of ions as spheres.
        Uses the self.charges dictionary provided during class initialization, which now includes both charge and coordination number.
        """
        def to_roman(n):
            roman_numerals = {
                1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V',
                6: 'VI', 7: 'VII', 8: 'VIII', 9: 'IX', 10: 'X',
                11: 'XI', 12: 'XII', 13: 'XIII', 14: 'XIV', 15: 'XV'
            }
            return roman_numerals.get(n, None)

        radius_lookup = {}
        radii_list = []

        for atom_type, (charge, coordination) in self.charges.items():
            # Generate a unique key based on element and charge
            key = (atom_type, charge)
            
            print(f"Looking up ionic radius for {atom_type} with charge {charge} and coordination {coordination}...")

            # Convert numeric coordination number to Roman numeral
            roman_coordination = to_roman(coordination)
            if not roman_coordination:
                print(f"Warning: Invalid coordination number {coordination} for {atom_type}.")
                continue

            if key not in radius_lookup:
                # Retrieve the element information
                elem = element(atom_type)
                print(f"Element data retrieved for {atom_type}: {elem}")
                
                # Debug: Print available ionic radii data
                print(f"Available ionic radii for {atom_type}:")
                for ir in elem.ionic_radii:
                    print(f"  Charge: {ir.charge}, Coordination: {ir.coordination}, Radius: {ir.ionic_radius} pm")
                
                # Get the ionic radii for the specified charge and Roman numeral coordination number
                matching_radius = next(
                    (ir for ir in elem.ionic_radii
                    if ir.charge == charge and ir.coordination == roman_coordination), None
                )
                
                if matching_radius:
                    radius = matching_radius.ionic_radius / 100.0  # Convert pm to Å
                    volume = (4/3) * np.pi * (radius ** 3)  # Calculate the volume of the sphere
                    radius_lookup[key] = {
                        'ionic_radius': radius,
                        'volume': volume  # Store the volume
                    }
                    radii_list.append(radius)
                    print(f"Radius found for {atom_type} with charge {charge} and coordination {roman_coordination}: {radius} Å")
                else:
                    print(f"No ionic radius found for {atom_type} with charge {charge} and coordination {roman_coordination}. Trying covalent radius.")
                    # Fallback to covalent radius
                    covalent_radius = elem.covalent_radius / 100.0  # Convert pm to Å
                    if covalent_radius:
                        volume = (4/3) * np.pi * (covalent_radius ** 3)
                        radius_lookup[key] = {
                            'ionic_radius': covalent_radius,
                            'volume': volume
                        }
                        print(f"Using covalent radius for {atom_type}: {covalent_radius} Å")
                    else:
                        print(f"Warning: No radius found for {atom_type} with charge {charge}.")
                        radius_lookup[key] = {
                            'ionic_radius': None,
                            'volume': None
                        }
                        radii_list.append(None)

        return radius_lookup, np.array(radii_list, dtype=np.float64)

    def estimate_total_molecular_volume(self, pdb_handler):
        """
        Estimates the total molecular volume by summing the volumes of spheres corresponding to each ionic radius.
        Weights the volume by the count of each element.
        
        Parameters:
        - pdb_handler: PDBFileHandler object containing all atoms.
        
        Returns:
        - total_volume: Estimated total molecular volume in cubic angstroms.
        """
        element_counts = defaultdict(int)
        
        all_atoms = pdb_handler.core_atoms + pdb_handler.shell_atoms  # Combine core and shell atoms
        
        # print(f"Calculating volume for PDB file: {pdb_handler.filepath}")
        # print(f"Total atoms found (core + shell): {len(all_atoms)}")
        
        # Count the occurrences of each element type in the PDB file
        for atom in all_atoms:
            key = (atom.element, self.charges.get(atom.element, (0, 0))[0])  # Use provided charges and default to (0, 0) if not found
            element_counts[key] += 1
        
        # # Output the elements and their counts
        # for key, count in element_counts.items():
            # print(f"Element: {key[0]}, Charge: {key[1]}, Count: {count}")
        
        # Calculate the total volume
        total_volume = 0.0
        for key, count in element_counts.items():
            if key in self.radius_lookup:
                if self.radius_lookup[key]['volume'] is not None:
                    weighted_volume = count * self.radius_lookup[key]['volume']
                    total_volume += weighted_volume
                    # print(f"Adding {count} * {self.radius_lookup[key]['volume']} for {key[0]} to total volume.")
                else:
                    print(f"Warning: Volume for {key[0]} with charge {key[1]} is None.")
            else:
                print(f"Warning: No radius found for {key[0]} with charge {key[1]} in lookup table.")
        
        # print(f"Total atoms used in volume calculation: {sum(element_counts.values())}")
        # print(f"Calculated total volume: {total_volume} cubic angstroms\n")
        
        return total_volume
    
    ## -- SAXS Calculations
    def calculate_total_iq(self, q_values, shape_type='sphere'):
        """
        Calculate the total scattering intensity I(q) for the polydisperse distribution of clusters.
        This is done as a weighted average of I(q) values, where the weights are the number of clusters of each size.

        Parameters:
        - q_values: A numpy array of q-values in inverse angstroms.
        - shape_type: 'sphere' or 'ellipsoid' to choose the volume calculation method.
        
        Returns:
        - total_iq: A numpy array of weighted average I(q) values.
        """
        total_iq = np.zeros_like(q_values)
        total_clusters = 0
        
        for data in self.cluster_data:
            # Retrieve the volume and scattering dimensions based on the cluster shape
            if shape_type == 'sphere':
                volume = data['volume']
                sphere_scattering = SphereScattering(volume=volume)
                iq_values = sphere_scattering.calculate_iq(q_values)
            elif shape_type == 'ellipsoid':
                Rgx = data['Rgx']
                Rgy = data['Rgy']
                Rgz = data['Rgz']
                ellipsoid_scattering = EllipsoidScattering(a=Rgx, b=Rgy, c=Rgz)
                iq_values = ellipsoid_scattering.calculate_iq(q_values)
            else:
                raise ValueError(f"Unknown shape type: {shape_type}")

            # Weight I(q) by the number of clusters of this size
            num_clusters = len(self.cluster_size_distribution[data['cluster_size']])
            weighted_iq = iq_values * num_clusters
            # Add to the total I(q)
            total_iq += weighted_iq
            # Accumulate the total number of clusters
            total_clusters += num_clusters

        # Normalize by the total number of clusters to get the weighted average
        total_iq /= total_clusters
        
        return total_iq

    def plot_total_iq(self, q_values):
        """
        Plot the total I(q) vs. q on a log-log scale.
        
        Parameters:
        - q_values: A numpy array of q-values in inverse angstroms.
        """
        total_iq = self.calculate_total_iq(q_values)
        
        # Create the plot
        plt.figure(figsize=(8, 6))
        plt.loglog(q_values, total_iq, marker='o', linestyle='-', color='r')
        plt.xlabel('q (Å⁻¹)')
        plt.ylabel('I(q)')
        plt.title('Total Scattering Intensity I(q) vs. Scattering Vector q')
        plt.grid(True, which="both", ls="--")
        plt.show()

    def save_total_iq(self, q_values, sample_name="sample"):
        """
        Save the total I(q) vs. q data to a .txt file.
        
        Parameters:
        - q_values: A numpy array of q-values in inverse angstroms.
        - sample_name: A string prefix for the filename.
        """
        total_iq = self.calculate_total_iq(q_values)
        
        # Get the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create the filename with the sample name and timestamp
        filename = f"{sample_name}_IQ_{timestamp}.txt"
        
        # Save the data to the file
        data = np.column_stack((q_values, total_iq))
        np.savetxt(filename, data, header="q (Å⁻¹)\tI(q)", fmt="%.6e", delimiter="\t")
        
        print(f"Total I(q) saved to {filename}")
        
    ## -- Plotting Methods
    @staticmethod
    def custom_glossy_marker(ax, x, y, base_color, markersize=8, offset=(0.08, 0.08)):
        for (xi, yi) in zip(x, y):
            # Draw the base marker
            ax.plot(xi, yi, 'o', markersize=markersize, color=base_color, zorder=1)

            gloss_params = [
                (markersize * 0.008, 0.3),  # Largest circle, more transparent
                (markersize * 0.005, 0.6),  # Middle circle, less transparent
                (markersize * 0.002, 1.0)   # Smallest circle, no transparency
                ]
            # # Offset for the glossy effect
            # x_offset, y_offset = offset
        
            x_offset = markersize/20 * offset[0]
            y_offset = markersize/20 * offset[1]

            # Overlay glossy effect - smaller concentric circles as highlights
            for i, (size, alpha) in enumerate(gloss_params):
                circle = plt.Circle((xi - x_offset, yi + y_offset), size, color='white', alpha=alpha, transform=ax.transData, zorder=2+i)
                ax.add_patch(circle)

    def plot_cluster_size_distribution(self, all_cluster_sizes):
        unique_sizes, counts = np.unique(all_cluster_sizes, return_counts=True)

        plt.figure(figsize=(8, 6))
        plt.bar(unique_sizes, counts, color='blue', edgecolor='black')
        plt.xlabel(f'Cluster Size ({self.target_elements[0]} Atom Count)', fontsize = 14)
        plt.ylabel('Number of Clusters', fontsize = 14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.title('Histogram of Cluster Sizes')
        plt.grid(True)
        plt.show()

    def plot_coordination_histogram(self, coordination_stats_per_size, title=None):
        sizes = sorted(coordination_stats_per_size.keys())
        pairs = set(pair for data in coordination_stats_per_size.values() for pair in data.keys())

        # Calculate mean coordination numbers and standard deviations
        coord_data = {
            pair: [np.mean(coordination_stats_per_size[size].get(pair, [0])) for size in sizes] for pair in pairs
        }
        coord_stds = {
            pair: [np.std(coordination_stats_per_size[size].get(pair, [0])) for size in sizes] for pair in pairs
        }
        cluster_counts = [len(coordination_stats_per_size[size][next(iter(pairs))]) for size in sizes]

        # Calculate weighted averages and standard deviations for legend labels
        weighted_avgs = {}
        weighted_stds = {}
        for pair in pairs:
            weighted_sum = sum(coord_data[pair][i] * cluster_counts[i] for i in range(len(sizes)))
            total_clusters = sum(cluster_counts)
            weighted_avg = weighted_sum / total_clusters if total_clusters > 0 else 0
            weighted_avgs[pair] = weighted_avg
            weighted_stds[pair] = np.mean(coord_stds[pair])  # Simple average of standard deviations for the legend

        # Default title if not provided
        if title is None:
            title = f'{self.target_elements[0]} Coordination Number vs. Cluster Size'

        plt.figure(figsize=(10, 6))

        bottom = np.zeros(len(sizes))

        for pair in pairs:
            neighbor_element = pair[1]
            if neighbor_element == 'O':
                color = (1.0, 0, 0, 0.7)  # red with 70% transparency
            elif neighbor_element == 'I':
                color = (0.3, 0, 0.3, 0.7)  # purple with 70% transparency
            elif neighbor_element == 'S':
                color = (0.545, 0.545, 0, 0.7)  # dark yellow with 70% transparency
            else:
                color = (0.5, 0.5, 0.5, 0.7)  # gray as a fallback

            coord_values = np.array(coord_data[pair])
            std_values = np.array(coord_stds[pair])

            # Plot bars
            plt.bar(sizes, coord_values, bottom=bottom, color=color, edgecolor='black', linewidth=1,
                    label=f"{pair[0]} - {pair[1]} , CN: {weighted_avgs[pair]:.2f} ± {weighted_stds[pair]:.2f}")
            
            # Add error bars for standard deviations centered on the top of each box
            plt.errorbar(sizes, bottom + coord_values, yerr=std_values, fmt='none', ecolor='black', capsize=5)

            bottom += coord_values

        plt.axhline(y=5, color='gray', linestyle='--')  # Dashed line at y = 5
        plt.ylim(0, 6)  # Increase y-axis bound to 6
        
        # Increase font sizes
        plt.xlabel(f'Cluster Size ({self.target_elements[0]} Atom Count)', fontsize=14)
        plt.ylabel(f'{self.target_elements[0]} Coordination Number', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        # Place legend in a box on the right
        plt.legend(frameon=True, fontsize=12, loc='upper right', bbox_to_anchor=(1, 1), edgecolor='black')

        plt.title(title, fontsize=16)
        plt.show()

    def plot_average_volume_vs_cluster_size(self):
        """
        Plots the average volume of clusters versus the cluster size with error bars representing the standard deviation.
        Uses custom glossy markers for the data points.
        """
        # Ensure the statistics are calculated
        if not hasattr(self, 'average_volumes_per_size'):
            self.generate_statistics()

        sizes = sorted(self.average_volumes_per_size.keys())
        avg_volumes = [self.average_volumes_per_size[size] for size in sizes]
        std_devs = [np.std(self.cluster_size_distribution[size]) for size in sizes]  # Calculate standard deviations

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the error bars and the data points
        plt.errorbar(sizes, avg_volumes, yerr=std_devs, fmt='o-', color='blue', ecolor='black', capsize=5, label='Average Volume')
        plt.xlabel(f'Cluster Size ({self.target_elements[0]} Atom Count)', fontsize=14)
        plt.ylabel(r'$<V_{\mathrm{cluster}}> \ (\mathrm{\AA}^{3})$', fontsize=14)
        plt.title('Average Cluster Volume vs Cluster Size', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.legend(['Average Volume'], fontsize=12)
        plt.show()

    def plot_average_volume_vs_cluster_size_rg(self):
        """
        Plots the average volume of clusters versus the cluster size with error bars representing the standard deviation.
        This version of the plot specifically indicates that the volumes are calculated using the radius of gyration (R_g).
        """
        # Ensure the statistics are calculated
        if not hasattr(self, 'average_volumes_per_size'):
            self.generate_statistics()

        sizes = sorted(self.average_volumes_per_size.keys())
        avg_volumes = [self.average_volumes_per_size[size] for size in sizes]
        std_devs = [np.std(self.cluster_size_distribution[size]) for size in sizes]  # Calculate standard deviations

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the error bars and the data points
        plt.errorbar(sizes, avg_volumes, yerr=std_devs, fmt='o-', color='blue', ecolor='black', capsize=5, label='Average Volume')
        plt.xlabel(f'Cluster Size ({self.target_elements[0]} Atom Count)', fontsize=14)
        plt.ylabel(r'$<V_{\mathrm{cluster}}> \ (\mathrm{\AA}^{3})$', fontsize=14)
        plt.title('Average Cluster Volume vs Cluster Size (Based on $R_g$)', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.legend(['Average Volume (Based on $R_g$)'], fontsize=12)
        plt.show()

    def plot_volume_percentage_of_scatterers(self, box_size_angstroms, num_boxes):
        """
        Plots the volume percentage of scatterers for each cluster size using the average cluster volume
        divided by the total volume of all clusters. Highlights the mode bar with a complementary color.
        The calculations for the median and mean cluster sizes are included but the labels are commented out.

        Parameters:
        - box_size_angstroms: float, the size of the box in angstroms.
        - num_boxes: int, the number of boxes containing clusters.
        """
        # Calculate the total volume of all clusters
        total_cluster_volume = 0.0
        for size, volumes in self.cluster_size_distribution.items():
            avg_volume = np.mean(volumes)
            total_cluster_volume += avg_volume * len(volumes)

        # Calculate the volume percentage for each cluster size
        sizes = sorted(self.cluster_size_distribution.keys())
        volume_percentages = []

        for size in sizes:
            if len(self.cluster_size_distribution[size]) > 0:
                avg_volume = np.mean(self.cluster_size_distribution[size])
                volume_percentage = (avg_volume * len(self.cluster_size_distribution[size]) / total_cluster_volume) * 100
                volume_percentages.append(volume_percentage)
            else:
                volume_percentages.append(0)

        # Calculate the mode of the volume percentage distribution
        mode_index = np.argmax(volume_percentages)
        mode_size = sizes[mode_index]
        mode_percentage = volume_percentages[mode_index]

        # Calculate the weighted median and mean of the cluster sizes
        weighted_cluster_sizes = []
        for size, percentage in zip(sizes, volume_percentages):
            weighted_cluster_sizes.extend([size] * int(percentage * 100))  # Weight by percentage

        median_size = np.median(weighted_cluster_sizes) if weighted_cluster_sizes else 0
        mean_size = np.mean(weighted_cluster_sizes) if weighted_cluster_sizes else 0

        # Plotting the volume percentage histogram
        plt.figure(figsize=(10, 6))
        
        # Highlight the mode cluster size bar with a complementary color
        bar_colors = ['green'] * len(sizes)
        bar_colors[mode_index] = 'orange'  # Highlight the mode bar

        plt.bar(sizes, volume_percentages, color=bar_colors, edgecolor='black')
        plt.xlabel(f'Cluster Size ({self.target_elements[0]} Atom Count)', fontsize=14)
        
        # Update the ylabel to the correct format
        # plt.ylabel(r'$\phi \times <V_{\mathrm{c}}> \ (\mathrm{\AA}^{3})$', fontsize=14)
        plt.ylabel(r'$\phi$ (Volume %)', fontsize=14)
        
        plt.title(f'% Scattering Contribution vs Cluster Size ({self.target_elements[0]} Atom Count)', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)

        # Display the mode of the cluster size contributing to scattering
        mode_cluster_volume = np.mean(self.cluster_size_distribution[mode_size])
        mode_cluster_count = len(self.cluster_size_distribution[mode_size])
        mode_volume_percentage = (mode_cluster_volume * mode_cluster_count / total_cluster_volume) * 100

        annotation_text = (
            f'Mode: Cluster Size = {mode_size}, Total Cluster Volume % = {mode_volume_percentage:.2f}%' #\n'
            # f'Median Cluster Size = {median_size:.2f}\n'  # Commented out
            # f'Mean Cluster Size = {mean_size:.2f}'  # Commented out
        )
        plt.annotate(annotation_text, xy=(0.98, 0.98), xycoords='axes fraction', ha='right', va='top', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

        plt.show()

    def plot_charge_vs_cluster_size(self):
        """
        Plots the average charge per cluster versus the cluster size.
        """
        sizes = [data['cluster_size'] for data in self.cluster_data]
        charges = [data['charge'] for data in self.cluster_data]

        plt.figure(figsize=(10, 6))
        plt.scatter(sizes, charges, color='blue', label='Charge/Cluster')
        plt.xlabel(f'Cluster Size ({self.target_elements[0]} Count)', fontsize=14)
        plt.ylabel(r'<Charge>/Cluster (e)', fontsize=14)
        plt.title('Average Charge per Cluster vs Cluster Size', fontsize=16)
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.show()

    # def plot_coordination_histogram(self, coordination_stats_per_size, title=None):
    #     sizes = sorted(coordination_stats_per_size.keys())
    #     pairs = set(pair for data in coordination_stats_per_size.values() for pair in data.keys())

    #     coord_data = {pair: [np.mean(coordination_stats_per_size[size].get(pair, [0])) for size in sizes] for pair in pairs}
    #     cluster_counts = [len(coordination_stats_per_size[size][next(iter(pairs))]) for size in sizes]

    #     # Calculate weighted averages
    #     weighted_avgs = {}
    #     for pair in pairs:
    #         weighted_sum = sum(coord_data[pair][i] * cluster_counts[i] for i in range(len(sizes)))
    #         total_clusters = sum(cluster_counts)
    #         weighted_avgs[pair] = weighted_sum / total_clusters

    #     # Default title if not provided
    #     if title is None:
    #         title = f'{self.target_elements[0]} Coordination Number v. Cluster Size'

    #     plt.figure(figsize=(10, 6))

    #     bottom = np.zeros(len(sizes))

    #     for pair in pairs:
    #         neighbor_element = pair[1]
    #         if neighbor_element == 'O':
    #             color = (1.0, 0, 0, 0.7)  # red with 50% transparency
    #         elif neighbor_element == 'I':
    #             color = (0.3, 0, 0.3, 0.7)  # purple with 50% transparency
    #         elif neighbor_element == 'S':
    #             color = (0.545, 0.545, 0, 0.7)  # dark yellow with 50% transparency
    #         else:
    #             color = (0.5, 0.5, 0.5, 0.7)  # gray as a fallback

    #         coord_values = np.array(coord_data[pair])
    #         plt.bar(sizes, coord_values, bottom=bottom, color=color, edgecolor='black', linewidth=1, label=f"{pair[0]} - {pair[1]}")
    #         bottom += coord_values

    #     plt.axhline(y=5, color='gray', linestyle='--')  # Dashed line at y = 5
    #     plt.ylim(0, 6.5)  # Increase y-axis bound to 6
        
    #     # Increase font sizes
    #     plt.xlabel(f'Cluster Size ({self.target_elements[0]} Atom Count)', fontsize=14)
    #     plt.ylabel(f'{self.target_elements[0]} Coordination Number', fontsize=14)
    #     plt.xticks(fontsize=12)
    #     plt.yticks(fontsize=12)
        
    #     # Place legend in a box on the right
    #     plt.legend(frameon=True, fontsize=12, loc='upper right', bbox_to_anchor=(1, 1), edgecolor='black')

    #     # Add annotation with weighted averages in the top-left
    #     annotation_text = 'Average Coordination Numbers:\n' + '\n'.join([f"{pair[0]} - {pair[1]}: {weighted_avgs[pair]:.2f}" for pair in pairs])
    #     plt.annotate(annotation_text, xy=(0.02, 0.98), xycoords='axes fraction', ha='left', va='top', fontsize=12,
    #                 bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    #     plt.title(title, fontsize=16)
    #     plt.show()

    def plot_phi_Vc_vs_cluster_size(self, box_size_angstroms=None, num_boxes=None):
        """
        Plots the product of the volume fraction and the average cluster volume for each cluster size.
        The y-axis represents phi * <V_c>, where phi is the volume fraction and <V_c> is the average cluster volume.
        
        Parameters:
        - box_size_angstroms: float, the size of the box in angstroms.
        - num_boxes: int, the number of boxes containing clusters.
        """
        # Calculate the total volume of all clusters
        total_cluster_volume = 0.0
        for size, volumes in self.cluster_size_distribution.items():
            avg_volume = np.mean(volumes)
            total_cluster_volume += avg_volume * len(volumes)

        # Calculate the product of volume fraction and average volume for each cluster size
        sizes = sorted(self.cluster_size_distribution.keys())
        phi_Vc_values = []

        for size in sizes:
            if len(self.cluster_size_distribution[size]) > 0:
                avg_volume = np.mean(self.cluster_size_distribution[size])
                volume_fraction = (avg_volume * len(self.cluster_size_distribution[size]) / total_cluster_volume)
                phi_Vc = avg_volume * volume_fraction
                phi_Vc_values.append(phi_Vc)
            else:
                phi_Vc_values.append(0)

        # Calculate the mode of the phi * <V_c> distribution
        mode_index = np.argmax(phi_Vc_values)
        mode_size = sizes[mode_index]
        mode_phi_Vc = phi_Vc_values[mode_index]

        # Plotting the phi * <V_c> histogram
        plt.figure(figsize=(10, 6))
        
        # Highlight the mode cluster size bar with a complementary color
        bar_colors = ['green'] * len(sizes)
        bar_colors[mode_index] = 'orange'  # Highlight the mode bar

        plt.bar(sizes, phi_Vc_values, color=bar_colors, edgecolor='black')
        plt.xlabel(f'Cluster Size ({self.target_elements[0]} Atom Count)', fontsize=14)
        
        # Set the ylabel to the combined format
        plt.ylabel(r'$\phi \times <V_{\mathrm{c}}> \ (\mathrm{\AA}^{3})$', fontsize=14)
        
        plt.title(r'$\phi \times <V_{\mathrm{c}}>$ vs Cluster Size', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)

        # Display the mode of the cluster size contributing to phi * <V_c>
        annotation_text = (
            f'Mode: Cluster Size = {mode_size}' #, phi * <V_c> = {mode_phi_Vc:.2f} $\mathrm{{\AA}}^3$'
        )
        plt.annotate(annotation_text, xy=(0.98, 0.98), xycoords='axes fraction', ha='right', va='top', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

        plt.show()

    ## -- Batch Sorted Plotting Methods
    # Average Coordination Number by Cluster Size and Neighboring Element
    def plot_average_coordination_numbers(self):
        """
        Plots the average coordination numbers for each cluster size,
        showing contributions from different neighbor elements.
        """
        # Prepare data
        data = []
        for subfolder, stats in self.cluster_coordination_stats.items():
            # Extract cluster size from subfolder name
            parts = subfolder.split('_')
            for part in parts:
                if part.startswith('ac'):
                    size = int(part[2:])
                    break
            else:
                continue  # Skip if cluster size not found

            for pair, counts in stats.items():
                mean_coordination = np.mean(counts)
                neighbor_element = pair[1]
                data.append({
                    'Cluster Size': size,
                    'Neighbor Element': neighbor_element,
                    'Mean Coordination Number': mean_coordination
                })

        # Convert to DataFrame
        import pandas as pd  # Ensure pandas is imported
        df = pd.DataFrame(data)

        # Pivot for plotting
        pivot_df = df.pivot_table(
            index='Cluster Size',
            columns='Neighbor Element',
            values='Mean Coordination Number',
            aggfunc=np.sum
        ).fillna(0)

        # Plot stacked bar chart
        pivot_df.sort_index(inplace=True)
        pivot_df.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='Set2')

        plt.xlabel('Cluster Size (Number of Pb Atoms)')
        plt.ylabel('Average Coordination Number per Pb Atom')
        plt.title('Average Coordination Numbers by Cluster Size and Neighbor Element')
        plt.legend(title='Neighbor Element')
        plt.tight_layout()
        plt.show()

    # Heatmap of Coordination Numbers
    def calculate_and_plot_coordination_heatmap_from_data(self, central_element, neighbor_elements, x_range, y_range):
        """
        Calculates and plots a 2D heatmap of coordination numbers using precomputed data.

        Parameters:
        - central_element (str): The element symbol of the central atom (e.g., 'Pb').
        - neighbor_elements (list): A list containing exactly two neighbor elements (e.g., ['I', 'O']).
        - x_range (tuple): The range for the x-axis coordination number (min, max) inclusive.
        - y_range (tuple): The range for the y-axis coordination number (min, max) inclusive.
        """
        if len(neighbor_elements) != 2:
            raise ValueError("Exactly two neighbor elements must be provided.")

        x_neighbor = neighbor_elements[0]
        y_neighbor = neighbor_elements[1]

        # Initialize a 2D grid for the heatmap
        heatmap_data = np.zeros((y_range[1] - y_range[0] + 1, x_range[1] - x_range[0] + 1))

        # Iterate over coordination details collected earlier
        for pdb_file, details_list in self.coordination_details.items():
            for details in details_list:
                target_atom_element = details['target_atom_element']
                if target_atom_element != central_element:
                    continue

                coordination_stats = details['coordination_stats']
                # Get coordination numbers for the neighbor elements
                x_cn = coordination_stats.get((central_element, x_neighbor), (0, 0))[0]
                y_cn = coordination_stats.get((central_element, y_neighbor), (0, 0))[0]

                # Ensure the coordination numbers are within specified ranges and update the heatmap
                if x_range[0] <= x_cn <= x_range[1] and y_range[0] <= y_cn <= y_range[1]:
                    x_idx = x_cn - x_range[0]
                    y_idx = y_cn - y_range[0]
                    heatmap_data[y_idx, x_idx] += 1

        # Generate dynamic labels and title
        x_label = f'Coordination Number with {x_neighbor}'
        y_label = f'Coordination Number with {y_neighbor}'
        title = f'Coordination Environment Heatmap for {central_element}'

        # Plot the heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(heatmap_data, annot=True, fmt="g", cmap="YlGnBu", cbar_kws={'label': 'Count'})
        plt.xlabel(x_label, fontsize=14)  # Enlarging the x-axis label
        plt.ylabel(y_label, fontsize=14)  # Enlarging the y-axis label
        plt.title(title)
        plt.xticks(ticks=np.arange(x_range[1] - x_range[0] + 1) + 0.5, labels=np.arange(x_range[0], x_range[1] + 1), rotation=0)
        plt.yticks(ticks=np.arange(y_range[1] - y_range[0] + 1) + 0.5, labels=np.arange(y_range[0], y_range[1] + 1), rotation=0)
        plt.show()

    # Heatmap of Sharing Patterns
    def plot_sharing_patterns_heatmap(self, sharing_patterns):
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np

        # Prepare data
        data = []
        total_targets = 0  # To keep track of total target atoms
        for pattern, count in sharing_patterns.items():
            num_targets, target_element, num_neighbors, neighbor_element = pattern
            data.append({
                'Number of Target Atoms': num_targets,
                'Number of Neighbor Atoms': num_neighbors,
                'Count': count
            })
            total_targets += num_targets * count  # Update total target atoms

        df = pd.DataFrame(data)

        # Pivot the DataFrame to create a matrix
        pivot_df = df.pivot_table(
            index='Number of Target Atoms',
            columns='Number of Neighbor Atoms',
            values='Count',
            aggfunc='sum',
            fill_value=0
        )

        # Create a mask for non-sharing patterns (where Number of Target Atoms == 1)
        mask = pivot_df.index == 1

        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            pivot_df,
            annot=True,
            fmt='d',
            cmap='YlGnBu',
            cbar_kws={'label': 'Count'},
            linewidths=0.5,
            linecolor='gray'
        )

        # Overlay hatch on non-sharing patterns
        for y_index, y_value in enumerate(pivot_df.index):
            for x_index, x_value in enumerate(pivot_df.columns):
                if y_value == 1:
                    plt.gca().add_patch(plt.Rectangle(
                        (x_index, y_index), 1, 1, fill=False, hatch='///', edgecolor='red', lw=0
                    ))

        plt.title('Sharing Patterns Heatmap (Hatched cells indicate non-sharing targets)')
        plt.xlabel('Number of Neighbor Atoms')
        plt.ylabel('Number of Target Atoms')

        # Calculate totals
        total_sharing_targets = df[df['Number of Target Atoms'] > 1]['Number of Target Atoms'] * df[df['Number of Target Atoms'] > 1]['Count']
        total_sharing_targets_count = total_sharing_targets.sum()
        total_non_sharing_targets = df[df['Number of Target Atoms'] == 1]['Number of Target Atoms'] * df[df['Number of Target Atoms'] == 1]['Count']
        total_non_sharing_targets_count = total_non_sharing_targets.sum()

        # Add total counts annotation
        plt.figtext(
            0.5, -0.05,
            f"Total Target Atoms: {int(total_targets)}, Sharing Targets: {int(total_sharing_targets_count)}, "
            f"Non-Sharing Targets: {int(total_non_sharing_targets_count)}",
            wrap=True, horizontalalignment='center', fontsize=12
        )

        plt.tight_layout()
        plt.show()

    # Plotting Sharing Pattern Distributions
    def plot_sharing_pattern_distribution(self, neighbor_atom):
        """
        Plots a histogram where the x-axis represents specific sharing patterns between target atoms
        and the specified neighbor atom. The y-axis represents the counts of target atoms that exhibit
        each sharing pattern.

        Parameters:
        - neighbor_atom (str): The neighbor atom type to focus on (e.g., 'I').
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from collections import defaultdict

        # Ensure coordination details are available
        if not hasattr(self, 'coordination_details'):
            print("Coordination details not found. Please run the coordination calculations first.")
            return

        # Data structures to hold sharing information
        neighbor_to_targets = defaultdict(set)  # Maps neighbor atom IDs to sets of connected target atom IDs
        target_elements_set = set()  # To collect target elements for dynamic labeling

        # Iterate over the coordination details
        for pdb_file, target_atom_data_list in self.coordination_details.items():
            for target_atom_data in target_atom_data_list:
                target_atom_id = target_atom_data['target_atom_id']
                target_atom_element = target_atom_data['target_atom_element']
                neighbor_atom_ids = target_atom_data['neighbor_atom_ids']
                counting_stats = target_atom_data['counting_stats']

                # Add target atom element to the set for labeling
                target_elements_set.add(target_atom_element)

                for neighbor_atom_id in neighbor_atom_ids:
                    # Get the neighbor atom's element from counting_stats
                    neighbor_info = counting_stats[neighbor_atom_id]
                    neighbor_element = neighbor_info['element']

                    if neighbor_element != neighbor_atom:
                        continue  # Skip neighbor atoms not of the specified type

                    # Add the target atom to the set of connected targets for this neighbor
                    neighbor_to_targets[neighbor_atom_id].add(target_atom_id)

        # Now, analyze the sharing patterns
        sharing_patterns = defaultdict(lambda: defaultdict(set))  # sharing_patterns[num_targets][neighbor_id] = set of target_atom_ids

        # Collect the sharing patterns
        for neighbor_id, target_set in neighbor_to_targets.items():
            num_targets = len(target_set)
            # Record the neighbor atoms grouped by the number of target atoms they are shared with
            sharing_patterns[num_targets][neighbor_id] = target_set

        # Now, create combinations of num_targets and num_shared_neighbors
        pattern_counts = {}  # Key: (num_targets, num_shared_neighbors), Value: number of target atoms involved

        for num_targets, neighbor_dict in sharing_patterns.items():
            num_shared_neighbors = len(neighbor_dict)
            pattern = (num_targets, num_shared_neighbors)

            # Collect all target atoms involved in this pattern
            target_atoms_in_pattern = set()
            for neighbor_id, target_set in neighbor_dict.items():
                target_atoms_in_pattern.update(target_set)

            num_target_atoms = len(target_atoms_in_pattern)

            pattern_counts[pattern] = num_target_atoms

        # Prepare data for plotting
        patterns = list(pattern_counts.keys())
        counts = [pattern_counts[pattern] for pattern in patterns]

        # Sort patterns for consistent plotting
        patterns_sorted = sorted(patterns, key=lambda x: (x[0], x[1]))
        counts_sorted = [pattern_counts[pattern] for pattern in patterns_sorted]

        # Create x-axis labels
        x_labels = [f"{tgt}_{shr}" for tgt, shr in patterns_sorted]

        # Prepare dynamic axis labels
        target_elements_str = ', '.join(sorted(target_elements_set))

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 8))

        x_positions = np.arange(len(x_labels))
        ax.bar(x_positions, counts_sorted, color='skyblue', edgecolor='black')

        # Set x-axis labels with subscripts
        x_tick_labels = [f"{tgt}$_{{{shr}}}$" for tgt, shr in patterns_sorted]
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_tick_labels, fontsize=12)

        # Set labels and title
        ax.set_xlabel(f"Number of {target_elements_str} Atoms Sharing {neighbor_atom} Neighbors", fontsize=14)
        ax.set_ylabel(f"Number of {target_elements_str} Atoms", fontsize=14)
        ax.set_title(f"Sharing Patterns of {neighbor_atom} Neighbors among {target_elements_str} Atoms", fontsize=16)

        # Show grid
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()

    def plot_sharing_pattern_histogram(self, sharing_patterns, target_atom='Pb', color_palette='Pastel1'):
        """
        Plots a histogram where the x-axis represents specific sharing patterns between target atoms
        and neighbor atoms. The y-axis represents the number of instances of each sharing pattern.
        
        Parameters:
        - sharing_patterns (dict): The sharing patterns data.
        - target_atom (str): The target atom type to focus on (e.g., 'Pb').
        - color_palette (str): The color palette to use for the histogram bars.
                                Options: 'Pastel1', 'Pastel2', 'tab20a', 'tab20b', 'tab20c'.
                                Default is 'Pastel1'.
        """
        # Supported color palettes
        supported_palettes = ['Pastel1', 'Pastel2', 'tab20a', 'tab20b', 'tab20c']
        
        # Validate the color_palette parameter
        if color_palette not in supported_palettes:
            raise ValueError(f"Invalid color_palette '{color_palette}'. Supported palettes are: {supported_palettes}")
        
        # Data structures
        pattern_counts = defaultdict(int)  # Key: 'N_M', Value: count of instances
        total_instances_num_targets_eq_1 = 0  # To sum counts where num_targets == 1

        # Process sharing_patterns
        for pattern, count in sharing_patterns.items():
            num_targets, target_element, num_neighbors, neighbor_element = pattern

            if target_element != target_atom:
                continue  # Skip patterns not involving the specified target atom

            if num_targets == 1:
                # Sum all instances where num_targets == 1 under '1_0'
                total_instances_num_targets_eq_1 += count
            else:
                # Create the 'N_M' label
                key = f"{num_targets}_{num_neighbors}"
                pattern_counts[key] += count

        # Add the aggregated instances for num_targets == 1 under '1_0'
        if total_instances_num_targets_eq_1 > 0:
            pattern_counts['1_0'] = total_instances_num_targets_eq_1

        # Prepare data for plotting
        patterns = list(pattern_counts.keys())
        counts = [pattern_counts[pattern] for pattern in patterns]

        # Sort patterns for consistent plotting
        def sort_key(label):
            num_targets, num_neighbors = map(int, label.split('_'))
            return (num_targets, num_neighbors)

        patterns_sorted = sorted(patterns, key=sort_key)
        counts_sorted = [pattern_counts[pattern] for pattern in patterns_sorted]

        # Create x-axis labels with subscripts (e.g., '1$_{0}$')
        x_labels = [label.replace('_', '$_{') + '}$' for label in patterns_sorted]

        # Debug: Print the patterns being assigned
        print("\nAssigned Sharing Patterns:")
        for key in patterns_sorted:
            num_targets, num_neighbors = key.split('_')
            num_targets = int(num_targets)
            num_neighbors = int(num_neighbors)
            print(f"Pattern {num_targets}_{num_neighbors}: {pattern_counts[key]} instance(s)")

        # Number of histogram blocks
        num_blocks = len(x_labels)

        # Helper function to get evenly spaced colors from a discrete palette
        def get_evenly_spaced_colors(cmap, num_colors):
            """
            Retrieves a list of colors evenly spaced across the given colormap.

            Parameters:
            - cmap (matplotlib.colors.Colormap): The colormap to sample colors from.
            - num_colors (int): The number of colors needed.

            Returns:
            - List of RGBA tuples.
            """
            if hasattr(cmap, 'colors'):
                palette_colors = cmap.colors  # For discrete colormaps
                if num_colors == 1:
                    return [palette_colors[0]]
                step = (len(palette_colors) - 1) / (num_colors - 1) if num_colors > 1 else 0
                indices = [int(round(step * i)) for i in range(num_colors)]
                colors = [palette_colors[idx] for idx in indices]
            else:
                # For continuous colormaps, sample evenly
                colors = cmap(np.linspace(0, 1, num_colors))
            return colors

        # Retrieve the selected colormap
        cmap = plt.get_cmap(color_palette)
        
        # Get evenly spaced colors from the colormap
        colors = get_evenly_spaced_colors(cmap, num_blocks)

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 8))

        x_positions = np.arange(len(x_labels))
        bars = ax.bar(x_positions, counts_sorted, color=colors, edgecolor='black')

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=12)

        # Set labels and title
        ax.set_xlabel(f"Sharing Patterns (Number of {target_atom} Atoms Sharing Neighbor Atoms)", fontsize=14)
        ax.set_ylabel("Number of Instances", fontsize=14)
        ax.set_title(f"Sharing Patterns of Neighbor Atoms among {target_atom} Atoms", fontsize=16)

        # Add count annotations above each bar (non-bold)
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,  # X-coordinate: center of the bar
                height,                             # Y-coordinate: top of the bar
                f'{height}',                        # Text: the count
                ha='center',                        # Horizontal alignment
                va='bottom',                        # Vertical alignment
                fontsize=12,
                color='black'
                # Removed fontweight='bold' to make the text non-bold
            )

        # Show grid
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Optional: Add a legend mapping colors to patterns (if needed)
        # This can be useful for clarity when using many colors
        # Uncomment the following lines to add a legend
        """
        handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(num_blocks)]
        ax.legend(handles, x_labels, title="Sharing Patterns", bbox_to_anchor=(1.05, 1), loc='upper left')
        """

        plt.tight_layout()
        plt.show()

    # Plotting Coordination Distribution Information
    def prepare_coordination_distribution_data(self):
        """
        Prepares data for plotting the distribution of coordination numbers,
        including the counts of specific coordination combinations.
        """
        import pandas as pd

        data = []
        for pdb_file, details_list in self.coordination_details.items():
            for details in details_list:
                target_atom_element = details['target_atom_element']
                coordination_stats = details['coordination_stats']

                # Collect coordination numbers with each neighbor element
                neighbor_coords = {}
                total_coordination_number = 0
                for (target_elem, neighbor_elem), (coord_num, _) in coordination_stats.items():
                    neighbor_coords[neighbor_elem] = coord_num
                    total_coordination_number += coord_num

                # Create a tuple representing the coordination combination
                # We'll sort the items to ensure consistent ordering
                coordination_combination = tuple(sorted(neighbor_coords.items()))

                data.append({
                    'Total Coordination Number': total_coordination_number,
                    'Coordination Combination': coordination_combination,
                    'Target Atom Element': target_atom_element
                })

        # Create a DataFrame
        df = pd.DataFrame(data)

        # Ensure 'Target Atom Element' is included in the grouping
        coordination_distribution = df.groupby(
            ['Total Coordination Number', 'Coordination Combination', 'Target Atom Element']
        ).size().reset_index(name='Count')

        # Calculate total counts for each total coordination number and target atom element
        total_counts = coordination_distribution.groupby(
            ['Total Coordination Number', 'Target Atom Element']
        )['Count'].sum().reset_index(name='Total Count')

        # Merge total counts back into the DataFrame
        coordination_distribution = coordination_distribution.merge(
            total_counts, on=['Total Coordination Number', 'Target Atom Element']
        )

        # Calculate the percentage for each coordination combination
        coordination_distribution['Percentage'] = (
            coordination_distribution['Count'] / coordination_distribution['Total Count']
        ) * 100

        # Sort the DataFrame for consistent plotting
        coordination_distribution.sort_values(
            by=['Total Coordination Number', 'Target Atom Element', 'Percentage'],
            ascending=[True, True, False], inplace=True
        )

        self.coordination_distribution_df = coordination_distribution  # Store for later use

    def format_combination_label(self, combination):
        """
        Formats the coordination combination tuple into a readable string.

        Parameters:
        - combination (tuple of tuples): Each tuple contains (element, coordination_number).

        Returns:
        - str: Formatted coordination combination.
        """
        if not combination:
            return "None"

        # Sort the combination for consistency
        sorted_combination = sorted(combination, key=lambda x: x[0])

        # Format each element and its coordination number
        formatted_parts = [f"{elem}:{coord_num}" for elem, coord_num in sorted_combination]

        # Join the parts with commas
        formatted_label = ", ".join(formatted_parts)

        return formatted_label

    def format_combination_label_exclude(self, combination, exclude_atom):
        """
        Formats the coordination combination into a readable string,
        excluding the specified neighbor atom.

        Parameters:
        - combination (tuple of tuples): Each tuple contains (element, coordination_number).
        - exclude_atom (str): The neighbor atom type to exclude from the label.

        Returns:
        - str: Formatted coordination combination excluding the specified atom.
        """
        # Exclude the specified atom
        filtered_combination = [(elem, coord_num) for elem, coord_num in combination if elem != exclude_atom]

        if not filtered_combination:
            return "None"

        # Sort the combination for consistency
        sorted_combination = sorted(filtered_combination, key=lambda x: x[0])

        # Format each element and its coordination number
        formatted_parts = [f"{elem}:{coord_num}" for elem, coord_num in sorted_combination]

        # Join the parts with commas
        formatted_label = ", ".join(formatted_parts)

        return formatted_label

    def plot_coordination_number_distribution(self):
        """
        Plots a stacked histogram of the total coordination numbers,
        showing the distribution of specific coordination combinations.
        Blocks are stacked in order based on the ratio of the two neighboring atoms.
        In cases where one coordination atom is zero, blocks are stacked in ascending order
        of the count of the other neighboring atom.

        Only blocks representing ≥ 0.5% of the total are annotated.
        The y-axis is expanded by 10% to accommodate annotations.
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
        import numpy as np
        from matplotlib.colors import to_rgba
        from matplotlib.ticker import MaxNLocator, AutoMinorLocator

        # Ensure data is prepared
        if not hasattr(self, 'coordination_distribution_df'):
            self.prepare_coordination_distribution_data()
        df = self.coordination_distribution_df.copy()

        # Compute global total count
        global_total_count = df['Count'].sum()

        # Pivot the DataFrame
        pivot_df = df.pivot_table(
            index='Total Coordination Number',
            columns='Coordination Combination',
            values='Count',
            aggfunc='sum',
            fill_value=0
        )

        # Get the list of unique total coordination numbers
        total_coord_numbers = pivot_df.index.tolist()

        # Get neighbor elements for dynamic x-axis label
        neighbor_elements = set()
        for combination in df['Coordination Combination']:
            for neighbor_elem, _ in combination:
                neighbor_elements.add(neighbor_elem)
        neighbor_elements = sorted(neighbor_elements)
        neighbor_elements_str = ', '.join(neighbor_elements)

        # Get target atom elements for dynamic labels
        target_elements = df['Target Atom Element'].unique()
        target_elements_str = ', '.join(target_elements)

        # Define colors for each total coordination number from 'tab20b' palette
        num_total_coord_numbers = len(total_coord_numbers)
        full_color_palette = sns.color_palette('tab20b', 20)  # 'tab20b' has 20 colors

        # Evenly space colors across the palette, cycling if necessary
        if num_total_coord_numbers <= 20:
            color_indices = np.linspace(0, 19, num=num_total_coord_numbers, dtype=int)
        else:
            # If more than 20 coordination numbers, cycle the palette
            color_indices = np.tile(np.arange(20), int(np.ceil(num_total_coord_numbers / 20)))[:num_total_coord_numbers]
        base_color_palette = [full_color_palette[i] for i in color_indices]
        total_coord_color_map = dict(zip(total_coord_numbers, base_color_palette))

        # Create a mapping of coordination combinations to shades
        # We'll store the shades used for each combination to reuse in the next plot
        self.combination_color_map = {}

        # Precompute total counts for each total coordination number
        total_counts_per_coord_num = df.groupby('Total Coordination Number')['Count'].sum().to_dict()

        # Find the maximum total count
        max_total_count = max(total_counts_per_coord_num.values())

        # Calculate y_max and expand by 10%
        y_max_original = ((int(max_total_count) + 9) // 10) * 10  # Original y_max rounded up to nearest 10
        y_max = int(np.ceil(y_max_original * 1.1 / 10)) * 10    # Increase y_max by 10% and round up to nearest 10

        # Set up the plot (Initialize once)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_ylim(0, y_max)

        # Set bar width to make columns narrower
        bar_width = 0.5  # Adjust this value as needed (default is 0.8)

        # Ensure neighbor_elements are sorted for consistent ordering
        neighbor_elements = sorted(neighbor_elements)
        if len(neighbor_elements) >= 2:
            elem1, elem2 = neighbor_elements[0], neighbor_elements[1]
        else:
            # Handle cases with less than two neighbor elements
            elem1 = neighbor_elements[0]
            elem2 = None

        # Loop over total coordination numbers to plot each bar
        for idx, total_coord_num in enumerate(total_coord_numbers):
            # Filter df for the current total coordination number
            current_df = df[df['Total Coordination Number'] == total_coord_num]

            # Sum counts for each coordination combination
            counts = current_df.groupby('Coordination Combination')['Count'].sum()

            # Get the precomputed total count for this total coordination number
            total_count = total_counts_per_coord_num[total_coord_num]

            # Base color for this total coordination number
            base_color = total_coord_color_map[total_coord_num]
            base_rgba = to_rgba(base_color)

            # Generate shades for combinations
            num_combinations = len(counts)
            if num_combinations > 0:
                shades = np.linspace(1.0, 0.6, num_combinations)
            else:
                shades = []

            # Prepare combinations_counts list
            combinations_counts = list(counts.items())

            # Define sorting key function
            def sorting_key_function(item):
                combination, count = item
                # Extract coordination numbers for neighbor elements
                coord_nums = {elem: coord_num for elem, coord_num in combination}
                # Get coordination numbers for the two neighbor elements
                coord_num1 = coord_nums.get(elem1, 0)
                coord_num2 = coord_nums.get(elem2, 0) if elem2 else 0
                if coord_num1 > 0 and coord_num2 > 0:
                    ratio = coord_num1 / coord_num2
                    return (0, -ratio)
                elif (coord_num1 == 0 and coord_num2 > 0) or (coord_num2 == 0 and coord_num1 > 0):
                    # Stack in ascending order of the non-zero coordination number
                    non_zero_coord_num = coord_num1 if coord_num1 > 0 else coord_num2
                    return (1, non_zero_coord_num)
                else:
                    # Both coordination numbers are zero
                    return (2, 0)

            # Sort combinations based on the sorting key
            sorted_combinations = sorted(combinations_counts, key=sorting_key_function)

            # Now assign colors and plot
            bottom = 0
            for comb_idx, (combination, count) in enumerate(sorted_combinations):
                if count == 0:
                    continue  # Skip zero counts

                # Calculate percentage
                percentage = (count / global_total_count) * 100

                if percentage >= 0.5:
                    # Color for this block
                    shade = shades[comb_idx]
                    color = tuple(np.array(base_rgba[:3]) * shade) + (1.0,)

                    # Save the color mapping for this combination
                    self.combination_color_map[combination] = color

                    # Compute global percentage
                    global_percentage = percentage

                    # Plot the block
                    ax.bar(
                        total_coord_num,
                        count,
                        bottom=bottom,
                        color=color,
                        edgecolor='black',
                        width=bar_width,
                        align='center'
                    )

                    # Determine text color based on background brightness
                    brightness = np.mean(color[:3])
                    text_color = 'black' if brightness > 0.5 else 'white'

                    # Prepare the label (coordination combination only)
                    label = f"{self.format_combination_label(combination)}"

                    # Place the coordination combination label inside the block
                    ax.text(
                        total_coord_num,
                        bottom + count / 2,
                        label,
                        ha='center',
                        va='center',
                        fontsize=10,
                        rotation=0,
                        color=text_color
                    )

                    # Place the global percentage annotation to the right of the block
                    ax.text(
                        total_coord_num + bar_width / 2 + 0.05,
                        bottom + count / 2,
                        f"{global_percentage:.1f}%",
                        ha='left',
                        va='center',
                        fontsize=10,
                        color='black'
                    )
                else:
                    # Color for this block
                    shade = shades[comb_idx]
                    color = tuple(np.array(base_rgba[:3]) * shade) + (1.0,)

                    # Save the color mapping for this combination even if not annotated
                    self.combination_color_map[combination] = color

                    # Plot the block without annotations
                    ax.bar(
                        total_coord_num,
                        count,
                        bottom=bottom,
                        color=color,
                        edgecolor='black',
                        width=bar_width,
                        align='center'
                    )

                bottom += count

        # After plotting all bars, add total percentage annotations
        for total_coord_num in total_coord_numbers:
            total_count = total_counts_per_coord_num[total_coord_num]
            total_percentage = (total_count / global_total_count) * 100

            # Place the total percentage annotation at the top of the column
            ax.text(
                total_coord_num,
                total_count + y_max * 0.01,  # Slightly above the top of the bar
                f"{total_percentage:.1f}%",
                ha='center',
                va='bottom',
                fontsize=10,
                color='black'
            )

        # Set y-axis to integer ticks with a fixed number of ticks
        ax.yaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))  # Fixed to 10 ticks

        # Add minor ticks to the y-axis
        ax.grid(which='minor', axis='y', linestyle='--', alpha=0.3)

        # Set labels and title with enlarged font sizes
        ax.set_xlabel(f'Total {target_elements_str} Coordination Number', fontsize=14)
        ax.set_ylabel(f'Number of {target_elements_str} Atoms', fontsize=14)
        ax.set_title('Distribution of Coordination Numbers and Environments', fontsize=16)

        # Enlarge axis tick labels
        ax.tick_params(axis='both', which='major', labelsize=12)

        # Remove the legend if not needed
        ax.legend().set_visible(False)

        # Final layout adjustments and display the plot
        plt.tight_layout()
        plt.show()

    def plot_neighbor_atom_distribution(self, neighbor_atom):
        """
        Plots a histogram where each bar represents the number of coordinated neighbor atoms,
        and blocks within each bar are stacked based on Sharing Patterns from top down.

        Only blocks representing ≥ 0.5% of the total are annotated.
        The y-axis is expanded by 10% to accommodate annotations.

        Parameters:
        - neighbor_atom (str): The neighbor atom type to focus on (e.g., 'I').
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
        import numpy as np
        from matplotlib.colors import to_rgba
        from matplotlib.ticker import MaxNLocator, AutoMinorLocator

        # Ensure data is prepared
        if not hasattr(self, 'coordination_distribution_df'):
            self.prepare_coordination_distribution_data()
        df = self.coordination_distribution_df.copy()

        # Ensure combination_color_map is available
        if not hasattr(self, 'combination_color_map'):
            self.plot_coordination_number_distribution()  # This will generate the mapping
            df = self.coordination_distribution_df.copy()

        # Compute global total count
        global_total_count = df['Count'].sum()

        # Extract target atom elements for dynamic labels
        target_elements = df['Target Atom Element'].unique()
        target_elements_str = ', '.join(target_elements)

        # Extract coordination number for the specified neighbor atom
        def get_neighbor_coord_num(combination):
            for elem, coord_num in combination:
                if elem == neighbor_atom:
                    return coord_num
            return 0  # If the neighbor atom is not in the combination

        df['Neighbor Coordination Number'] = df['Coordination Combination'].apply(get_neighbor_coord_num)

        # Group by Neighbor Coordination Number, Coordination Combination, and Total Coordination Number
        grouped_df = df.groupby(['Neighbor Coordination Number', 'Coordination Combination', 'Total Coordination Number'])['Count'].sum().reset_index()

        # Precompute total counts for each neighbor coordination number
        total_counts_per_neighbor_coord_num = grouped_df.groupby('Neighbor Coordination Number')['Count'].sum().to_dict()

        # Find the maximum total count
        max_total_count = max(total_counts_per_neighbor_coord_num.values())

        # Calculate y_max and expand by 10%
        y_max_original = ((int(max_total_count) + 9) // 10) * 10  # Original y_max rounded up to nearest 10
        y_max = int(np.ceil(y_max_original * 1.1 / 10)) * 10    # Increase y_max by 10% and round up to nearest 10

        # Set up the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_ylim(0, y_max)

        # Get the list of unique neighbor coordination numbers
        neighbor_coord_numbers = sorted(grouped_df['Neighbor Coordination Number'].unique())

        # Set bar width to make columns narrower
        bar_width = 0.5  # Adjust as needed

        # Define a function to format the combination label excluding the neighbor atom
        def format_combination_label_exclude(combination, exclude_atom):
            """
            Formats the coordination combination into a readable string,
            excluding the specified neighbor atom.

            Parameters:
            - combination (tuple of tuples): Each tuple contains (element, coordination_number).
            - exclude_atom (str): The neighbor atom type to exclude from the label.

            Returns:
            - str: Formatted coordination combination excluding the specified atom.
            """
            # Exclude the specified atom
            filtered_combination = [(elem, coord_num) for elem, coord_num in combination if elem != exclude_atom]

            if not filtered_combination:
                return "None"

            # Sort the combination for consistency
            sorted_combination = sorted(filtered_combination, key=lambda x: x[0])

            # Format each element and its coordination number
            formatted_parts = [f"{elem}:{coord_num}" for elem, coord_num in sorted_combination]

            # Join the parts with commas
            formatted_label = ", ".join(formatted_parts)

            return formatted_label

        # Loop over neighbor coordination numbers to plot each bar
        for idx, neighbor_coord_num in enumerate(neighbor_coord_numbers):
            # Filter data for the current neighbor coordination number
            current_df = grouped_df[grouped_df['Neighbor Coordination Number'] == neighbor_coord_num]

            # Sort combinations by 'Total Coordination Number' descending
            sorted_combinations = current_df.sort_values('Total Coordination Number', ascending=False)

            # Get the total count for this neighbor coordination number
            total_count = total_counts_per_neighbor_coord_num[neighbor_coord_num]

            bottom = 0
            for _, row in sorted_combinations.iterrows():
                combination = row['Coordination Combination']
                count = row['Count']
                total_coord_num = row['Total Coordination Number']

                # Calculate percentage
                percentage = (count / global_total_count) * 100

                if percentage >= 0.5:
                    # Retrieve the color from the combination_color_map
                    color = self.combination_color_map.get(combination, (0.5, 0.5, 0.5, 1.0))  # Default to gray if not found

                    # Compute global percentage
                    global_percentage = percentage

                    # Plot the block
                    ax.bar(
                        neighbor_coord_num,
                        count,
                        bottom=bottom,
                        color=color,
                        edgecolor='black',
                        width=bar_width,
                        align='center'
                    )

                    # Determine text color based on background brightness
                    brightness = np.mean(color[:3])
                    text_color = 'black' if brightness > 0.5 else 'white'

                    # Prepare the label excluding the neighbor atom
                    label = self.format_combination_label_exclude(combination, neighbor_atom)

                    # Place the coordination combination label inside the block
                    ax.text(
                        neighbor_coord_num,
                        bottom + count / 2,
                        label,
                        ha='center',
                        va='center',
                        fontsize=10,
                        rotation=0,
                        color=text_color
                    )

                    # Place the global percentage annotation to the right of the block
                    ax.text(
                        neighbor_coord_num + bar_width / 2 + 0.05,
                        bottom + count / 2,
                        f"{global_percentage:.1f}%",
                        ha='left',
                        va='center',
                        fontsize=10,
                        color='black'
                    )
                else:
                    # Retrieve the color from the combination_color_map
                    color = self.combination_color_map.get(combination, (0.5, 0.5, 0.5, 1.0))  # Default to gray if not found

                    # Plot the block without annotations
                    ax.bar(
                        neighbor_coord_num,
                        count,
                        bottom=bottom,
                        color=color,
                        edgecolor='black',
                        width=bar_width,
                        align='center'
                    )

                bottom += count

        # After plotting all blocks for each neighbor_coord_num, annotate the total percentage
        for neighbor_coord_num in neighbor_coord_numbers:
            total_count = total_counts_per_neighbor_coord_num[neighbor_coord_num]
            total_percentage = (total_count / global_total_count) * 100

            # Place the total percentage annotation at the top of the column
            ax.text(
                neighbor_coord_num,
                total_count + y_max * 0.01,  # Slightly above the top of the bar
                f"{total_percentage:.1f}%",
                ha='center',
                va='bottom',
                fontsize=10,
                color='black'
            )

        # Set y-axis to integer ticks with a fixed number of ticks
        ax.yaxis.set_major_locator(MaxNLocator(nbins=10))  # Fixed to 10 ticks

        # Add minor ticks to the y-axis
        ax.grid(which='minor', axis='y', linestyle='--', alpha=0.3)

        # Set labels and title with enlarged font sizes
        ax.set_xlabel(f'Number of Coordinated {neighbor_atom} Atoms', fontsize=14)
        ax.set_ylabel(f'Number of {target_elements_str} Atoms', fontsize=14)
        ax.set_title(f'Distribution of {neighbor_atom} Coordination Numbers', fontsize=16)

        # Enlarge axis tick labels
        ax.tick_params(axis='both', which='major', labelsize=12)

        # Remove the legend if not needed
        ax.legend().set_visible(False)

        plt.tight_layout()
        plt.show()

'''
    # def plot_sharing_pattern_histogram(self, sharing_patterns, target_atom='Pb', color_palette='Pastel1'):
    #     """
    #     Plots a histogram where the x-axis represents specific sharing patterns between target atoms
    #     and neighbor atoms. The y-axis represents the number of instances of each sharing pattern.
        
    #     Parameters:
    #     - sharing_patterns (dict): The sharing patterns data.
    #     - target_atom (str): The target atom type to focus on (e.g., 'Pb').
    #     - color_palette (str): The color palette to use for the histogram bars.
    #                             Options: 'Pastel1', 'Pastel2', 'tab20a', 'tab20b', 'tab20c'.
    #                             Default is 'Pastel1'.
    #     """
    #     # Supported color palettes
    #     supported_palettes = ['Pastel1', 'Pastel2', 'tab20a', 'tab20b', 'tab20c']
        
    #     # Validate the color_palette parameter
    #     if color_palette not in supported_palettes:
    #         raise ValueError(f"Invalid color_palette '{color_palette}'. Supported palettes are: {supported_palettes}")
        
    #     # Data structures
    #     pattern_counts = defaultdict(int)  # Key: 'N_M', Value: count of instances
    #     total_instances_num_targets_eq_1 = 0  # To sum counts where num_targets == 1

    #     # Process sharing_patterns
    #     for pattern, count in sharing_patterns.items():
    #         num_targets, target_element, num_neighbors, neighbor_element = pattern

    #         if target_element != target_atom:
    #             continue  # Skip patterns not involving the specified target atom

    #         if num_targets == 1:
    #             # Sum all instances where num_targets == 1 under '1_0'
    #             total_instances_num_targets_eq_1 += count
    #         else:
    #             # Create the 'N_M' label
    #             key = f"{num_targets}_{num_neighbors}"
    #             pattern_counts[key] += count

    #     # Add the aggregated instances for num_targets == 1 under '1_0'
    #     if total_instances_num_targets_eq_1 > 0:
    #         pattern_counts['1_0'] = total_instances_num_targets_eq_1

    #     # Prepare data for plotting
    #     patterns = list(pattern_counts.keys())
    #     counts = [pattern_counts[pattern] for pattern in patterns]

    #     # Sort patterns for consistent plotting
    #     def sort_key(label):
    #         num_targets, num_neighbors = map(int, label.split('_'))
    #         return (num_targets, num_neighbors)

    #     patterns_sorted = sorted(patterns, key=sort_key)
    #     counts_sorted = [pattern_counts[pattern] for pattern in patterns_sorted]

    #     # Create x-axis labels with subscripts (e.g., '1$_{0}$')
    #     x_labels = [label.replace('_', '$_{') + '}$' for label in patterns_sorted]

    #     # Debug: Print the patterns being assigned
    #     print("\nAssigned Sharing Patterns:")
    #     for key in patterns_sorted:
    #         num_targets, num_neighbors = key.split('_')
    #         num_targets = int(num_targets)
    #         num_neighbors = int(num_neighbors)
    #         print(f"Pattern {num_targets}_{num_neighbors}: {pattern_counts[key]} instance(s)")

    #     # Number of histogram blocks
    #     num_blocks = len(x_labels)

    #     # Retrieve the selected colormap
    #     cmap = plt.get_cmap(color_palette)
        
    #     # Extract colors from the colormap
    #     if hasattr(cmap, 'colors'):
    #         palette_colors = cmap.colors  # For discrete colormaps
    #     else:
    #         # For continuous colormaps, sample evenly
    #         palette_colors = cmap(np.linspace(0, 1, num_blocks))
        
    #     # Assign colors to each bar, cycling through the palette if necessary
    #     colors = [palette_colors[i % len(palette_colors)] for i in range(num_blocks)]

    #     # Plotting
    #     fig, ax = plt.subplots(figsize=(12, 8))

    #     x_positions = np.arange(len(x_labels))
    #     bars = ax.bar(x_positions, counts_sorted, color=colors, edgecolor='black')

    #     ax.set_xticks(x_positions)
    #     ax.set_xticklabels(x_labels, fontsize=12)

    #     # Set labels and title
    #     ax.set_xlabel(f"Sharing Patterns (Number of {target_atom} Atoms Sharing Neighbor Atoms)", fontsize=14)
    #     ax.set_ylabel("Number of Instances", fontsize=14)
    #     ax.set_title(f"Sharing Patterns of Neighbor Atoms among {target_atom} Atoms", fontsize=16)

    #     # Add count annotations above each bar
    #     for bar in bars:
    #         height = bar.get_height()
    #         ax.text(
    #             bar.get_x() + bar.get_width() / 2,  # X-coordinate: center of the bar
    #             height,                             # Y-coordinate: top of the bar
    #             f'{height}',                        # Text: the count
    #             ha='center',                        # Horizontal alignment
    #             va='bottom',                        # Vertical alignment
    #             fontsize=12,
    #             fontweight='bold',
    #             color='black'
    #         )

    #     # Show grid
    #     ax.grid(axis='y', linestyle='--', alpha=0.7)

    #     # Optional: Add a legend mapping colors to patterns (if needed)
    #     # This can be useful for clarity when using many colors
    #     # Uncomment the following lines to add a legend
    #     """
    #     handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(num_blocks)]
    #     ax.legend(handles, x_labels, title="Sharing Patterns", bbox_to_anchor=(1.05, 1), loc='upper left')
    #     """

    #     plt.tight_layout()
    #     plt.show()
'''