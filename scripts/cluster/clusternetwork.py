import seaborn as sns
from string import ascii_uppercase
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
import os

class ClusterNetwork:
    def __init__(self, core_atoms, shell_atoms, node_elements, linker_elements, terminator_elements, segment_cutoff, core_residue_names, shell_residue_names):
        self.core_atoms = core_atoms
        self.shell_atoms = shell_atoms
        self.node_elements = node_elements
        self.linker_elements = linker_elements
        self.terminator_elements = terminator_elements
        self.segment_cutoff = segment_cutoff
        self.core_residue_names = core_residue_names
        self.shell_residue_names = shell_residue_names
        self.network_id_generator = self.generate_network_ids()

    ## -- Coordination Network Analysis
    def generate_network_ids(self):
        reserved_ids = set(self.core_residue_names + self.shell_residue_names)
        for first in ascii_uppercase:
            for second in ascii_uppercase:
                for third in ascii_uppercase:
                    network_id = f"{first}{second}{third}"
                    if network_id not in reserved_ids:
                        yield network_id

    def assign_network_id(self, atom, network_id):
        stack = [atom]
        while stack:
            current = stack.pop()
            if current.network_id is None:
                current.network_id = network_id
                connected_atoms = self.get_connected_atoms(current)
                for next_atom in connected_atoms:
                    if next_atom.network_id is None:
                        stack.append(next_atom)

    def get_connected_atoms(self, atom, threshold=None):
        if threshold is None:
            threshold = self.segment_cutoff
        connected_atoms = []
        for other_atom in self.core_atoms:
            if atom != other_atom and self.are_connected(atom, other_atom, threshold):
                connected_atoms.append(other_atom)
        return connected_atoms

    def are_connected(self, atom1, atom2, threshold):
        distance = np.linalg.norm(np.array(atom1.coordinates) - np.array(atom2.coordinates))
        return distance <= threshold

    def analyze_networks(self):
        networks = []
        for atom in self.core_atoms:
            if atom.network_id is None:
                network_id = next(self.network_id_generator)
                self.assign_network_id(atom, network_id)
                networks.append(network_id)
        return networks

    ## -- PDB File Generation
    @staticmethod
    def _create_output_folder(output_path, folder_name):
        """
        Creates an output folder. If the folder exists, uses the existing folder without appending a timestamp.
        
        Parameters:
        - output_path (str): Path to the output directory.
        - folder_name (str): Name of the folder to create.
        
        Returns:
        - full_output_path (str): The full path of the created or existing folder.
        
        Raises:
        - FileExistsError: If the folder exists and you choose to handle it differently.
        """
        full_output_path = os.path.join(output_path, folder_name)

        # If the folder already exists, do not append timestamp
        if os.path.exists(full_output_path):
            print(f"Folder '{full_output_path}' already exists. Using the existing folder.")
        else:
            # Create the directory since it does not exist
            os.makedirs(full_output_path, exist_ok=True)
            print(f"Created folder: {full_output_path}")
        
        return full_output_path

    def write_individual_cluster_pdb_files(self, pdb_handler, output_path, folder_name):
        """
        Writes separate PDB files for each unique cluster with the original residue names preserved.
        
        Parameters:
        - pdb_handler (PDBFileHandler): The PDB file handler containing atom data.
        - output_path (str): The base directory where the PDB files will be saved.
        - folder_name (str): The name of the folder to create in the output directory.
        
        Raises:
        - FileExistsError: If a PDB file to be written already exists in the target folder.
        """
        # Create the output folder
        full_output_path = self._create_output_folder(output_path, folder_name)

        # Extract the original filename without the extension
        original_filename = os.path.splitext(os.path.basename(pdb_handler.filepath))[0]

        for network_id in self.analyze_networks():
            # Create a list of atoms that belong to the current network
            cluster_atoms = [atom for atom in pdb_handler.core_atoms + pdb_handler.shell_atoms if atom.network_id == network_id]

            # Preserve original residue names (no change needed)
            for atom in cluster_atoms:
                atom.residue_name = atom.residue_name

            # Generate a new file name with the original filename and the network ID
            output_filename = f"{original_filename}_{network_id}.pdb"
            output_file_path = os.path.join(full_output_path, output_filename)

            # Check if the file already exists
            if os.path.exists(output_file_path):
                raise FileExistsError(f"File '{output_file_path}' already exists. Aborting to prevent overwriting.")

            # Write the atoms to a new PDB file
            pdb_handler.write_pdb_file(output_file_path, atoms=cluster_atoms)
            print(f"Written PDB file for cluster {network_id} to {output_file_path}")

    def write_cluster_pdb_files_with_coordinated_shell(self, pdb_handler, output_path, folder_name, target_elements, neighbor_elements, distance_thresholds, shell_residue_names):
        """
        Writes separate PDB files for each unique cluster with the original residue names and numbers preserved,
        and optionally includes shell residues that are coordinated to the cluster.
        
        Parameters:
        - pdb_handler (PDBFileHandler): The PDB file handler containing atom data.
        - output_path (str): The base directory where the PDB files will be saved.
        - folder_name (str): The name of the folder to create in the output directory.
        - target_elements (list): The list of target elements to include.
        - neighbor_elements (list): The list of neighbor elements to include.
        - distance_thresholds (dict): A dictionary of distance thresholds for atom pairs.
        - shell_residue_names (list): List of shell residue names to include.
        
        Raises:
        - FileExistsError: If a PDB file to be written already exists in the target folder.
        """
        # Create the output folder
        full_output_path = self._create_output_folder(output_path, folder_name)

        # Extract the original filename without the extension
        original_filename = os.path.splitext(os.path.basename(pdb_handler.filepath))[0]

        for network_id in self.analyze_networks():
            cluster_atoms = [atom for atom in pdb_handler.core_atoms if atom.network_id == network_id]

            included_residue_numbers = set()
            for atom in cluster_atoms:
                if atom.element in target_elements:
                    for shell_atom in pdb_handler.shell_atoms:
                        if shell_atom.element in neighbor_elements and shell_atom.residue_name in shell_residue_names:
                            pair = (atom.element, shell_atom.element)
                            if pair in distance_thresholds:
                                distance = np.linalg.norm(np.array(atom.coordinates) - np.array(shell_atom.coordinates))
                                if distance <= distance_thresholds[pair]:
                                    included_residue_numbers.add((shell_atom.residue_name, shell_atom.residue_number))

            atoms_to_write = []
            for shell_atom in pdb_handler.shell_atoms:
                if (shell_atom.residue_name, shell_atom.residue_number) in included_residue_numbers:
                    atoms_to_write.append(shell_atom)
            atoms_to_write.extend(cluster_atoms)

            output_filename = f"{original_filename}_{network_id}.pdb"
            output_file_path = os.path.join(full_output_path, output_filename)

            # Check if the file already exists
            if os.path.exists(output_file_path):
                raise FileExistsError(f"File '{output_file_path}' already exists. Aborting to prevent overwriting.")

            # Write the atoms to a new PDB file
            pdb_handler.write_pdb_file(output_file_path, atoms=atoms_to_write)
            print(f"Written PDB file for cluster {network_id} with coordinated shell residues to {output_file_path}")

    def rename_clusters_in_pdb(self, pdb_handler, output_path, output_filename):
        """
        Identifies unique clusters from the input PDB file, assigns a unique residue name to each cluster,
        and writes a single PDB file with updated residue names for the clusters.

        Parameters:
        - pdb_handler (PDBFileHandler): The PDB file handler containing atom data.
        - output_path (str): The directory where the new PDB file will be saved.
        - output_filename (str): The name of the output PDB file.
        """
        # Analyze the networks to identify clusters and assign unique network IDs
        networks = self.analyze_networks()

        # Update residue names of all atoms based on their assigned network ID (3-character residue names)
        updated_atoms = pdb_handler.core_atoms + pdb_handler.shell_atoms
        for atom in updated_atoms:
            if atom.network_id:
                atom.residue_name = atom.network_id  # Set network ID as residue name (3 characters)

        # Generate the output PDB file with the updated residue names
        output_file_path = os.path.join(output_path, output_filename)
        pdb_handler.write_pdb_file(output_file_path, atoms=updated_atoms)

        print(f"Clusters identified, residue names updated, and saved to {output_file_path}")

    ## -- Atom Count Validation Methods
    def count_linker_elements_in_input_pdb(self, pdb_handler):
        """
        Count the number of each linker element in the input PDB file.

        Parameters:
        - pdb_handler (PDBFileHandler): The PDB file handler containing atom data.

        Returns:
        - linker_element_counts (dict): A dictionary with the counts of each linker element in the input PDB.
        """
        linker_element_counts = {element: 0 for element in self.linker_elements}

        # Iterate over core atoms and count linker elements
        for atom in pdb_handler.core_atoms + pdb_handler.shell_atoms:
            if atom.element in self.linker_elements:
                linker_element_counts[atom.element] += 1

        print("Linker element counts in input PDB:")
        for element, count in linker_element_counts.items():
            print(f"{element}: {count}")

        return linker_element_counts

    def count_linker_elements_in_output_pdb_files(self, output_directory, folder_name, pdb_handler):
        """
        Count the number of each linker element across all output PDB files and verify they match the input PDB counts.

        Parameters:
        - output_directory (str): The base directory where the output PDB files are stored.
        - folder_name (str): The folder name where PDB files were saved.
        - pdb_handler (PDBFileHandler): The PDB file handler for the input file, used for comparison.
        """
        # Construct the full path to the output PDB folder
        output_pdb_folder = os.path.join(output_directory, folder_name)

        # Call the method to count linker elements in the input PDB file
        input_linker_counts = self.count_linker_elements_in_input_pdb(pdb_handler)
        output_linker_counts = {element: 0 for element in self.linker_elements}

        # Iterate over each output PDB file and count linker elements
        for pdb_filename in os.listdir(output_pdb_folder):
            if pdb_filename.endswith(".pdb"):
                pdb_filepath = os.path.join(output_pdb_folder, pdb_filename)

                # Open the PDB file and count the linker elements
                with open(pdb_filepath, 'r') as pdb_file:
                    for line in pdb_file:
                        if line.startswith("ATOM") or line.startswith("HETATM"):
                            atom_element = line[76:78].strip()
                            if atom_element in self.linker_elements:
                                output_linker_counts[atom_element] += 1

        # Output the results
        print("\nLinker element counts in output PDB files:")
        for element, count in output_linker_counts.items():
            print(f"{element}: {count}")

        # Verify the counts between input and output
        all_match = True
        for element in self.linker_elements:
            if input_linker_counts[element] != output_linker_counts[element]:
                all_match = False
                print(f"Mismatch for {element}: Input count = {input_linker_counts[element]}, Output count = {output_linker_counts[element]}")
            else:
                print(f"{element} count matches between input and output.")

        if all_match:
            print("\nAll linker element counts match between the input PDB and output PDB files.")
        else:
            print("\nWarning: Some linker element counts do not match between the input PDB and output PDB files.")

    ## -- Coordination Number & Bond Distribution Analysis
    def calculate_coordination_numbers(self, target_elements, neighbor_elements, distance_thresholds):
        coordination_numbers = defaultdict(list)
        total_coordination_numbers = defaultdict(list)
        
        for atom in self.core_atoms:
            if atom.element in target_elements:
                counts = {neighbor: 0 for neighbor in neighbor_elements}
                counted_pairs = set()
                for other_atom in self.core_atoms + self.shell_atoms:
                    if other_atom.element in neighbor_elements and (atom.atom_id, other_atom.atom_id) not in counted_pairs:
                        pair = (atom.element, other_atom.element)
                        if pair in distance_thresholds:
                            if self.are_connected(atom, other_atom, distance_thresholds[pair]):
                                counts[other_atom.element] += 1
                                counted_pairs.add((atom.atom_id, other_atom.atom_id))
                                counted_pairs.add((other_atom.atom_id, atom.atom_id))
                total_coordination = 0
                for neighbor, count in counts.items():
                    coordination_numbers[(atom.element, neighbor)].append(count)
                    total_coordination += count
                total_coordination_numbers[atom.element].append(total_coordination)
        
        # Ensure that atoms with zero neighbors are included
        for target in target_elements:
            for neighbor in neighbor_elements:
                if (target, neighbor) not in coordination_numbers:
                    coordination_numbers[(target, neighbor)] = []
                for atom in self.core_atoms:
                    if atom.element == target:
                        if not any(self.are_connected(atom, other_atom, distance_thresholds[(target, neighbor)]) for other_atom in self.core_atoms + self.shell_atoms if other_atom.element == neighbor):
                            coordination_numbers[(target, neighbor)].append(0)
        for target in target_elements:
            for atom in self.core_atoms:
                if atom.element == target:
                    if not any(self.are_connected(atom, other_atom, distance_thresholds[(target, neighbor)]) for neighbor in neighbor_elements for other_atom in self.core_atoms + self.shell_atoms if other_atom.element == neighbor):
                        total_coordination_numbers[target].append(0)
    
        coordination_stats = {}
        total_counts = []

        for pair, counts in coordination_numbers.items():
            avg = np.mean(counts)
            std = np.std(counts)
            coordination_stats[pair] = (avg, std)
            total_counts.extend(counts)

        total_coordination_counts = []
        for target, counts in total_coordination_numbers.items():
            total_coordination_counts.extend(counts)

        total_avg = np.mean(total_coordination_counts)
        total_std = np.std(total_coordination_counts)
        total_stats = {"total_avg": total_avg, "total_std": total_std}

        return coordination_stats, total_stats

    def calculate_bond_lengths_within_network(self, element_pairs_with_cutoff):
        """
        Calculates the bond lengths within each cluster network for specified element pairs with individual cutoffs.

        Parameters:
        - element_pairs_with_cutoff (list): List of tuples specifying element pairs with cutoff, e.g., [('Pb', 'I', 3.8), ('I', 'I', 7.6)]

        Returns:
        - network_bond_lengths (dict): Dictionary with network IDs as keys and bond lengths as values.
        """
        network_bond_lengths = defaultdict(list)

        for atom in self.core_atoms:
            network_id = atom.network_id
            if network_id is None:
                continue

            for other_atom in self.core_atoms:
                if atom != other_atom and atom.network_id == other_atom.network_id:
                    pair = tuple(sorted((atom.element, other_atom.element)))
                    # Find the cutoff for the current pair
                    cutoff = next((cutoff for p, q, cutoff in element_pairs_with_cutoff if tuple(sorted((p, q))) == pair), None)
                    if cutoff:
                        distance = np.linalg.norm(np.array(atom.coordinates) - np.array(other_atom.coordinates))
                        if distance <= cutoff:
                            network_bond_lengths[network_id].append(distance)
                            print(f"Found bond length {distance:.2f} Å between {pair[0]} and {pair[1]} in network {network_id}")

        if not any(network_bond_lengths.values()):
            print("Warning: No bond lengths found for the specified pairs within their respective cutoffs.")
            
        return network_bond_lengths

    def calculate_bond_angles_within_network(self, element_triplets_with_cutoff):
        """
        Calculates the bond angles within each cluster network for specified element triplets with a cutoff.

        Parameters:
        - element_triplets_with_cutoff (list): List of tuples specifying element triplets with cutoff, e.g., [('I', 'Pb', 'I', 3.8)]

        Returns:
        - network_bond_angles (dict): Dictionary with network IDs as keys and bond angles as values.
        """
        def angle_between(v1, v2):
            v1_u = v1 / np.linalg.norm(v1)
            v2_u = v2 / np.linalg.norm(v2)
            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

        network_bond_angles = defaultdict(list)

        for (a, b, c, cutoff) in element_triplets_with_cutoff:
            for atom_b in self.core_atoms:
                if atom_b.element != b:
                    continue
                network_id = atom_b.network_id
                if network_id is None:
                    continue

                # Find all atoms that are within the cutoff distance
                adjacent_atoms = []
                for atom_a in self.core_atoms:
                    if atom_a.network_id == network_id and atom_a.element == a:
                        distance = np.linalg.norm(np.array(atom_b.coordinates) - np.array(atom_a.coordinates))
                        if distance <= cutoff:
                            adjacent_atoms.append((atom_a, distance))
                
                for atom_c in self.core_atoms:
                    if atom_c.network_id == network_id and atom_c.element == c and atom_c != atom_b:
                        distance = np.linalg.norm(np.array(atom_b.coordinates) - np.array(atom_c.coordinates))
                        if distance <= cutoff:
                            for atom_a, distance_a in adjacent_atoms:
                                if atom_a != atom_c:
                                    v1 = np.array(atom_a.coordinates) - np.array(atom_b.coordinates)
                                    v2 = np.array(atom_c.coordinates) - np.array(atom_b.coordinates)
                                    angle = np.degrees(angle_between(v1, v2))
                                    network_bond_angles[network_id].append(angle)
                                    print(f"Found bond angle {angle:.2f}° for triplet {a}-{b}-{c} in network {network_id}")

        if not any(network_bond_angles.values()):
            print("Warning: No bond angles found for the specified triplets within their respective cutoffs.")
            
        return network_bond_angles

    ## -- Plotting & Visualization
    def calculate_and_plot_distributions(self, element_pairs_with_cutoff, element_triplets_with_cutoff):
        """
        Calculates and plots the distributions of bond lengths and angles across all cluster networks.

        Parameters:
        - element_pairs_with_cutoff (list): List of tuples specifying element pairs with cutoff for bond length calculations.
        - element_triplets_with_cutoff (list): List of tuples specifying element triplets with cutoff for bond angle calculations.
        """
        # Calculate bond lengths and angles for each network
        network_bond_lengths = self.calculate_bond_lengths_within_network(element_pairs_with_cutoff)
        network_bond_angles = self.calculate_bond_angles_within_network(element_triplets_with_cutoff)

        # Aggregate results across all networks
        all_bond_lengths = defaultdict(list)
        all_bond_angles = defaultdict(list)

        # Handle bond lengths
        for (p, q, cutoff) in element_pairs_with_cutoff:
            pair = (p, q)
            for network_id, lengths in network_bond_lengths.items():
                all_bond_lengths[pair].extend(lengths)

        # Handle bond angles
        for (a, b, c, cutoff) in element_triplets_with_cutoff:
            triplet = (a, b, c)
            for network_id, angles in network_bond_angles.items():
                all_bond_angles[triplet].extend(angles)

        # Plot bond length distribution
        if any(all_bond_lengths.values()):
            plt.figure()
            for (p, q), lengths in all_bond_lengths.items():
                label = f"{p} - {q} Distances"
                plt.hist(lengths, bins=30, alpha=0.75, label=label)
                average_length = np.mean(lengths)
                median_length = np.median(lengths)
                plt.axvline(average_length, color='r', linestyle='dashed', linewidth=1)
                plt.axvline(median_length, color='b', linestyle='dashed', linewidth=1)
                # Add text for average and median
                plt.text(average_length, plt.ylim()[1] * 0.9, f'Avg: {average_length:.2f} Å', color='r', ha='center')
                plt.text(median_length, plt.ylim()[1] * 0.85, f'Median: {median_length:.2f} Å', color='b', ha='center')

            plt.xlabel('Bond Length (Å)')
            plt.ylabel('Frequency')
            plt.title('Bond Length Distribution Across Networks')
            plt.legend()
            plt.show()
        else:
            print("Warning: No bond lengths found for the specified pairs within their respective cutoffs.")

        # Plot bond angle distribution
        if any(all_bond_angles.values()):
            plt.figure()
            for (a, b, c), angles in all_bond_angles.items():
                label = f"{a} - {b} - {c} Angles"
                plt.hist(angles, bins=30, alpha=0.75, label=label)
                average_angle = np.mean(angles)
                median_angle = np.median(angles)
                plt.axvline(average_angle, color='r', linestyle='dashed', linewidth=1)
                plt.axvline(median_angle, color='b', linestyle='dashed', linewidth=1)
                # Add text for average and median
                plt.text(average_angle, plt.ylim()[1] * 0.9, f'Avg: {average_angle:.2f}°', color='r', ha='center')
                plt.text(median_angle, plt.ylim()[1] * 0.85, f'Median: {median_angle:.2f}°', color='b', ha='center')

            plt.xlabel('Bond Angle (degrees)')
            plt.ylabel('Frequency')
            plt.title('Bond Angle Distribution Across Networks')
            plt.legend()
            plt.show()
        else:
            print("Warning: No bond angles found for the specified triplets within their respective cutoffs.")

    def print_coordination_numbers(self, coordination_stats, total_stats):
        print("Coordination Numbers:")
        print("Target-Neighbor   Average   Standard Deviation")
        print("---------------------------------------------")
        for pair, stats in coordination_stats.items():
            target, neighbor = pair
            avg, std = stats
            print(f"{target}-{neighbor:<15} {avg:<8.2f} {std:<18.2f}")
        print("---------------------------------------------")
        print(f"Total              {total_stats['total_avg']:<8.2f} {total_stats['total_std']:<18.2f}")

    def visualize_networks(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        network_ids = list(set(atom.network_id for atom in self.core_atoms if atom.network_id is not None))
        color_map = plt.cm.get_cmap('hsv', len(network_ids))

        for i, network_id in enumerate(network_ids):
            color = color_map(i)
            for atom in self.core_atoms:
                if atom.network_id == network_id:
                    ax.scatter(atom.coordinates[0], atom.coordinates[1], atom.coordinates[2], color=color, label=network_id)
                    connected_atoms = self.get_connected_atoms(atom)
                    for connected_atom in connected_atoms:
                        if connected_atom.network_id == network_id:
                            ax.plot([atom.coordinates[0], connected_atom.coordinates[0]],
                                    [atom.coordinates[1], connected_atom.coordinates[1]],
                                    [atom.coordinates[2], connected_atom.coordinates[2]],
                                    color=color)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = list(dict.fromkeys(labels))  # Remove duplicates while preserving order
        unique_handles = [handles[labels.index(label)] for label in unique_labels]
        ax.legend(unique_handles, unique_labels)
        plt.show()
        
    def calculate_and_plot_heatmap(self, central_element, x_pair, y_pair, x_range, y_range, distance_cutoffs):
        """
        Calculates and plots a 2D heatmap of coordination numbers for a given central atom type and two coordination pairs.

        Parameters:
        - central_element (str): The element symbol of the central atom (e.g., 'Pb').
        - x_pair (tuple): The pair for the x-axis coordination (e.g., ('Pb', 'I')).
        - y_pair (tuple): The pair for the y-axis coordination (e.g., ('Pb', 'O')).
        - x_range (tuple): The range for the x-axis coordination number (min, max) inclusive.
        - y_range (tuple): The range for the y-axis coordination number (min, max) inclusive.
        - distance_cutoffs (dict): The cutoff distances for counting atoms in the coordination sphere.
        """
        # Initialize a 2D grid for the heatmap
        heatmap_data = np.zeros((y_range[1] - y_range[0] + 1, x_range[1] - x_range[0] + 1))

        # Determine sorted pairs for consistent dictionary lookups
        x_pair_sorted = tuple(sorted(x_pair))
        y_pair_sorted = tuple(sorted(y_pair))

        for atom in self.core_atoms:
            if atom.element != central_element:
                continue

            # Initialize coordination numbers for this atom
            x_cn = 0
            y_cn = 0

            for other_atom in self.core_atoms + self.shell_atoms:
                if other_atom == atom:
                    continue
                distance = np.linalg.norm(np.array(atom.coordinates) - np.array(other_atom.coordinates))
                
                # Check for x_pair coordination
                if other_atom.element == x_pair[1]:
                    try:
                        cutoff_distance = distance_cutoffs[x_pair] if x_pair in distance_cutoffs else distance_cutoffs[x_pair_sorted]
                        if distance <= cutoff_distance:
                            x_cn += 1
                    except KeyError:
                        raise KeyError(f"Distance cutoff not found for pair {x_pair_sorted}")

                # Check for y_pair coordination
                if other_atom.element == y_pair[1]:
                    try:
                        cutoff_distance = distance_cutoffs[y_pair] if y_pair in distance_cutoffs else distance_cutoffs[y_pair_sorted]
                        if distance <= cutoff_distance:
                            y_cn += 1
                    except KeyError:
                        raise KeyError(f"Distance cutoff not found for pair {y_pair_sorted}")

            # Ensure the coordination numbers are within specified ranges and update the heatmap
            if x_range[0] <= x_cn <= x_range[1] and y_range[0] <= y_cn <= y_range[1]:
                heatmap_data[y_cn - y_range[0], x_cn - x_range[0]] += 1

        # Generate dynamic labels and title
        x_label = f'CN: {x_pair[0]} - {x_pair[1]}'
        y_label = f'CN: {y_pair[0]} - {y_pair[1]}'
        title = f'{x_pair[0]} - {x_pair[1]} v. {y_pair[0]} - {y_pair[1]} Coordination Numbers'

        # Plot the heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(heatmap_data, annot=True, fmt="g", cmap="YlGnBu", cbar_kws={'label': 'Count'})
        plt.xlabel(x_label, fontsize=14)  # Enlarging the x-axis label
        plt.ylabel(y_label, fontsize=14)  # Enlarging the y-axis label
        plt.title(title)
        plt.xticks(ticks=np.arange(x_range[1] - x_range[0] + 1) + 0.5, labels=np.arange(x_range[0], x_range[1] + 1), rotation=0)
        plt.yticks(ticks=np.arange(y_range[1] - y_range[0] + 1) + 0.5, labels=np.arange(y_range[0], y_range[1] + 1), rotation=0)
        plt.show()
