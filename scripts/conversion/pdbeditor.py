class PDBEditor:
    
    @staticmethod
    def update_or_add_residue_names(input_pdb, output_pdb, residue_mapping):
        """
        Update or add residue names in a PDB file.

        Parameters:
        - input_pdb: str, path to the input PDB file
        - output_pdb: str, path to the output PDB file
        - residue_mapping: dict, mapping of old residue names to new residue names
        """
        with open(input_pdb, 'r') as infile, open(output_pdb, 'w') as outfile:
            for line in infile:
                if line.startswith(("ATOM", "HETATM")):
                    residue_name = line[17:20].strip()
                    new_residue_name = residue_mapping.get(residue_name, residue_name)
                    line = line[:17] + new_residue_name.ljust(3) + line[20:]
                outfile.write(line)

    @staticmethod
    def add_residue_name(input_pdb, output_pdb, default_residue_name="DMS"):
        """
        Add a default residue name to lines in a PDB file that lack one.

        Parameters:
        - input_pdb: str, path to the input PDB file
        - output_pdb: str, path to the output PDB file
        - default_residue_name: str, default residue name to add if missing (default is 'DMSO')
        """
        with open(input_pdb, 'r') as infile, open(output_pdb, 'w') as outfile:
            for line in infile:
                if line.startswith(("ATOM", "HETATM")):
                    current_residue_name = line[17:20].strip()
                    if not current_residue_name:
                        line = line[:17] + default_residue_name.ljust(3) + line[20:]
                outfile.write(line)

    @staticmethod
    def read_residue_names(pdb_file):
        """
        Read and return residue names from a PDB file.

        Parameters:
        - pdb_file: str, path to the PDB file
        
        Returns:
        - set: A set of residue names found in the PDB file.
        """
        residue_names = set()  # Using a set to avoid duplicates

        with open(pdb_file, 'r') as file:
            for line in file:
                if line.startswith(("ATOM", "HETATM")):
                    residue_name = line[17:20].strip()
                    residue_names.add(residue_name)

        return residue_names

    @staticmethod
    def rename_hydrogens(input_pdb, output_pdb):
        """
        Rename hydrogen atoms in a PDB file based on their closest carbon atom.

        Parameters:
        - input_pdb: str, path to the input PDB file
        - output_pdb: str, path to the output PDB file
        """
        with open(input_pdb, 'r') as file:
            lines = file.readlines()

        atom_coords = {}
        carbon_hydrogen_map = {}
        carbon_order = []
        hydrogen_counter = {}

        # First pass to identify all atoms and store their coordinates
        for line in lines:
            if line.startswith('ATOM'):
                atom_index = int(line[6:11].strip())
                atom_name = line[12:16].strip()
                atom_type = line[76:78].strip()
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())

                atom_coords[atom_index] = (atom_name, atom_type, x, y, z)

                if atom_type == 'C':
                    carbon_hydrogen_map[atom_index] = []
                    carbon_order.append(atom_index)
                    hydrogen_counter[atom_index] = 0

        # Second pass to associate hydrogens with their closest carbons
        for atom_index, (atom_name, atom_type, x, y, z) in atom_coords.items():
            if atom_type == 'H':
                closest_carbon = None
                min_distance = float('inf')
                for carbon_index, (carbon_name, carbon_type, cx, cy, cz) in atom_coords.items():
                    if carbon_type == 'C':
                        distance = ((x - cx)**2 + (y - cy)**2 + (z - cz)**2)**0.5
                        if distance < min_distance:
                            min_distance = distance
                            closest_carbon = carbon_index

                if closest_carbon is not None:
                    carbon_hydrogen_map[closest_carbon].append(atom_index)

        # Third pass to rename hydrogens
        with open(output_pdb, 'w') as file:
            for line in lines:
                if line.startswith('ATOM'):
                    atom_index = int(line[6:11].strip())
                    atom_name = line[12:16].strip()
                    atom_type = line[76:78].strip()

                    if atom_type == 'H':
                        for i, carbon_index in enumerate(carbon_order, start=1):
                            if atom_index in carbon_hydrogen_map[carbon_index]:
                                hydrogen_counter[carbon_index] += 1
                                new_name = f"H{i}{hydrogen_counter[carbon_index]}"
                                line = line[:12] + new_name.ljust(4) + line[16:]
                                break
                file.write(line)

    @staticmethod
    def remove_residue(input_pdb, output_pdb, residue_name):
        """
        Remove all atoms corresponding to a specified residue name from a PDB file.

        Parameters:
        - input_pdb: str, path to the input PDB file
        - output_pdb: str, path to the output PDB file
        - residue_name: str, residue name to remove from the PDB file
        """
        with open(input_pdb, 'r') as infile, open(output_pdb, 'w') as outfile:
            for line in infile:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    # Extract the residue name from the line
                    current_residue_name = line[17:20].strip()
                    # If the current residue name is not the one to be removed, write the line to the output file
                    if current_residue_name != residue_name:
                        outfile.write(line)
                else:
                    # Write all non-ATOM and non-HETATM lines to the output file
                    outfile.write(line)

    @staticmethod
    def map_atom_pairs(input_pdb, output_pdb, residue_name, atom1, atom2, pair_label="P"):
        """
        Generalized method to map and rename atom pairs (e.g., C-H bonds) within a residue.
        
        Parameters:
        - input_pdb: str, path to the input PDB file
        - output_pdb: str, path to the output PDB file
        - residue_name: str, the residue name to look for (e.g., "DMS")
        - atom1: str, the first atom in the pair (e.g., "C")
        - atom2: str, the second atom in the pair (e.g., "H")
        - pair_label: str, label to add to the renamed pairs (default "P")
        """
        with open(input_pdb, 'r') as file:
            lines = file.readlines()

        atom_coords = {}
        pair_map = {}
        pair_order = []
        pair_counter = {}

        # First pass to identify all atoms and store their coordinates
        for line in lines:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                current_residue = line[17:20].strip()
                if current_residue == residue_name:
                    atom_index = int(line[6:11].strip())
                    atom_name = line[12:16].strip()
                    atom_type = line[76:78].strip()
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())

                    atom_coords[atom_index] = (atom_name, atom_type, x, y, z)

                    if atom_type == atom1:
                        pair_map[atom_index] = []
                        pair_order.append(atom_index)
                        pair_counter[atom_index] = 0

        # Second pass to associate atom2 (e.g., hydrogen) with the closest atom1 (e.g., carbon)
        for atom_index, (atom_name, atom_type, x, y, z) in atom_coords.items():
            if atom_type == atom2:
                closest_atom1 = None
                min_distance = float('inf')
                for atom1_index, (atom1_name, atom1_type, cx, cy, cz) in atom_coords.items():
                    if atom1_type == atom1:
                        distance = ((x - cx)**2 + (y - cy)**2 + (z - cz)**2)**0.5
                        if distance < min_distance:
                            min_distance = distance
                            closest_atom1 = atom1_index

                if closest_atom1 is not None:
                    pair_map[closest_atom1].append(atom_index)

        # Third pass to rename atom pairs
        with open(output_pdb, 'w') as file:
            for line in lines:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    atom_index = int(line[6:11].strip())
                    atom_name = line[12:16].strip()
                    atom_type = line[76:78].strip()

                    if atom_index in pair_map:
                        # Renaming the pair
                        pair_counter[atom_index] += 1
                        new_name = f"{pair_label}{pair_counter[atom_index]}"
                        line = line[:12] + new_name.ljust(4) + line[16:]

                file.write(line)

        print(f"Renamed atom pairs ({atom1}-{atom2}) for residue {residue_name} and saved to {output_pdb}.")
