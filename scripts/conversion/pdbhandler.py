import re

class PDBFileHandler:
    def __init__(self, filepath, core_residue_names, shell_residue_names):
        """
        Initializes a PDBFileHandler instance.
        
        Parameters:
        - filepath (str): Path to the PDB file.
        - core_residue_names (list): List of core residue names.
        - shell_residue_names (list): List of shell residue names.
        """
        self.filepath = filepath
        self.core_residue_names = core_residue_names
        self.shell_residue_names = shell_residue_names
        self.core_atoms = []
        self.shell_atoms = []
        self.read_pdb_file()

    def read_pdb_file(self):
        """
        Reads a PDB file and parses the atom information.
        Separates core and shell atoms based on residue names.
        """
        with open(self.filepath, 'r') as file:
            for line in file:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    atom = self.parse_atom_line(line)
                    if atom.residue_name in self.core_residue_names:
                        self.core_atoms.append(atom)
                    elif atom.residue_name in self.shell_residue_names:
                        self.shell_atoms.append(atom)

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

    def update_residue_names(self, updated_atoms):
        """
        Updates the residue names of atoms based on the provided list.

        Parameters:
        - updated_atoms (list): List of Atom objects with updated residue names.
        """
        atom_map = {atom.atom_id: atom for atom in self.core_atoms + self.shell_atoms}
        for updated_atom in updated_atoms:
            if updated_atom.atom_id in atom_map:
                atom_map[updated_atom.atom_id].residue_name = updated_atom.residue_name

    def write_pdb_file(self, output_path, atoms):
        """
        Writes the atoms to a new PDB file with the correct residue names and numbers,
        sorted by atom_id in ascending order.
        
        Parameters:
        - output_path (str): Path to the output PDB file.
        - atoms (list): List of Atom objects to write.
        """
        # Sort atoms by atom_id in ascending order
        atoms_sorted = sorted(atoms, key=lambda atom: atom.atom_id)
        
        with open(output_path, 'w') as file:
            for atom in atoms_sorted:
                file.write(
                    f"ATOM  {atom.atom_id:5d}  {atom.atom_name:<4}{atom.residue_name:<3} X{atom.residue_number:4d}    "
                    f"{atom.coordinates[0]:8.3f}{atom.coordinates[1]:8.3f}{atom.coordinates[2]:8.3f}  1.00  0.00          {atom.element:<2}\n"
                )

    def get_atom_details(self):
        """
        Returns a list of tuples containing atom details (ID, name, element, coordinates).
        
        Returns:
        - details (list): List of tuples containing atom details.
        """
        details = [(atom.atom_id, atom.atom_name, atom.element, atom.coordinates) for atom in self.core_atoms + self.shell_atoms]
        return details

    def print_atom_details(self):
        """
        Prints the details of each atom in a readable format.
        """
        for atom in self.core_atoms + self.shell_atoms:
            print(f"Atom ID: {atom.atom_id}, Name: {atom.atom_name}, Element: {atom.element}, Coordinates: {atom.coordinates}")

class Atom:
    def __init__(self, atom_id, atom_name, residue_name, residue_number, x, y, z, element):
        """
        Initializes an Atom instance.
        
        Parameters:
        - atom_id (int): Unique identifier for the atom.
        - atom_name (str): Unique atom identifier.
        - residue_name (str): Residue name for the atom.
        - residue_number (int): Residue number for the atom.
        - x (float): X coordinate.
        - y (float): Y coordinate.
        - z (float): Z coordinate.
        - element (str): Element type.
        """
        self.atom_id = atom_id
        self.atom_name = atom_name
        self.residue_name = residue_name
        self.residue_number = residue_number
        self.coordinates = (x, y, z)
        self.element = element
        self.network_id = None

    def __repr__(self):
        return f"Atom({self.atom_id}, {self.atom_name}, {self.residue_name}, {self.residue_number}, {self.coordinates}, {self.element})"
