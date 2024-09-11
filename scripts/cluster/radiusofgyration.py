import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from scipy.linalg import eigh
from mendeleev import element  # To fetch ionic radii

class RadiusOfGyrationCalculator:
    def __init__(self, atom_positions=None, atom_elements=None, atom_charges=None, electron_lookup=None):
        """
        Initialize with atom positions, elements, and formal charges.
        Reuse the electron lookup table if provided, otherwise create a new one.
        """
        if atom_positions is not None and atom_elements is not None and atom_charges is not None:
            self.atom_positions = np.array(atom_positions)
            self.atom_elements = atom_elements
            self.atom_charges = atom_charges
            self.electron_lookup = electron_lookup if electron_lookup is not None else {}
            self._update_electron_lookup()
            self.electron_weights = self._assign_electron_counts()

    def load_from_pdb(self, pdb_handler, atom_charges):
        """
        Load atom positions and elements from a PDB handler directly.
        
        :param pdb_handler: PDBFileHandler object containing all atoms.
        :param atom_charges: Dictionary with element as key and charge as value.
        """
        all_atoms = pdb_handler.core_atoms + pdb_handler.shell_atoms
        
        self.atom_positions = np.array([atom.coordinates for atom in all_atoms])
        self.atom_elements = [atom.element for atom in all_atoms]
        self.atom_charges = [atom_charges[atom.element] for atom in all_atoms]
        
        self._update_electron_lookup()  # Update the lookup table with any new elements
        self.electron_weights = self._assign_electron_counts()

    def _update_electron_lookup(self):
        """
        Update the electron lookup table with any new elements found in the atom list.
        """
        unique_elements = set(self.atom_elements)
        for element in unique_elements:
            if element not in self.electron_lookup:
                element_data = mendeleev.element(element)
                electron_count = element_data.electrons
                formal_charge = self.atom_charges[self.atom_elements.index(element)]
                adjusted_electron_count = electron_count - formal_charge
                self.electron_lookup[element] = adjusted_electron_count

    def _assign_electron_counts(self):
        """
        Assign electron counts to each coordinate position based on the element and charge.
        """
        electron_weights = np.array([self.electron_lookup[element] for element in self.atom_elements])
        return electron_weights

    def calculate_center_of_mass(self):
        total_electrons = np.sum(self.electron_weights)
        center_of_mass = np.sum(self.atom_positions * self.electron_weights[:, np.newaxis], axis=0) / total_electrons
        return center_of_mass

    def calculate_radius_of_gyration(self):
        center_of_mass = self.calculate_center_of_mass()
        distances_squared = np.sum((self.atom_positions - center_of_mass) ** 2, axis=1)
        Rg_squared = np.sum(self.electron_weights * distances_squared) / np.sum(self.electron_weights)
        Rg = np.sqrt(Rg_squared)
        return Rg

    def calculate_volume(self, method='sphere'):
        if method == 'sphere':
            Rg = self.calculate_radius_of_gyration()
            volume = (4/3) * np.pi * (Rg ** 3)
            return volume
        
        elif method == 'ellipsoid':
            center_of_mass = self.calculate_center_of_mass()
            adjusted_positions = self.atom_positions - center_of_mass

            # Inertia tensor calculation
            inertia_tensor = np.zeros((3, 3))
            for weight, pos in zip(self.electron_weights, adjusted_positions):
                inertia_tensor += weight * (np.dot(pos[:, np.newaxis], pos[np.newaxis, :]) - np.eye(3) * np.sum(pos**2))

            # Calculate eigenvalues
            eigenvalues, _ = eigh(inertia_tensor)

            # Ensure eigenvalues are positive
            eigenvalues = np.abs(eigenvalues)
            
            # Correct normalization for semi-axes
            semi_axes = np.sqrt(5 * eigenvalues / np.sum(self.electron_weights))
            a, b, c = semi_axes

            # Calculate radii of gyration for each axis
            Rgx = np.sqrt((b**2 + c**2) / 5)
            Rgy = np.sqrt((a**2 + c**2) / 5)
            Rgz = np.sqrt((a**2 + b**2) / 5)

            # Calculate volume of the ellipsoid
            # volume = (4/3) * np.pi * a * b * c
            volume = (4/3) * np.pi * Rgx * Rgy * Rgz
            return volume, Rgx, Rgy, Rgz
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
'''import numpy as np
import mendeleev
from scipy.linalg import eigh

class RadiusOfGyrationCalculator:
    def __init__(self, atom_positions=None, atom_elements=None, atom_charges=None):
        if atom_positions is not None and atom_elements is not None and atom_charges is not None:
            self.atom_positions = np.array(atom_positions)
            self.atom_elements = atom_elements
            self.atom_charges = atom_charges
            self.electron_lookup = self._calculate_electron_counts()
            self.electron_weights = self._assign_electron_counts()

    def load_from_pdb(self, pdb_file, core_residue_names, shell_residue_names, atom_charges):
        pdb_handler = PDBFileHandler(pdb_file, core_residue_names, shell_residue_names)
        all_atoms = pdb_handler.core_atoms + pdb_handler.shell_atoms
        self.atom_positions = np.array([atom.coordinates for atom in all_atoms])
        self.atom_elements = [atom.element for atom in all_atoms]
        self.atom_charges = [atom_charges[atom.element] for atom in all_atoms]
        self.electron_lookup = self._calculate_electron_counts()
        self.electron_weights = self._assign_electron_counts()

    def _calculate_electron_counts(self):
        unique_elements = set(self.atom_elements)
        electron_lookup = {}
        for element in unique_elements:
            element_data = mendeleev.element(element)
            electron_count = element_data.electrons
            formal_charge = self.atom_charges[self.atom_elements.index(element)]
            adjusted_electron_count = electron_count - formal_charge
            electron_lookup[element] = adjusted_electron_count
        return electron_lookup

    def _assign_electron_counts(self):
        electron_weights = np.array([self.electron_lookup[element] for element in self.atom_elements])
        return electron_weights

    def calculate_center_of_mass(self):
        total_electrons = np.sum(self.electron_weights)
        center_of_mass = np.sum(self.atom_positions * self.electron_weights[:, np.newaxis], axis=0) / total_electrons
        return center_of_mass

    def calculate_radius_of_gyration(self):
        center_of_mass = self.calculate_center_of_mass()
        distances_squared = np.sum((self.atom_positions - center_of_mass) ** 2, axis=1)
        Rg_squared = np.sum(self.electron_weights * distances_squared) / np.sum(self.electron_weights)
        Rg = np.sqrt(Rg_squared)
        return Rg

    def calculate_volume(self, method='sphere'):
        if method == 'sphere':
            Rg = self.calculate_radius_of_gyration()
            volume = (4/3) * np.pi * (Rg ** 3)
            return volume
        
        elif method == 'ellipsoid':
                center_of_mass = self.calculate_center_of_mass()
                adjusted_positions = self.atom_positions - center_of_mass

                # Inertia tensor calculation
                inertia_tensor = np.zeros((3, 3))
                for weight, pos in zip(self.electron_weights, adjusted_positions):
                    inertia_tensor += weight * (np.dot(pos[:, np.newaxis], pos[np.newaxis, :]) - np.eye(3) * np.sum(pos**2))

                # Calculate eigenvalues
                eigenvalues, _ = eigh(inertia_tensor)

                # Ensure eigenvalues are positive
                eigenvalues = np.abs(eigenvalues)
                
                # Correct normalization for semi-axes
                semi_axes = np.sqrt(5 * eigenvalues / np.sum(self.electron_weights))
                a, b, c = semi_axes

                # Calculate radii of gyration for each axis
                Rgx = np.sqrt((b**2 + c**2) / 5)
                Rgy = np.sqrt((a**2 + c**2) / 5)
                Rgz = np.sqrt((a**2 + b**2) / 5)

                # Calculate volume of the ellipsoid
                volume = (4/3) * np.pi * Rgx * Rgy * Rgz
                return volume, Rgx, Rgy, Rgz
'''