import numpy as np
from typing import TypedDict, Optional, Dict

class VolumeEstimator:
    """
    A class to estimate the volume fractions of solute atoms and solvent molecules in a solution.
    
    Attributes:
        mass_solute (float): Mass of the solute (PbI2) in grams.
        mass_solvent (float): Mass of the solvent (DMSO) in grams.
        density_solution (float): Density of the solution in g/cm³.
        density_neat_solvent (float): Density of the neat solvent (DMSO) in g/cm³.
        molar_mass_solvent (float): Molar mass of the solvent (DMSO) in g/mol.
        molar_mass_solute (float): Molar mass of the solute (PbI2) in g/mol.
        ionic_radii (dict): Ionic radii of solute atoms in angstroms, e.g., {'Pb2+': 1.19, 'I-': 2.20}.
        stoichiometry (dict): Stoichiometric coefficients of solute atoms, e.g., {'Pb2+': 1, 'I-': 2}.
    """
    
    def __init__(self, 
                 mass_solute,          # g
                 mass_solvent,         # g
                 density_solution,     # g/cm³
                 density_neat_solvent, # g/cm³
                 molar_mass_solvent,   # g/mol
                 molar_mass_solute,    # g/mol
                 ionic_radii,          # dict, e.g., {'Pb2+': 1.19, 'I-': 2.20}
                 stoichiometry         # dict, e.g., {'Pb2+':1, 'I-':2}
                ):
        """
        Initializes the VolumeEstimator with the given parameters.
        """
        self.mass_solute = mass_solute
        self.mass_solvent = mass_solvent
        self.density_solution = density_solution
        self.density_neat_solvent = density_neat_solvent
        self.molar_mass_solvent = molar_mass_solvent
        self.molar_mass_solute = molar_mass_solute
        self.ionic_radii = ionic_radii
        self.stoichiometry = stoichiometry
        self.N_A = 6.022e23  # Avogadro's Number in molecules/mol
        
    def calculate_total_mass(self):
        """
        Calculates the total mass of the solution.
        
        Returns:
            float: Total mass in grams (g).
        """
        m_total = self.mass_solute + self.mass_solvent  # grams
        return m_total
    
    def calculate_total_volume(self, m_total):
        """
        Calculates the total volume of the solution using its density.
        
        Parameters:
            m_total (float): Total mass of the solution in grams (g).
        
        Returns:
            float: Total volume in cubic centimeters (cm³).
        """
        V_solution_cm3 = m_total / self.density_solution  # cm³
        return V_solution_cm3
    
    def calculate_moles(self):
        """
        Calculates the number of moles of solute and solvent.
        
        Returns:
            tuple: (n_solute, n_solvent) in moles (mol).
        """
        n_solute = self.mass_solute / self.molar_mass_solute  # mol
        n_solvent = self.mass_solvent / self.molar_mass_solvent  # mol
        return n_solute, n_solvent
    
    def calculate_number_of_atoms(self, n_solute):
        """
        Calculates the number of solute units and individual atoms.
        
        Parameters:
            n_solute (float): Number of moles of solute.
        
        Returns:
            dict: Number of atoms for each solute ion type.
        """
        num_solute_units = n_solute * self.N_A  # formula units
        num_atoms = {}
        for atom, stoich in self.stoichiometry.items():
            num_atoms[atom] = num_solute_units * stoich
        return num_atoms
    
    def calculate_number_of_solvent_molecules(self, n_solvent):
        """
        Calculates the number of solvent molecules.
        
        Parameters:
            n_solvent (float): Number of moles of solvent.
        
        Returns:
            float: Number of solvent molecules.
        """
        num_solvent_molecules = n_solvent * self.N_A  # molecules
        return num_solvent_molecules
    
    def calculate_volume_per_solvent_molecule(self):
        """
        Calculates the volume occupied by a single solvent molecule in the neat solvent.
        
        Returns:
            float: Volume per solvent molecule in cubic centimeters (cm³/molecule).
        """
        V_solvent_cm3_per_molecule = self.molar_mass_solvent / (self.density_neat_solvent * self.N_A)  # cm³/molecule
        return V_solvent_cm3_per_molecule
    
    def calculate_total_solvent_volume(self, num_solvent_molecules, V_solvent_cm3_per_molecule):
        """
        Calculates the total volume occupied by solvent molecules in the solution.
        
        Parameters:
            num_solvent_molecules (float): Number of solvent molecules.
            V_solvent_cm3_per_molecule (float): Volume per solvent molecule in cm³/molecule.
        
        Returns:
            float: Total solvent volume in cubic centimeters (cm³).
        """
        V_solvent_total_cm3 = num_solvent_molecules * V_solvent_cm3_per_molecule  # cm³
        return V_solvent_total_cm3
    
    def calculate_volume_fraction_solvent(self, V_solvent_total_cm3, V_solution_cm3):
        """
        Calculates the volume fraction occupied by solvent molecules.
        
        Parameters:
            V_solvent_total_cm3 (float): Total solvent volume in cm³.
            V_solution_cm3 (float): Total solution volume in cm³.
        
        Returns:
            float: Volume fraction of solvent (unitless).
        """
        phi_solvent = V_solvent_total_cm3 / V_solution_cm3  # unitless
        return phi_solvent
    
    def calculate_volume_fraction_solute(self, phi_solvent):
        """
        Calculates the volume fraction occupied by solute atoms.
        
        Parameters:
            phi_solvent (float): Volume fraction of solvent (unitless).
        
        Returns:
            float: Volume fraction of solute (unitless).
        """
        phi_solute = 1 - phi_solvent  # unitless
        return phi_solute
    
    def calculate_relative_volume_contributions(self, phi_solute):
        """
        Calculates the relative volume contributions of each solute atom type based on stoichiometry and ionic radii.
        
        Parameters:
            phi_solute (float): Volume fraction of solute (unitless).
        
        Returns:
            dict: Relative volume fractions for each solute atom type (unitless).
        """
        sum_radii_cubes = 0
        for atom, stoich in self.stoichiometry.items():
            sum_radii_cubes += stoich * (self.ionic_radii[atom] ** 3)
        
        phi_atoms = {}
        for atom, stoich in self.stoichiometry.items():
            phi_atoms[atom] = phi_solute * (stoich * (self.ionic_radii[atom] ** 3)) / sum_radii_cubes
        return phi_atoms
    
    def allocate_absolute_volumes(self, phi_atoms, V_solution_cm3):
        """
        Allocates absolute volumes to each solute atom type based on their volume fractions.
        
        Parameters:
            phi_atoms (dict): Relative volume fractions for each solute atom type (unitless).
            V_solution_cm3 (float): Total solution volume in cm³.
        
        Returns:
            dict: Absolute volumes for each solute atom type in cm³.
        """
        V_atoms_cm3 = {}
        for atom, phi in phi_atoms.items():
            V_atoms_cm3[atom] = phi * V_solution_cm3  # cm³
        return V_atoms_cm3
    
    def convert_volumes_to_A3(self, V_solvent_cm3, V_atoms_cm3):
        """
        Converts volumes from cubic centimeters (cm³) to cubic angstroms (Å³).
        
        Parameters:
            V_solvent_cm3 (float): Total solvent volume in cm³.
            V_atoms_cm3 (dict): Absolute volumes for each solute atom type in cm³.
        
        Returns:
            dict: Volumes in cubic angstroms (Å³).
        """
        cm3_to_A3 = 1e24  # 1 cm³ = 1e24 Å³
        V_solvent_A3 = V_solvent_cm3 * cm3_to_A3  # Å³
        V_atoms_A3 = {}
        for atom, V in V_atoms_cm3.items():
            V_atoms_A3[atom] = V * cm3_to_A3  # Å³
        return V_solvent_A3, V_atoms_A3
    
    def validate_volume_conservation(self, phi_solvent, phi_solute, phi_atoms):
        """
        Validates that the sum of volume fractions approximates 1 (or 100%).
        
        Parameters:
            phi_solvent (float): Volume fraction of solvent (unitless).
            phi_solute (float): Volume fraction of solute (unitless).
            phi_atoms (dict): Relative volume fractions for each solute atom type (unitless).
        
        Returns:
            bool: True if the sum is approximately 1, False otherwise.
        """
        sum_phi = phi_solvent + phi_solute
        sum_phi_atoms = sum(phi_atoms.values())
        total = sum_phi_atoms + phi_solvent
        return np.isclose(total, 1.0, atol=1e-4)
    
    def estimate_volumes(self):
        """
        Main method to estimate the volumes of solvent molecules and solute atoms.
        
        Returns:
            dict: A dictionary containing the volumes in cubic angstroms (Å³) for solvent and each solute atom type.
                  Example:
                  {
                      'Solvent': 8.18e25,  # Å³
                      'Pb2+': 1.34e24,     # Å³
                      'I-': 1.70e25        # Å³
                  }
        """
        # Step 1: Calculate Total Mass of the Solution
        m_total = self.calculate_total_mass()
        print(f"Total Mass of Solution: {m_total} g")
        
        # Step 2: Calculate Total Volume of the Solution
        V_solution_cm3 = self.calculate_total_volume(m_total)
        print(f"Total Volume of Solution: {V_solution_cm3:.4f} cm³")
        
        # Step 3: Calculate Number of Moles of Solute and Solvent
        n_solute, n_solvent = self.calculate_moles()
        print(f"Moles of Solute (PbI2): {n_solute:.4f} mol")
        print(f"Moles of Solvent (DMSO): {n_solvent:.4f} mol")
        
        # Step 4: Calculate Number of Solute Units and Atoms
        num_atoms = self.calculate_number_of_atoms(n_solute)
        for atom, count in num_atoms.items():
            print(f"Number of {atom} Atoms: {count:.4e} atoms")
        
        # Step 5: Calculate Number of Solvent Molecules
        num_solvent_molecules = self.calculate_number_of_solvent_molecules(n_solvent)
        print(f"Number of Solvent (DMSO) Molecules: {num_solvent_molecules:.4e} molecules")
        
        # Step 6: Calculate Volume per Solvent Molecule
        V_solvent_cm3_per_molecule = self.calculate_volume_per_solvent_molecule()
        print(f"Volume per Solvent Molecule (DMSO): {V_solvent_cm3_per_molecule:.4e} cm³/molecule")
        
        # Step 7: Calculate Total Volume Occupied by Solvent Molecules
        V_solvent_total_cm3 = self.calculate_total_solvent_volume(num_solvent_molecules, V_solvent_cm3_per_molecule)
        print(f"Total Volume Occupied by Solvent (DMSO): {V_solvent_total_cm3:.4f} cm³")
        
        # Step 8: Calculate Volume Fraction Occupied by Solvent
        phi_solvent = self.calculate_volume_fraction_solvent(V_solvent_total_cm3, V_solution_cm3)
        phi_solvent_percent = phi_solvent * 100
        print(f"Volume Fraction of Solvent (DMSO): {phi_solvent_percent:.2f}%")
        
        # Step 9: Calculate Volume Fraction Occupied by Solute
        phi_solute = self.calculate_volume_fraction_solute(phi_solvent)
        phi_solute_percent = phi_solute * 100
        print(f"Volume Fraction of Solute: {phi_solute_percent:.2f}%")
        
        # Step 10: Calculate Relative Volume Contributions of Solute Atoms
        phi_atoms = self.calculate_relative_volume_contributions(phi_solute)
        for atom, phi in phi_atoms.items():
            phi_percent = phi * 100
            print(f"Volume Fraction of {atom}: {phi_percent:.2f}%")
        
        # Step 11: Allocate Absolute Volumes to Solute Atoms
        V_atoms_cm3 = self.allocate_absolute_volumes(phi_atoms, V_solution_cm3)
        for atom, V in V_atoms_cm3.items():
            print(f"Volume Occupied by {atom}: {V:.4f} cm³")
        
        # Step 12: Convert Volumes to Cubic Angstroms (Å³)
        V_solvent_A3, V_atoms_A3 = self.convert_volumes_to_A3(V_solvent_total_cm3, V_atoms_cm3)
        
        # Step 13: Validate Volume Conservation
        is_valid = self.validate_volume_conservation(phi_solvent, phi_solute, phi_atoms)
        if is_valid:
            print("Volume conservation validated: Sum of volume fractions is approximately 100%.")
        else:
            print("Volume conservation error: Sum of volume fractions does not approximate 100%.")
        
        # Step 14: Create Output Dictionary
        volumes = {
            'Solvent': V_solvent_A3,     # Å³
            'Pb2+': V_atoms_A3.get('Pb2+', 0),       # Å³
            'I-': V_atoms_A3.get('I-', 0)            # Å³
        }
        
        return volumes

# Define the BulkVolumeParams TypedDict
class BulkVolumeParams(TypedDict):
    mass_percent_solute: float
    density_solution: float
    density_neat_solvent: float
    molar_mass_solvent: float
    molar_mass_solute: float
    ionic_radii: Dict[str, float]
    stoichiometry: Dict[str, int]
    atomic_masses: Dict[str, float]
    solute_residues: Dict[str, str]
    solvent_name: str
    total_mass: Optional[float]  # Optional: defaults to 100 g if not specified

class BulkVolume:
    """
    A class to estimate the volume fractions of solute atoms and solvent molecules in a solution
    based on the mass percent of the solute, and to estimate the number of atoms/molecules
    within a cubic box of specified dimensions in angstroms (Å).
    """

    def __init__(self, 
                 mass_percent_solute,    # e.g., 10 for 10%
                 density_solution,       # g/cm³
                 density_neat_solvent,   # g/cm³
                 molar_mass_solvent,     # g/mol
                 molar_mass_solute,      # g/mol
                 ionic_radii,            # dict, e.g., {'Pb2+': 1.19, 'I-': 2.20}
                 stoichiometry,          # dict, e.g., {'Pb2+':1, 'I-':2}
                 atomic_masses,          # dict, e.g., {'Pb2+':207.2, 'I-':126.9}
                 solute_residues,        # dict, e.g., {'Pb2+': 'PBI', 'I-': 'PBI'}
                 solvent_name,           # 3-letter uppercase string, e.g., 'DMS'
                 total_mass=100.0        # grams, default to 100 g
                ):
        """
        Initializes the BulkVolume class with the given parameters.

        Parameters:
            mass_percent_solute (float): Mass percent of solute in the solution (%).
            density_solution (float): Density of the solution in g/cm³.
            density_neat_solvent (float): Density of the neat solvent in g/cm³.
            molar_mass_solvent (float): Molar mass of the solvent in g/mol.
            molar_mass_solute (float): Molar mass of the solute in g/mol.
            ionic_radii (dict): Ionic radii of solute atoms in angstroms (Å).
            stoichiometry (dict): Stoichiometric coefficients of solute atoms.
            atomic_masses (dict): Atomic masses of solute atoms in g/mol.
            solute_residues (dict): Mapping of solute atoms to their residue names, e.g., {'Pb2+': 'PBI', 'I-': 'PBI'}.
            solvent_name (str): 3-letter uppercase string representing the solvent, e.g., 'DMS'.
            total_mass (float): Total mass of the solution in grams (g). Default is 100 g.
        """
        self.mass_percent_solute = mass_percent_solute
        self.total_mass = total_mass
        self.density_solution = density_solution
        self.density_neat_solvent = density_neat_solvent
        self.molar_mass_solvent = molar_mass_solvent
        self.molar_mass_solute = molar_mass_solute
        self.ionic_radii = ionic_radii
        self.stoichiometry = stoichiometry
        self.atomic_masses = atomic_masses
        self.solute_residues = solute_residues
        self.solvent_name = solvent_name.upper()
        self.N_A = 6.022e23  # Avogadro's Number in molecules/mol

        # Initialize volume attributes
        self.V_solvent_A3 = 0.0
        self.V_atoms_A3 = {}

    def calculate_masses(self):
        """
        Calculates the masses of solute and solvent based on mass percent and total mass.

        Returns:
            tuple: (mass_solute, mass_solvent) in grams (g).
        """
        mass_solute = self.mass_percent_solute * self.total_mass / 100  # grams
        mass_solvent = self.total_mass - mass_solute  # grams
        return mass_solute, mass_solvent

    def calculate_total_volume(self, mass_solute, mass_solvent):
        """
        Calculates the total volume of the solution using its density.

        Parameters:
            mass_solute (float): Mass of solute in grams (g).
            mass_solvent (float): Mass of solvent in grams (g).

        Returns:
            float: Total volume of the solution in cubic centimeters (cm³).
        """
        V_solution_cm3 = self.total_mass / self.density_solution  # cm³
        return V_solution_cm3

    def calculate_moles(self, mass_solute, mass_solvent):
        """
        Calculates the number of moles of solute and solvent.

        Parameters:
            mass_solute (float): Mass of solute in grams (g).
            mass_solvent (float): Mass of solvent in grams (g).

        Returns:
            tuple: (n_solute, n_solvent) in moles (mol).
        """
        n_solute = mass_solute / self.molar_mass_solute  # mol
        n_solvent = mass_solvent / self.molar_mass_solvent  # mol
        return n_solute, n_solvent

    def calculate_number_of_atoms(self, n_solute):
        """
        Calculates the number of solute units and individual atoms.

        Parameters:
            n_solute (float): Number of moles of solute.

        Returns:
            dict: Number of atoms for each solute ion type.
        """
        num_solute_units = n_solute * self.N_A  # formula units
        num_atoms = {}
        for atom, stoich in self.stoichiometry.items():
            num_atoms[atom] = num_solute_units * stoich
        return num_atoms

    def calculate_number_of_solvent_molecules(self, n_solvent):
        """
        Calculates the number of solvent molecules.

        Parameters:
            n_solvent (float): Number of moles of solvent.

        Returns:
            float: Number of solvent molecules.
        """
        num_solvent_molecules = n_solvent * self.N_A  # molecules
        return num_solvent_molecules

    def calculate_volume_per_solvent_molecule(self):
        """
        Calculates the volume occupied by a single solvent molecule in the neat solvent.

        Returns:
            float: Volume per solvent molecule in cubic centimeters (cm³/molecule).
        """
        V_solvent_cm3_per_molecule = self.molar_mass_solvent / (self.density_neat_solvent * self.N_A)  # cm³/molecule
        return V_solvent_cm3_per_molecule

    def calculate_total_solvent_volume(self, num_solvent_molecules, V_solvent_cm3_per_molecule):
        """
        Calculates the total volume occupied by solvent molecules in the solution.

        Parameters:
            num_solvent_molecules (float): Number of solvent molecules.
            V_solvent_cm3_per_molecule (float): Volume per solvent molecule in cm³/molecule.

        Returns:
            float: Total solvent volume in cubic centimeters (cm³).
        """
        V_solvent_total_cm3 = num_solvent_molecules * V_solvent_cm3_per_molecule  # cm³
        return V_solvent_total_cm3

    def calculate_volume_fraction_solvent(self, V_solvent_total_cm3, V_solution_cm3):
        """
        Calculates the volume fraction occupied by solvent molecules.

        Parameters:
            V_solvent_total_cm3 (float): Total solvent volume in cm³.
            V_solution_cm3 (float): Total solution volume in cm³.

        Returns:
            float: Volume fraction of solvent (unitless).
        """
        phi_solvent = V_solvent_total_cm3 / V_solution_cm3  # unitless
        return phi_solvent

    def calculate_volume_fraction_solute(self, phi_solvent):
        """
        Calculates the volume fraction occupied by solute atoms.

        Parameters:
            phi_solvent (float): Volume fraction of solvent (unitless).

        Returns:
            float: Volume fraction of solute (unitless).
        """
        phi_solute = 1 - phi_solvent  # unitless
        return phi_solute

    def calculate_relative_volume_contributions(self, phi_solute):
        """
        Calculates the relative volume contributions of each solute atom type based on stoichiometry and ionic radii.

        Parameters:
            phi_solute (float): Volume fraction of solute (unitless).

        Returns:
            dict: Relative volume fractions for each solute atom type (unitless).
        """
        sum_radii_cubes = 0
        for atom, stoich in self.stoichiometry.items():
            sum_radii_cubes += stoich * (self.ionic_radii[atom] ** 3)
        
        phi_atoms = {}
        for atom, stoich in self.stoichiometry.items():
            phi_atoms[atom] = phi_solute * (stoich * (self.ionic_radii[atom] ** 3)) / sum_radii_cubes
        return phi_atoms

    def allocate_absolute_volumes(self, phi_atoms, V_solution_cm3):
        """
        Allocates absolute volumes to each solute atom type based on their volume fractions.

        Parameters:
            phi_atoms (dict): Relative volume fractions for each solute atom type (unitless).
            V_solution_cm3 (float): Total solution volume in cm³.

        Returns:
            dict: Absolute volumes for each solute atom type in cm³.
        """
        V_atoms_cm3 = {}
        for atom, phi in phi_atoms.items():
            V_atoms_cm3[atom] = phi * V_solution_cm3  # cm³
        return V_atoms_cm3

    def convert_volumes_to_A3(self, V_solvent_cm3, V_atoms_cm3):
        """
        Converts volumes from cubic centimeters (cm³) to cubic angstroms (Å³).

        Parameters:
            V_solvent_cm3 (float): Total solvent volume in cm³.
            V_atoms_cm3 (dict): Absolute volumes for each solute atom type in cm³.

        Returns:
            tuple: (V_solvent_A3, V_atoms_A3) where both are dictionaries with volumes in Å³.
        """
        cm3_to_A3 = 1e24  # 1 cm³ = 1e24 Å³
        V_solvent_A3 = V_solvent_cm3 * cm3_to_A3  # Å³
        V_atoms_A3 = {}
        for atom, V in V_atoms_cm3.items():
            V_atoms_A3[atom] = V * cm3_to_A3  # Å³
        return V_solvent_A3, V_atoms_A3

    def validate_volume_conservation(self, phi_solvent, phi_solute, phi_atoms):
        """
        Validates that the sum of volume fractions approximates 1 (or 100%).

        Parameters:
            phi_solvent (float): Volume fraction of solvent (unitless).
            phi_solute (float): Volume fraction of solute (unitless).
            phi_atoms (dict): Relative volume fractions for each solute atom type (unitless).

        Returns:
            bool: True if the sum is approximately 1, False otherwise.
        """
        sum_phi_atoms = sum(phi_atoms.values())
        total = phi_solvent + sum_phi_atoms
        return np.isclose(total, 1.0, atol=1e-4)

    def estimate_volumes(self):
        """
        Main method to estimate the volumes of solvent molecules and solute atoms.

        Returns:
            dict: A nested dictionary containing the volumes in cubic angstroms (Å³) for solvent and each solute atom type.
                  Example:
                  {
                      'DMS': {
                          'Solvent': {
                              'Total Volume': 8.18e25,            # Å³
                              'Volume per Molecule': 1.02e3       # Å³/molecule
                          }
                      },
                      'PBI': {
                          'Pb2+': {
                              'Total Volume': 1.34e24,            # Å³
                              'Volume per Atom': 1.34e24          # Å³/atom
                          },
                          'I-': {
                              'Total Volume': 1.70e25,            # Å³
                              'Volume per Atom': 2.34e25          # Å³/atom
                          }
                      }
                  }
        """
        # Step 1: Calculate Masses
        mass_solute, mass_solvent = self.calculate_masses()
        print(f"Mass of Solute: {mass_solute:.4f} g")
        print(f"Mass of Solvent ({self.solvent_name}): {mass_solvent:.4f} g")
        
        # Step 2: Calculate Total Volume of Solution
        V_solution_cm3 = self.calculate_total_volume(mass_solute, mass_solvent)
        print(f"Total Volume of Solution: {V_solution_cm3:.6f} cm³")
        
        # Step 3: Calculate Number of Moles
        n_solute, n_solvent = self.calculate_moles(mass_solute, mass_solvent)
        print(f"Moles of Solute: {n_solute:.6f} mol")
        print(f"Moles of Solvent ({self.solvent_name}): {n_solvent:.6f} mol")
        
        # Step 4: Calculate Number of Atoms
        num_atoms = self.calculate_number_of_atoms(n_solute)
        for atom, count in num_atoms.items():
            residue = self.solute_residues.get(atom, 'Unknown')
            print(f"Number of {atom} Atoms in Residue {residue}: {count:.4e} atoms")
        
        # Step 5: Calculate Number of Solvent Molecules
        num_solvent_molecules = self.calculate_number_of_solvent_molecules(n_solvent)
        print(f"Number of Solvent ({self.solvent_name}) Molecules: {num_solvent_molecules:.4e} molecules")
        
        # Step 6: Calculate Volume per Solvent Molecule
        V_solvent_cm3_per_molecule = self.calculate_volume_per_solvent_molecule()
        print(f"Volume per Solvent Molecule ({self.solvent_name}): {V_solvent_cm3_per_molecule:.6e} cm³/molecule")
        
        # Step 7: Calculate Total Solvent Volume
        V_solvent_total_cm3 = self.calculate_total_solvent_volume(num_solvent_molecules, V_solvent_cm3_per_molecule)
        print(f"Total Volume Occupied by Solvent ({self.solvent_name}): {V_solvent_total_cm3:.6f} cm³")
        
        # Step 8: Calculate Volume Fraction of Solvent
        phi_solvent = self.calculate_volume_fraction_solvent(V_solvent_total_cm3, V_solution_cm3)
        phi_solvent_percent = phi_solvent * 100
        print(f"Volume Fraction of Solvent ({self.solvent_name}): {phi_solvent_percent:.2f}%")
        
        # Step 9: Calculate Volume Fraction of Solute
        phi_solute = self.calculate_volume_fraction_solute(phi_solvent)
        phi_solute_percent = phi_solute * 100
        print(f"Volume Fraction of Solute: {phi_solute_percent:.2f}%")
        
        # Step 10: Calculate Relative Volume Contributions of Solute Atoms
        phi_atoms = self.calculate_relative_volume_contributions(phi_solute)
        for atom, phi in phi_atoms.items():
            residue = self.solute_residues.get(atom, 'Unknown')
            phi_percent = phi * 100
            print(f"Volume Fraction of {atom} in Residue {residue}: {phi_percent:.2f}%")
        
        # Step 11: Allocate Absolute Volumes to Solute Atoms
        V_atoms_cm3 = self.allocate_absolute_volumes(phi_atoms, V_solution_cm3)
        for atom, V in V_atoms_cm3.items():
            residue = self.solute_residues.get(atom, 'Unknown')
            print(f"Volume Occupied by {atom} in Residue {residue}: {V:.6f} cm³")
        
        # Step 12: Convert Volumes to Cubic Angstroms (Å³)
        V_solvent_A3, V_atoms_A3 = self.convert_volumes_to_A3(V_solvent_total_cm3, V_atoms_cm3)
        
        # Step 13: Validate Volume Conservation
        is_valid = self.validate_volume_conservation(phi_solvent, phi_solute, phi_atoms)
        if is_valid:
            print("Volume conservation validated: Sum of volume fractions is approximately 100%.")
        else:
            print("Volume conservation error: Sum of volume fractions does not approximate 100%.")
        
        # Step 14: Create Output Dictionary with Per Unit Volumes
        volumes = {
            self.solvent_name: {
                'Solvent': {
                    'Total Volume': V_solvent_A3,  # Å³
                    'Volume per Molecule': V_solvent_A3 / num_solvent_molecules  # Å³/molecule
                }
            }
        }
        for atom in self.ionic_radii.keys():
            residue = self.solute_residues.get(atom, 'Unknown')
            if residue not in volumes:
                volumes[residue] = {}
            volumes[residue][atom] = {
                'Total Volume': V_atoms_A3.get(atom, 0),  # Å³
                'Volume per Atom': V_atoms_A3.get(atom, 0) / num_atoms.get(atom, 1)  # Å³/atom
            }
        
        # Save volumes as class attributes
        self.V_solvent_A3 = V_solvent_A3
        self.V_atoms_A3 = V_atoms_A3
        
        # Print Estimated Volumes
        print("\nEstimated Volumes (in cubic angstroms, Å³):")
        for residue, atoms in volumes.items():
            for atom, vol_dict in atoms.items():
                print(f"Residue {residue} - {atom}:")
                for vol_type, volume in vol_dict.items():
                    print(f"  {vol_type}: {volume:.4e} Å³")
        
        self.volumes = volumes
        return volumes

    def estimate_atoms_in_box(self, box_side_A3):
        """
        Estimates the number of solute atoms of each type and solvent molecules
        that should reside within a cubic box of given side dimensions in angstroms (Å).
        Also calculates the estimated density in the box based on rounded counts.

        Parameters:
            box_side_A3 (float): Side length of the cubic box in angstroms (Å).

        Returns:
            tuple: (atoms_in_box, estimated_density)
                   atoms_in_box (dict): Number of atoms/molecules in the box.
                   estimated_density (float): Estimated density in g/cm³.
        """
        # Step 1: Estimate Volumes
        volumes = self.estimate_volumes()
        # Extract solvent volume and per molecule volume
        solvent_info = volumes[self.solvent_name]['Solvent']
        V_solvent_A3 = solvent_info['Total Volume']  # Å³
        V_solvent_per_molecule_A3 = solvent_info['Volume per Molecule']  # Å³/molecule
        
        # Extract solute volumes
        solute_volumes_A3 = {}
        for residue, atoms in volumes.items():
            if residue != self.solvent_name:
                for atom, vol_dict in atoms.items():
                    solute_volumes_A3[atom] = vol_dict['Total Volume']  # Å³
        
        # Step 2: Calculate Number Densities (atoms/molecules per cm³)
        # Convert volumes to cm³
        V_solvent_cm3 = V_solvent_A3 / 1e24  # cm³
        V_solute_cm3 = {atom: vol / 1e24 for atom, vol in solute_volumes_A3.items()}  # cm³
        V_solution_cm3 = self.total_mass / self.density_solution  # cm³

        # Number densities
        mass_solute, mass_solvent = self.calculate_masses()
        n_solute, n_solvent = self.calculate_moles(mass_solute, mass_solvent)
        num_atoms = self.calculate_number_of_atoms(n_solute)
        num_solvent_molecules = self.calculate_number_of_solvent_molecules(n_solvent)
        density_solvent = num_solvent_molecules / V_solution_cm3  # molecules/cm³
        density_solute = {}
        for atom in self.ionic_radii.keys():
            density_solute[atom] = num_atoms.get(atom, 0) / V_solution_cm3  # atoms/cm³

        # Step 3: Calculate Box Volume in cm³
        V_box_cm3 = (box_side_A3 ** 3) / 1e24  # cm³

        # Step 4: Calculate Number of Atoms/Molecules in Box
        num_solvent_in_box = density_solvent * V_box_cm3
        atoms_in_box = {
            'Solvent': int(round(num_solvent_in_box))
        }
        for atom, density in density_solute.items():
            atoms_in_box[atom] = int(round(density * V_box_cm3))

        print(f"\nBox Volume: {box_side_A3**3:.4e} Å³")
        print(f"Box Volume in cm³: {V_box_cm3:.6e} cm³")
        print(f"Number of Solvent ({self.solvent_name}) Molecules in Box: {atoms_in_box['Solvent']}")
        for atom in self.ionic_radii.keys():
            residue = self.solute_residues.get(atom, 'Unknown')
            print(f"Number of {atom} Atoms in Residue {residue}: {atoms_in_box[atom]} atoms")
        
        # Step 5: Calculate Mass in Box
        mass_solvent_in_box = atoms_in_box['Solvent'] * self.molar_mass_solvent / self.N_A  # g
        mass_solute_in_box = 0.0
        for atom, count in atoms_in_box.items():
            if atom != 'Solvent':
                mass_solute_in_box += count * self.atomic_masses[atom] / self.N_A  # g
        total_mass_in_box = mass_solvent_in_box + mass_solute_in_box  # g

        # Step 6: Calculate Estimated Density
        estimated_density = total_mass_in_box / V_box_cm3  # g/cm³

        print(f"Total Mass in Box: {total_mass_in_box:.6e} g")
        print(f"Estimated Density in Box: {estimated_density:.6f} g/cm³")

        return atoms_in_box, estimated_density
