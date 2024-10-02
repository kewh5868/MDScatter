import numpy as np
from typing import TypedDict, Optional, Dict

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
    electrons_info: Optional[Dict[str, Dict[str, int]]]  # Optional electrons per unit

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
                 total_mass=100.0,        # grams, default to 100 g
                 electrons_info=None     # dict, optional electrons per unit
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

        # Initialize electron density attributes
        self.electron_density_cm3 = None
        self.electron_density_A3 = None
        self.solution_electron_density_cm3 = None
        self.solution_electron_density_A3 = None
        self.delta_rho_cm3 = None
        self.delta_rho_A3 = None
    
        # Initialize electrons_info attribute
        self.electrons_info = electrons_info

    ## Bulk Volume Estimation
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

    def estimate_volumes(self):
        """
        Main method to estimate the volumes of solvent molecules and solute atoms,
        including volume fractions and molecule/atom counts.

        Returns:
            dict: A nested dictionary containing the volumes in cubic angstroms (Å³),
                volume fractions, and counts for solvent and each solute atom type.
                Example:
                {
                    'DMS': {
                        'Solvent': {
                            'Count': 8.18e25,                   # molecules
                            'Total Volume': 1.02e3,             # Å³
                            'Volume Fraction': 50.00,           # %
                            'Volume per Molecule': 1.02e3       # Å³/molecule
                        }
                    },
                    'PBI': {
                        'Pb2+': {
                            'Count': 1.34e24,                    # atoms
                            'Total Volume': 1.34e24,             # Å³
                            'Volume Fraction': 30.00,            # %
                            'Volume per Atom': 1.34e24           # Å³/atom
                        },
                        'I-': {
                            'Count': 1.70e25,                    # atoms
                            'Total Volume': 2.34e25,             # Å³
                            'Volume Fraction': 20.00,            # %
                            'Volume per Atom': 2.34e25           # Å³/atom
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
        
        # Step 14: Create Output Dictionary with Counts and Volume Fractions
        volumes = {
            self.solvent_name: {
                'Solvent': {
                    'Count': num_solvent_molecules,                         # Number of solvent molecules
                    'Total Volume': V_solvent_A3,                           # Å³
                    'Volume Fraction': phi_solvent_percent,                 # %
                    'Volume per Molecule': V_solvent_A3 / num_solvent_molecules  # Å³/molecule
                }
            }
        }
        
        for atom in self.ionic_radii.keys():
            residue = self.solute_residues.get(atom, 'Unknown')
            if residue not in volumes:
                volumes[residue] = {}
            volumes[residue][atom] = {
                'Count': num_atoms.get(atom, 0),                          # Number of atoms
                'Total Volume': V_atoms_A3.get(atom, 0),                  # Å³
                'Volume Fraction': phi_atoms.get(atom, 0) * 100,          # %
                'Volume per Atom': V_atoms_A3.get(atom, 0) / max(num_atoms.get(atom, 1), 1)  # Å³/atom
            }
        
        # Save volumes as class attributes
        self.volumes = volumes
        self.V_solvent_A3 = V_solvent_A3
        self.V_atoms_A3 = V_atoms_A3
        
        # Print Estimated Volumes with Counts and Volume Fractions
        print("\nEstimated Volumes (in cubic angstroms, Å³), Counts, and Volume Fractions:")
        for residue, components in volumes.items():
            for component, details in components.items():
                print(f"\nResidue {residue} - {component}:")
                for key, value in details.items():
                    if 'Volume Fraction' in key:
                        print(f"  {key}: {value:.2f}%")
                    elif 'Count' in key:
                        unit = 'molecules' if component == 'Solvent' else 'atoms'
                        print(f"  {key}: {value:.4e} {unit}")
                    else:
                        print(f"  {key}: {value:.4e} Å³")
        
        return volumes

    ## Electron Density Estimation 
    def set_electrons_info(self, electrons_info):
        """
        Sets the electrons_info attribute after initialization.

        Parameters:
            electrons_info (dict): A nested dictionary containing the number of electrons per unit for each component.
        """
        self.electrons_info = electrons_info

    def add_electrons_per_unit(self):
        """
        Update the volumes dictionary by adding 'Electrons per Unit' and 'Total Electrons'
        for each unique Residue/Molecule and Residue/Atom.

        Parameters:
            electrons_info (dict): A nested dictionary containing the number of electrons per unit for each component.
                                The structure should mirror that of self.volumes.
                                Example:
                                {
                                    'DMSO': {
                                        'Solvent': 42  # electrons per molecule
                                    },
                                    'PBI': {
                                        'Pb2+': 80,     # electrons per atom
                                        'I-': 54        # electrons per atom
                                    }
                                }

        Returns:
            dict: The updated volumes dictionary with 'Electrons per Unit' and 'Total Electrons' added.
        """
        if self.electrons_info is None:
            raise ValueError("Electrons information has not been provided. Please set electrons_info first.")
    
        for residue, components in self.electrons_info.items():
            if residue in self.volumes:
                for component, electrons in components.items():
                    if component in self.volumes[residue]:
                        # Determine if the component is a solvent or an atom based on its naming
                        if component.lower() == 'solvent':
                            # Add 'Electrons per Molecule'
                            self.volumes[residue][component]['Electrons per Molecule'] = electrons
                            print(f"Added 'Electrons per Molecule' for {residue} - {component}: {electrons}")
                            
                            # Calculate and add 'Total Electrons'
                            count = self.volumes[residue][component].get('Count', None)
                            if count is not None:
                                total_electrons = electrons * count
                                self.volumes[residue][component]['Total Electrons'] = total_electrons
                                print(f"Calculated 'Total Electrons' for {residue} - {component}: {total_electrons:.4e}")
                            else:
                                print(f"Warning: 'Count' not found for {residue} - {component}. Cannot calculate 'Total Electrons'.")
                        
                        else:
                            # Add 'Electrons per Atom'
                            self.volumes[residue][component]['Electrons per Atom'] = electrons
                            print(f"Added 'Electrons per Atom' for {residue} - {component}: {electrons}")
                            
                            # Calculate and add 'Total Electrons'
                            count = self.volumes[residue][component].get('Count', None)
                            if count is not None:
                                total_electrons = electrons * count
                                self.volumes[residue][component]['Total Electrons'] = total_electrons
                                print(f"Calculated 'Total Electrons' for {residue} - {component}: {total_electrons:.4e}")
                            else:
                                print(f"Warning: 'Count' not found for {residue} - {component}. Cannot calculate 'Total Electrons'.")
                    else:
                        print(f"Warning: Component '{component}' not found under residue '{residue}' in volumes.")
            else:
                print(f"Warning: Residue '{residue}' not found in volumes.")
        
        return self.volumes
   
    def calculate_solution_electron_density(self):
        """
        Calculate the electron density of the solution (solute + solvent) in electrons/cm³ and electrons/Å³.
        Uses the specified parameters and the electrons_info dictionary.
        Stores the values as class attributes: self.solution_electron_density_cm3 and self.solution_electron_density_A3.
        Also outputs the total electrons per constituent used in the calculation.

        Returns:
            dict: A dictionary containing the solution's electron density in both units and total electrons per constituent.
                Example:
                {
                    'Solution Electron Density per cm³': 1.23e+23,  # electrons/cm³
                    'Solution Electron Density per Å³': 1.23e-1,    # electrons/Å³
                    'Total Electrons Solvent': X.XXe+YY,
                    'Total Electrons Solute': X.XXe+YY
                }
        """
        if self.electrons_info is None:
            raise ValueError("Electrons information has not been provided. Please set electrons_info first.")
        
        # Calculate masses of solute and solvent
        mass_solute, mass_solvent = self.calculate_masses()
        total_mass = mass_solute + mass_solvent  # Should be equal to self.total_mass

        # Calculate number of moles and molecules of solvent
        n_solvent = mass_solvent / self.molar_mass_solvent  # moles of solvent
        num_solvent_molecules = n_solvent * self.N_A  # number of solvent molecules

        # Get electrons per molecule for solvent
        electrons_per_molecule_solvent = self.electrons_info.get(self.solvent_name, {}).get('Solvent', None)
        if electrons_per_molecule_solvent is None:
            raise ValueError(f"Electrons per molecule for solvent '{self.solvent_name}' not found in electrons_info.")

        total_electrons_solvent = electrons_per_molecule_solvent * num_solvent_molecules

        # Calculate number of moles of solute
        n_solute = mass_solute / self.molar_mass_solute  # moles of solute

        # Calculate total electrons from solute components
        total_electrons_solute = 0.0
        total_electrons_solute_components = {}  # For storing electrons per solute component
        for component, stoich_coeff in self.stoichiometry.items():
            n_component = n_solute * stoich_coeff  # moles of component
            num_atoms_component = n_component * self.N_A  # number of atoms of component
            residue_name = self.solute_residues.get(component, None)
            if residue_name is None:
                raise ValueError(f"Residue name for solute component '{component}' not found in solute_residues.")
            electrons_per_atom = self.electrons_info.get(residue_name, {}).get(component, None)
            if electrons_per_atom is None:
                raise ValueError(f"Electrons per atom for solute component '{component}' not found in electrons_info.")
            total_electrons_component = electrons_per_atom * num_atoms_component
            total_electrons_solute += total_electrons_component
            total_electrons_solute_components[component] = total_electrons_component

        # Total electrons in the solution
        total_electrons_solution = total_electrons_solvent + total_electrons_solute

        # Total volume of the solution using solution density
        total_volume_solution_cm3 = total_mass / self.density_solution  # cm³
        total_volume_solution_A3 = total_volume_solution_cm3 * 1e24     # Å³

        # Calculate electron densities
        electron_density_cm3 = total_electrons_solution / total_volume_solution_cm3  # electrons/cm³
        electron_density_A3 = total_electrons_solution / total_volume_solution_A3    # electrons/Å³

        # Save as class attributes
        self.solution_electron_density_cm3 = electron_density_cm3
        self.solution_electron_density_A3 = electron_density_A3

        # Display the results
        print("\nElectron Density of the Solution (Solute + Solvent):")
        print(f"  Electron Density (cm³): {electron_density_cm3:.4e} electrons/cm³")
        print(f"  Electron Density (Å³): {electron_density_A3:.4e} electrons/Å³")
        print("\nTotal Electrons per Constituent:")
        print(f"  Total Electrons from Solvent: {total_electrons_solvent:.4e}")
        for component, electrons in total_electrons_solute_components.items():
            print(f"  Total Electrons from {component}: {electrons:.4e}")
        print(f"  Total Electrons from Solute: {total_electrons_solute:.4e}")
        print(f"  Total Electrons in Solution: {total_electrons_solution:.4e}")

        # Return the results
        return {
            'Solution Electron Density per cm³': electron_density_cm3,
            'Solution Electron Density per Å³': electron_density_A3,
            'Total Electrons Solvent': total_electrons_solvent,
            'Total Electrons Solute': total_electrons_solute,
            'Total Electrons per Solute Component': total_electrons_solute_components
        }
