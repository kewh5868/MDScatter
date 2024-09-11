import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SphereShapeFunction:
    def __init__(self, diameter, ro_value=0.05):
        """
        Initialize the SphereShapeFunction with a given diameter D and atomic density ro_value.
        
        Parameters:
        diameter (float): Diameter of the sphere in angstroms (Å).
        ro_value (float): Atomic density (ρ₀) in atoms per cubic angstrom (atoms/Å³). Default is 0.05.
        """
        self.size = diameter
        self.ro_value = ro_value
    
    def gamma(self, r):
        """
        Calculate the analytical shape function gamma(r, D) for the given range of r.
        
        Parameters:
        r (numpy.ndarray): Array of interatomic distances r in angstroms (Å).
        
        Returns:
        numpy.ndarray: Array of gamma(r, D) values corresponding to the input r.
        """
        gamma_values = np.where(
            r <= self.size,
            1 - (3/2) * (r/self.size) + (1/2) * (r/self.size)**3,
            0
        )
        return gamma_values
    
    def volume(self):
        """
        Calculate the volume of the sphere.
        
        Returns:
        float: Volume of the sphere in cubic angstroms (Å³).
        """
        return (np.pi / 6) * self.size ** 3

    def calculate_gamma_term(self, r_values):
        """
        Calculate the gamma term: -4 * π * r * ρ₀ * γ(r) for the given r values.
        
        Parameters:
        r_values (numpy.ndarray): Array of r values in angstroms (Å).
        
        Returns:
        numpy.ndarray: Array of gamma term values.
        """
        gamma_values = self.gamma(r_values)
        gamma_term = -4 * np.pi * r_values * self.ro_value * gamma_values
        return gamma_term
    
    def plot_shape_and_gamma_term(self, r_range):
        """
        Plot both the shape function γ(r) and the gamma term -4 * π * r * ρ₀ * γ(r).
        
        Parameters:
        r_range (tuple): Tuple specifying the (min, max) range of r values in angstroms (Å).
        """
        r_values = np.linspace(r_range[0], r_range[1], 1000)
        gamma_values = self.gamma(r_values)
        gamma_term = self.calculate_gamma_term(r_values)

        # Create two subplots: one for gamma(r), and one for the gamma term
        plt.figure(figsize=(12, 10))

        # First panel: Shape function γ(r)
        plt.subplot(2, 1, 1)
        plt.plot(r_values, gamma_values, label=f"Sphere D = {self.size} Å")
        plt.xlabel("Distance r (Å)")
        plt.ylabel(r"Shape Function $\gamma(r)$")
        plt.title("Shape Function for a Single Sphere")
        plt.grid(True)
        plt.legend()

        # Second panel: Gamma term -4πrρ₀γ(r)
        plt.subplot(2, 1, 2)
        plt.plot(r_values, gamma_term, label=r"$-4\pi r \rho_0 \gamma(r)$")
        plt.xlabel("Distance r (Å)")
        plt.ylabel(r"$-4\pi r \rho_0 \gamma(r)$")
        plt.title(r"Calculation of $-4\pi r \rho_0 \gamma(r)$ for a Single Sphere")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()
   
# # Example usage:
# diameter = 20  # Diameter of the sphere in angstroms
# r_range = (0.5, 30)  # Range of r values in angstroms
# ro_value = 0.05  # Atomic density in atoms/Å³

# sphere_shape_function = SphereShapeFunction(diameter, ro_value)
# sphere_shape_function.plot_gamma(r_range)

class CubeShapeFunction:
    def __init__(self, L_angstroms=53.635):
        """Initialize the CubeShapeFunction class with the edge length of the cube in angstroms."""
        self.L_angstroms = L_angstroms

    def cvf_cube(self, r):
        """Calculate the Common Volume Function (CVF) for a solid cube."""
        if r >= self.L_angstroms:
            return 0
        elif r <= 0:
            return self.L_angstroms**3
        else:
            return (self.L_angstroms - r)**3

    def calculate_shape_function(self, r_min=0, r_max=30, num_points=100):
        """Calculate and normalize the shape function for a solid cube."""
        r_values = np.linspace(r_min, r_max, num_points)  # User-defined r range
        shape_function = np.zeros_like(r_values)

        # Directions to sample (e.g., [100], [110], [111])
        directions = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]])
        directions = directions / np.linalg.norm(directions, axis=1)[:, None]  # Normalize directions

        # Weights for averaging (simplified equal weighting)
        weights = np.ones(len(directions)) / len(directions)

        for i, r in enumerate(r_values):
            # Sum the CVFs for each direction
            for w, d in zip(weights, directions):
                shape_function[i] += w * self.cvf_cube(r * np.dot(d, d))
        
        # Normalize the shape function so that it ranges from 0 to 1
        shape_function /= shape_function.max()

        return r_values, shape_function

    def plot_shape_function(self, r_values, shape_function):
        """Plot the normalized shape function."""
        plt.plot(r_values, shape_function, label="Normalized Shape Function (Cube in Angstroms)")
        plt.xlabel("Distance r (Å)")
        plt.ylabel(r"Normalized Shape Function $\gamma(r)$")
        plt.title("Normalized Shape Function for a Solid Cube")
        plt.grid(True)
        plt.legend()
        plt.show()

    def calculate_gamma_term(self, rho_naught=0.056, r_min=0, r_max=30, num_points=100):
        """Calculate and output -4*pi*r*rho_naught*gamma(r), where gamma(r) is the normalized shape function of a cube."""
        r_values, shape_function = self.calculate_shape_function(r_min, r_max, num_points)
        gamma_term = -4 * np.pi * r_values * rho_naught * shape_function

        # Plotting the gamma term
        plt.plot(r_values, gamma_term, label=r"$-4\pi r \rho_0 \gamma(r)$")
        plt.xlabel("Distance r (Å)")
        plt.ylabel(r"$-4\pi r \rho_0 \gamma(r)$")
        plt.title(r"Calculation of $-4\pi r \rho_0 \gamma(r)$ for a Solid Cube")
        plt.grid(True)
        plt.legend()
        plt.show()

        return r_values, gamma_term

# # Example usage
# cube = CubeShapeFunction(L_angstroms=53.635)
# r_values, shape_function = cube.calculate_shape_function(r_min=0, r_max=30, num_points=100)
# cube.plot_shape_function(r_values, shape_function)
# r_values, gamma_term = cube.calculate_gamma_term(rho_naught=0.056, r_min=0, r_max=30, num_points=100)

class SphereDistributionShapeFunction:
    def __init__(self, diameters, counts, ro_value=0.05):
        """
        Initialize the SphereDistributionShapeFunction with arrays of diameters and counts.
        
        Parameters:
        diameters (numpy.ndarray): Array of sphere diameters in angstroms (Å).
        counts (numpy.ndarray): Array of counts corresponding to each diameter.
        ro_value (float): Atomic density (ρ₀) in atoms per cubic angstrom (atoms/Å³). Default is 0.05.
        """
        self.diameters = np.array(diameters)
        self.counts = np.array(counts)
        self.ro_value = ro_value
        
        # Normalize the counts to get the number-weighted distribution
        self.normalized_counts = self.counts / np.sum(self.counts)
    
    def gamma(self, r, D):
        """
        Calculate the analytical shape function gamma(r, D) for a given diameter D.
        
        Parameters:
        r (numpy.ndarray): Array of interatomic distances r in angstroms (Å).
        D (float): Diameter of the sphere in angstroms (Å).
        
        Returns:
        numpy.ndarray: Array of gamma(r, D) values corresponding to the input r.
        """
        gamma_values = np.where(
            r <= D,
            1 - (3/2) * (r/D) + (1/2) * (r/D)**3,
            0
        )
        return gamma_values
    
    def weighted_gamma(self, r):
        """
        Calculate the number-weighted shape function for the distribution of spheres.
        
        Parameters:
        r (numpy.ndarray): Array of interatomic distances r in angstroms (Å).
        
        Returns:
        numpy.ndarray: Array of weighted gamma(r) values.
        """
        weighted_gamma_values = np.zeros_like(r)
        for D, count_weight in zip(self.diameters, self.normalized_counts):
            weighted_gamma_values += count_weight * self.gamma(r, D)
        return weighted_gamma_values
    
    def plot_weighted_gamma(self, r_range):
        """
        Plot the number-weighted shape function 4πrρ₀γ(r) for the distribution of spheres.
        
        Parameters:
        r_range (tuple): A tuple specifying the start and end of the range of r (in angstroms).
        """
        r_values = np.linspace(r_range[0], r_range[1], 1000)
        weighted_gamma_values = self.weighted_gamma(r_values)
        shape_function = 4 * np.pi * r_values * self.ro_value * weighted_gamma_values
        
        plt.figure(figsize=(10, 6))
        plt.plot(r_values, shape_function, label=f'Distribution of Spheres', color='blue')
        plt.xlabel('r (Å)')
        plt.ylabel('4πrρ₀γ(r) (Å⁻²)')
        plt.title('Number-Weighted Shape Function 4πrρ₀γ(r) for a Distribution of Spheres')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.show()
        
    # # Example usage:
    # diameters = [7, 9, 14]  # Diameters of the spheres in angstroms
    # counts = [7000, 1000, 200]  # Number of spheres with corresponding diameters
    # ro_value = 0.056  # Atomic density in atoms/Å³

    # r_range = (0.5, 30)  # Range of r values in angstroms

    # distribution_shape_function = SphereDistributionShapeFunction(diameters, counts, ro_value)
    # distribution_shape_function.plot_weighted_gamma(r_range)

class CompoundShapeFunction:
    def __init__(self):
        self.shapes = []

    def add_shape(self, shape_function_class, size_distribution, count_distribution):
        """
        Add a shape to the compound function.

        Parameters:
        - shape_function_class: A class instance representing the shape function (e.g., SphereShapeFunction).
        - size_distribution: A list or array of sizes for this shape.
        - count_distribution: A list or array of counts corresponding to each size.
        """
        self.shapes.append((shape_function_class, size_distribution, count_distribution))

    def calculate_compound_gamma(self, r_range):
        """
        Calculate the compound gamma function by summing contributions from all shapes, weighted by their volume and probability distribution.
        
        Parameters:
        - r_range: Tuple specifying the (min, max) range of r values.

        Returns:
        - r_values: Array of r values.
        - compound_gamma: Normalized compound gamma values for the r range.
        """
        r_values = np.linspace(r_range[0], r_range[1], 1000)
        compound_gamma = np.zeros_like(r_values)
        total_volume = 0

        for shape_class, size_dist, count_dist in self.shapes:
            for size, count in zip(size_dist, count_dist):
                shape_instance = shape_class(size)
                gamma_values = shape_instance.gamma(r_values)
                volume = shape_instance.volume()
                total_volume += count * volume
                compound_gamma += count * volume * gamma_values

        # Normalize the compound gamma to ensure the shape function is normalized
        compound_gamma /= total_volume

        return r_values, compound_gamma

    def calculate_gamma_term(self, r_range, rho_naught=0.056):
        """
        Calculate and output -4*pi*r*rho_naught*gamma(r) for the compound shape function.
        
        Parameters:
        - r_range: Tuple specifying the (min, max) range of r values.
        - rho_naught: Atomic density (ρ₀) in atoms per cubic angstrom (atoms/Å³).
        
        Returns:
        - r_values: Array of r values.
        - gamma_term: Array of calculated gamma term values.
        """
        r_values, compound_gamma = self.calculate_compound_gamma(r_range)
        gamma_term = -4 * np.pi * r_values * rho_naught * compound_gamma

        return r_values, gamma_term

    def plot_compound_gamma(self, r_range, rho_naught=0.056):
        """
        Plot the normalized compound gamma and the complete gamma term in two panels.
        
        Parameters:
        - r_range: Tuple specifying the (min, max) range of r values.
        - rho_naught: Atomic density (ρ₀) in atoms per cubic angstrom (atoms/Å³). Default is 0.056.
        """
        r_values, compound_gamma = self.calculate_compound_gamma(r_range)
        r_values, gamma_term = self.calculate_gamma_term(r_range, rho_naught)

        plt.figure(figsize=(12, 10))

        # First panel: Normalized compound shape function
        plt.subplot(2, 1, 1)
        plt.plot(r_values, compound_gamma, label="Normalized Compound Shape Function")
        plt.xlabel("Distance r (Å)")
        plt.ylabel(r"Normalized Shape Function $\gamma(r)$")
        plt.title("Normalized Compound Shape Function")
        plt.grid(True)
        plt.legend()

        # Second panel: Complete gamma term
        plt.subplot(2, 1, 2)
        plt.plot(r_values, gamma_term, label=r"$-4\pi r \rho_0 \gamma(r)$")
        plt.xlabel("Distance r (Å)")
        plt.ylabel(r"$-4\pi r \rho_0 \gamma(r)$")
        plt.title(r"Calculation of $-4\pi r \rho_0 \gamma(r)$ for the Compound Shape Function")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

class GofRCalculator:
    def __init__(self, rr_file, diameters, counts, ro_value=0.05):
        """
        Initialize the GofRCalculator with a file containing r vs. R(r) data and arrays of diameters and counts.
        
        Parameters:
        rr_file (str): Path to the file containing r (Å) vs. R(r) data.
        diameters (numpy.ndarray): Array of sphere diameters in angstroms (Å).
        counts (numpy.ndarray): Array of counts corresponding to each diameter.
        ro_value (float): Atomic density (ρ₀) in atoms per cubic angstrom (atoms/Å³). Default is 0.05.
        """
        self.ro_value = ro_value
        
        # Load r vs. R(r) data from the file (focus on r_A and sum columns)
        self.r_values, self.rr_values = self._load_rr_file(rr_file)
        
        # Initialize the sphere distribution shape function
        self.shape_function = SphereDistributionShapeFunction(diameters, counts, ro_value)
        
        # Calculate the total shape function
        self.total_shape_function = self.calculate_total_shape_function()
    
    def _load_rr_file(self, file_path):
        """
        Load the r (Å) vs. R(r) data from a file with specific columns.
        
        Parameters:
        file_path (str): Path to the file containing r (Å) vs. R(r) data.
        
        Returns:
        numpy.ndarray: r values (Å).
        numpy.ndarray: R(r) values (sum column).
        """
        # Read the data, specifying that the first row is the header
        data = pd.read_csv(file_path, delim_whitespace=True)
        
        # Extract the r_A and sum columns
        r_values = data['r_A'].values
        rr_values = data['sum'].values
        
        return r_values, rr_values
    
    def calculate_total_shape_function(self):
        """
        Calculate the total shape function (4πrρ₀γ(r)) for the given r values.
        
        Returns:
        numpy.ndarray: Total shape function values corresponding to the r values.
        """
        weighted_gamma_values = self.shape_function.weighted_gamma(self.r_values)
        total_shape_function = 4 * np.pi * self.r_values * self.ro_value * weighted_gamma_values
        return total_shape_function
    
    def calculate_gofr(self):
        """
        Calculate G(r) using the formula G(r) = R(r)/r - total shape function (4πρ₀γ(r)).
        
        Returns:
        numpy.ndarray: G(r) values.
        """
        gofr_values = self.rr_values / self.r_values - self.total_shape_function
        return gofr_values
    
    def plot_gofr(self):
        """
        Plot G(r) along with R(r)/r and the total shape function for comparison.
        """
        gofr_values = self.calculate_gofr()
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.r_values, gofr_values, label='G(r)', color='blue')
        plt.plot(self.r_values, self.rr_values / self.r_values, label='R(r)/r', color='red', linestyle='--')
        plt.plot(self.r_values, self.total_shape_function, label='Total Shape Function', color='green', linestyle='-.')
        plt.xlabel('r (Å)')
        plt.ylabel('G(r) (Å⁻²)')
        plt.title('G(r) Calculation from R(r) and Total Shape Function')
        plt.legend()
        plt.grid(True)
        plt.show()
