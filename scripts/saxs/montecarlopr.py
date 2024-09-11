import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

class MonteCarloPr:
    def __init__(self, num_points_per_set=100, num_sets=1000):
        """
        Initializes the Monte Carlo sampling class.

        Parameters:
        - num_points_per_set: Number of points to sample per set.
        - num_sets: Number of sets for Monte Carlo sampling.
        """
        self.num_points_per_set = num_points_per_set
        self.num_sets = num_sets

    def sample_random_points(self, points, num_samples):
        """Randomly sample points from the given set of points."""
        indices = np.random.choice(len(points), size=num_samples, replace=False)
        return points[indices]
    
    def calculate_pair_distances(self, points):
        """Calculate pairwise distances between all points."""
        num_points = len(points)
        distances = []
        for i in range(num_points):
            for j in range(i + 1, num_points):
                distance = np.linalg.norm(points[i] - points[j])
                distances.append(distance)
        return np.array(distances)
    
    def calculate_pr(self, points, num_pairs, bins=100):
        """
        Monte Carlo simulation to compute P(r) for the points in the visualized volume.

        Parameters:
        - points: Array of points from the electron density map.
        - num_pairs: Number of pairs to sample.
        - bins: Number of bins for the histogram of distances.
        
        Returns:
        - r_values: Array of r (distance) values.
        - hist: Array of P(r) values (pair distance distribution).
        """
        all_distances = []

        # Step 1: Estimate the number of pairs in the box
        total_num_points = len(points)
        max_possible_pairs = (total_num_points * (total_num_points - 1)) // 2

        # Ensure num_pairs is less than the maximum possible pairs
        num_pairs = min(num_pairs, max_possible_pairs)

        # Step 2: Sample points for multiple sets
        for _ in range(self.num_sets):
            # Sample random points from the grid
            sampled_points = self.sample_random_points(points, self.num_points_per_set)
            
            # Step 3: Randomly sample pairs and calculate pairwise distances
            distances = self.calculate_pair_distances(sampled_points)
            sampled_distances = np.random.choice(distances, size=num_pairs, replace=False)
            all_distances.extend(sampled_distances)
        
        all_distances = np.array(all_distances)

        # Step 4: Create a histogram of distances
        hist, bin_edges = np.histogram(all_distances, bins=bins, density=True)
        r_values = (bin_edges[:-1] + bin_edges[1:]) / 2  # Midpoint of bins
        
        return r_values, hist
    
    def plot_pr(self, points, num_pairs, bins=100):
        """Plot the P(r) function for the visualized volume."""
        r_values, pr_values = self.calculate_pr(points, num_pairs, bins)
        
        plt.plot(r_values, pr_values)
        plt.xlabel('r (Distance)')
        plt.ylabel('P(r)')
        plt.title('Pair Distance Distribution Function P(r) - Monte Carlo Sampling')
        plt.grid(True)
        plt.show()

    def plot_pr_smoothed(self, points, num_pairs, bins=100, smoothing_sigma=1.0):
        """Plot the smoothed P(r) function."""
        r_values, pr_values = self.calculate_pr(points, num_pairs, bins)
        
        # Apply Gaussian smoothing
        pr_values_smoothed = gaussian_filter1d(pr_values, sigma=smoothing_sigma)
        
        # Plot the smoothed P(r) function
        plt.plot(r_values, pr_values_smoothed, label='Smoothed P(r)')
        plt.xlabel('r (Distance)')
        plt.ylabel('P(r)')
        plt.title('Smoothed Pair Distance Distribution Function P(r)')
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_pr_smoothed_savgol(self, points, num_pairs, bins=100, window_length=11, polyorder=2):
        """Plot the smoothed P(r) function using Savitzky-Golay filter."""
        r_values, pr_values = self.calculate_pr(points, num_pairs, bins)
        
        # Apply Savitzky-Golay smoothing
        pr_values_smoothed = savgol_filter(pr_values, window_length=window_length, polyorder=polyorder)
        
        # Plot the smoothed P(r) function
        plt.plot(r_values, pr_values_smoothed, label='Smoothed P(r)')
        plt.xlabel('r (Distance)')
        plt.ylabel('P(r)')
        plt.title('Smoothed Pair Distance Distribution Function P(r)')
        plt.grid(True)
        plt.legend()
        plt.show()


    # # Example DMSO molecule structure (approximate positions):
    # coordinates = [
    #     (0.000, 0.000, 0.000),  # Sulfur (S)
    #     (1.530, 0.000, 0.000),  # Oxygen (O)
    #     (-1.090, -1.090, 0.000),  # Carbon (C1 - Methyl group)
    #     (-1.090, 1.090, 0.000),  # Carbon (C2 - Methyl group)
    #     (-2.140, -1.090, 0.000),  # Hydrogen (H1 - Methyl group)
    #     (-1.090, -2.140, 0.000),  # Hydrogen (H2 - Methyl group)
    #     (-2.140, 1.090, 0.000),   # Hydrogen (H3 - Methyl group)
    #     (-1.090, 2.140, 0.000)    # Hydrogen (H4 - Methyl group)
    # ]

    # elements = ['S', 'O', 'C', 'C', 'H', 'H', 'H', 'H']
    # charges = [0, 0, 0, 0, 0, 0, 0, 0]

    # # Create instance of ElectronDensityMapper
    # grid_size_input = 150  # Adjust grid density here
    # mapper = ElectronDensityMapper(coordinates, elements, charges, grid_size=grid_size_input)

    # # Set a cutoff value for the electron density and filter points
    # cutoff_value = 0.03
    # x, y, z = np.linspace(mapper.grid_limits[0], mapper.grid_limits[1], mapper.grid_size), \
    #           np.linspace(mapper.grid_limits[0], mapper.grid_limits[1], mapper.grid_size), \
    #           np.linspace(mapper.grid_limits[0], mapper.grid_limits[1], mapper.grid_size)
    # X, Y, Z = np.meshgrid(x, y, z)
    # density_flat = mapper.density_map.flatten()
    # valid_points = density_flat > cutoff_value
    # points = np.vstack((X.flatten()[valid_points], Y.flatten()[valid_points], Z.flatten()[valid_points])).T

    # monte_carlo_pr = MonteCarloPr(num_points_per_set=100, num_sets=1000)
    # num_pairs = 500  # Set the number of pairs to sample

    # # # Plot the smoothed P(r) function using Gaussian filter
    # # monte_carlo_pr.plot_pr_smoothed(points, num_pairs, bins=100, smoothing_sigma=1.0)

    # # OR plot the smoothed P(r) function using Savitzky-Golay filter
    # monte_carlo_pr.plot_pr_smoothed_savgol(points, num_pairs, bins=100, window_length=11, polyorder=2)
