import numpy as np
import plotly.graph_objects as go
import mendeleev
from mendeleev import element

class ElectronDensityMapper:
    def __init__(self, coordinates, elements, charges, grid_size=100, grid_limits=(-10, 10)):
        """
        Initializes the class with atom coordinates, element symbols, and ion charges.
        
        Parameters:
        - coordinates: list of tuples representing the atomic coordinates [(x1, y1, z1), (x2, y2, z2), ...]
        - elements: list of element symbols ['H', 'O', 'Na', ...]
        - charges: list of corresponding ion charges for each element [1, -1, 1, ...]
        - grid_size: resolution of the grid for the 3D space
        - grid_limits: bounds of the 3D grid (min, max) for each axis
        """
        self.coordinates = np.array(coordinates)
        self.elements = elements
        self.charges = charges
        self.grid_size = grid_size
        self.grid_limits = grid_limits
        
        # Find the unique element types and their charges
        self.unique_atoms = self._find_unique_atoms()
        
        # Build a dictionary mapping each unique element and charge to its properties
        self.atom_properties = self._build_atom_library()
        
        # Generate a 3D electron density map
        self.density_map = self._compute_density_map()
        
        print("Electron Density Min:", np.min(self.density_map))
        print("Electron Density Max:", np.max(self.density_map))

        # Extract individual properties for use in getter methods
        self.electron_count_map, self.ionic_radius_map, self.gaussian_width_map = self._map_properties_to_atoms()
    
    def _find_unique_atoms(self):
        """Find unique combinations of elements and charges."""
        return set(zip(self.elements, self.charges))
    
    def _build_atom_library(self):
        """Builds a library with electron count, atomic radius (in Å), and Gaussian width."""
        atom_lib = {}
        for element_symbol, charge in self.unique_atoms:
            try:
                elem = element(element_symbol)  # Get element from mendeleev
                
                # Get electron count (atomic number)
                electron_count = elem.electrons  # No need to subtract charge for neutral atoms
                
                # Get the atomic radius (in pm, convert to Å)
                atomic_radius_pm = elem.atomic_radius
                if atomic_radius_pm is not None:
                    atomic_radius = atomic_radius_pm / 100  # Convert pm to Å
                else:
                    atomic_radius = np.nan
                
                # Gaussian width (σ) derived from atomic radius
                gaussian_width = atomic_radius / 2.355 if atomic_radius is not np.nan else np.nan
                
                # Print debugging information
                print(f"Element: {element_symbol}, Electron Count: {electron_count}, "
                    f"Atomic Radius (Å): {atomic_radius}, Gaussian Width: {gaussian_width}")
                
                atom_lib[(element_symbol, charge)] = (electron_count, atomic_radius, gaussian_width)
            except Exception as e:
                print(f"Error retrieving data for element {element_symbol}: {e}")
                atom_lib[(element_symbol, charge)] = (np.nan, np.nan, np.nan)  # Handle missing data safely
        
        return atom_lib

    def _map_properties_to_atoms(self):
        """Maps the electron count, ionic radii, and Gaussian widths from the atom library to each coordinate."""
        electron_count_map = []
        ionic_radius_map = []
        gaussian_width_map = []
        
        for elem, charge in zip(self.elements, self.charges):
            electron_count, ionic_radius, gaussian_width = self.atom_properties[(elem, charge)]
            electron_count_map.append(electron_count)
            ionic_radius_map.append(ionic_radius)
            gaussian_width_map.append(gaussian_width)
        
        return np.array(electron_count_map), np.array(ionic_radius_map), np.array(gaussian_width_map)
    
    def _compute_density_map(self):
        """Computes the electron density map by summing Gaussian contributions from each atom."""
        # Create a 3D grid
        x, y, z = np.linspace(self.grid_limits[0], self.grid_limits[1], self.grid_size), \
                  np.linspace(self.grid_limits[0], self.grid_limits[1], self.grid_size), \
                  np.linspace(self.grid_limits[0], self.grid_limits[1], self.grid_size)
        X, Y, Z = np.meshgrid(x, y, z)
        density = np.zeros_like(X)
        
        # Sum Gaussian contributions from each atom
        for (x0, y0, z0), elem, charge in zip(self.coordinates, self.elements, self.charges):
            electron_count, _, gaussian_width = self.atom_properties[(elem, charge)]
            
            # Calculate the squared distance from the center of the atom
            r_squared = (X - x0)**2 + (Y - y0)**2 + (Z - z0)**2
            
            # Apply a 3σ cutoff: If r > 3σ, set density to zero
            mask = r_squared <= (3 * gaussian_width)**2
            gaussian = np.zeros_like(X)
            gaussian[mask] = (electron_count / (gaussian_width * np.sqrt(2 * np.pi))**3) * \
                             np.exp(-r_squared[mask] / (2 * gaussian_width**2))
            density += gaussian
        
        return density

    def visualize_smooth_surface(self, threshold=0.01):
        """
        Visualize a smooth isosurface that encloses the electron density.
        
        Parameters:
        - threshold: The electron density value for which the isosurface is drawn.
        """
        x, y, z = np.linspace(self.grid_limits[0], self.grid_limits[1], self.grid_size), \
                np.linspace(self.grid_limits[0], self.grid_limits[1], self.grid_size), \
                np.linspace(self.grid_limits[0], self.grid_limits[1], self.grid_size)

        X, Y, Z = np.meshgrid(x, y, z)
        
        # Flatten the density map
        x_flat = X.flatten()
        y_flat = Y.flatten()
        z_flat = Z.flatten()
        density_flat = self.density_map.flatten()
        
        # Plot the isosurface using Plotly
        fig = go.Figure(data=go.Isosurface(
            x=x_flat,
            y=y_flat,
            z=z_flat,
            value=density_flat,
            isomin=threshold,
            isomax=np.max(density_flat),
            surface_count=10,  # Single isosurface
            opacity=0.6,
            colorscale="Viridis",
            caps=dict(x_show=False, y_show=False, z_show=False)
        ))
        
        fig.update_layout(
            title=f"Isosurface at Electron Density = {threshold}",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        
        fig.show()

    def visualize_gaussian_spheres(self, cutoff=0.01):
        """Visualize an array of Gaussian spheres above a given density cutoff."""
        x, y, z = np.linspace(self.grid_limits[0], self.grid_limits[1], self.grid_size), \
                  np.linspace(self.grid_limits[0], self.grid_limits[1], self.grid_size), \
                  np.linspace(self.grid_limits[0], self.grid_limits[1], self.grid_size)

        # Apply the cutoff to the density map, set near-zero values to NaN
        density_cutoff = np.where(self.density_map > cutoff, self.density_map, np.nan)
        
        # Flatten the arrays for Plotly
        X, Y, Z = np.meshgrid(x, y, z)
        x_flat = X.flatten()
        y_flat = Y.flatten()
        z_flat = Z.flatten()
        density_flat = density_cutoff.flatten()

        # Plot the isosurface using Plotly
        fig = go.Figure(data=go.Isosurface(
            x=x_flat,
            y=y_flat,
            z=z_flat,
            value=density_flat,
            isomin=cutoff,
            isomax=np.nanmax(density_flat),
            surface_count=10,
            opacity=0.6,
            colorscale="Viridis",
            caps=dict(x_show=False, y_show=False, z_show=False)
        ))

        fig.update_layout(
            title="3D Gaussian Spheres with 3σ Cutoff and Empty Space for Zero Density",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )

        fig.show()

    def calculate_enclosed_volume(self, threshold):
        """
        Calculate the volume enclosed by the electron density isocontour.
        
        Parameters:
        - threshold: The electron density threshold for defining the isocontour.
        
        Returns:
        - enclosed_volume: The volume enclosed by the isocontour surface.
        """
        # Create the 3D grid spacing (assume uniform grid)
        delta_x = (self.grid_limits[1] - self.grid_limits[0]) / self.grid_size
        delta_y = delta_x  # Assume cubic grid cells
        delta_z = delta_x
        
        # Compute the volume of a single grid cell
        cell_volume = delta_x * delta_y * delta_z
        
        # Find all points where electron density >= threshold
        enclosed_points = np.where(self.density_map >= threshold)
        
        # Total enclosed volume is the number of such points multiplied by the cell volume
        enclosed_volume = len(enclosed_points[0]) * cell_volume
        
        return enclosed_volume

    def plot_3d_density_grid(self, cutoff):
        """
        Plots the 3D grid of points with electron density values above a given cutoff.
        
        Parameters:
        - cutoff: Electron density threshold for including points in the plot.
        """
        x, y, z = np.linspace(self.grid_limits[0], self.grid_limits[1], self.grid_size), \
                  np.linspace(self.grid_limits[0], self.grid_limits[1], self.grid_size), \
                  np.linspace(self.grid_limits[0], self.grid_limits[1], self.grid_size)
        X, Y, Z = np.meshgrid(x, y, z)
        
        # Flatten the arrays for Plotly
        x_flat = X.flatten()
        y_flat = Y.flatten()
        z_flat = Z.flatten()
        density_flat = self.density_map.flatten()

        # Apply the cutoff and keep only the points above the threshold
        valid_points = density_flat > cutoff
        x_filtered = x_flat[valid_points]
        y_filtered = y_flat[valid_points]
        z_filtered = z_flat[valid_points]
        density_filtered = density_flat[valid_points]

        # Plot the 3D grid of points
        fig = go.Figure(data=[go.Scatter3d(
            x=x_filtered, y=y_filtered, z=z_filtered,
            mode='markers',
            marker=dict(
                size=3,
                color=density_filtered,
                colorscale='Viridis',
                opacity=0.8
            )
        )])

        fig.update_layout(
            title=f'3D Electron Density Grid (Cutoff: {cutoff})',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )

        fig.show()

    # # Example usage with the DMSO molecule:
    # threshold_value = 0.0001  # Adjust based on the desired contour level
    # enclosed_volume = mapper.calculate_enclosed_volume(threshold_value)
    # print(f"Enclosed Volume at threshold {threshold_value}: {enclosed_volume} Å^3")

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

    # elements = ['S', 'O', 'C', 'C', 'H', 'H', 'H', 'H']  # Element types
    # charges = [0, 0, 0, 0, 0, 0, 0, 0]  # Charges for neutral atoms

    # # Create instance of the ElectronDensityMapper
    # mapper = ElectronDensityMapper(coordinates, elements, charges)

    # # Set the cutoff value to remove low-density regions and visualize Gaussian spheres
    # cutoff_value = 0.00015  # Adjust this value as needed
    # mapper.visualize_gaussian_spheres(cutoff=cutoff_value)
    # # mapper.visualize_smooth_surface(threshold=0.5)

    # Example DMSO molecule structure (approximate positions):
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

    # elements = ['S', 'O', 'C', 'C', 'H', 'H', 'H', 'H']  # Element types
    # charges = [0, 0, 0, 0, 0, 0, 0, 0]  # Charges for neutral atoms

    # # User input for changing grid density (resolution)
    # grid_size_input = 150  # Adjust grid density here (higher value = finer resolution)

    # # Create instance of ElectronDensityMapper with the new grid size
    # mapper = ElectronDensityMapper(coordinates, elements, charges, grid_size=grid_size_input)

    # # Set a cutoff value for the electron density and plot the grid of points
    # cutoff_value = 0.0001  # Adjust the cutoff value to visualize different density levels
    # mapper.plot_3d_density_grid(cutoff=cutoff_value)
