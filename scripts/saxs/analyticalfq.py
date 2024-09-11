import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from scipy.linalg import eigh
from mendeleev import element  # To fetch ionic radii

class SphereScattering:
    def __init__(self, radius_of_gyration, electron_density_contrast=1.0):
        """
        Initialize the SphereScattering class.

        Parameters:
        - radius_of_gyration: The radius of gyration (Rg) in angstroms.
        - electron_density_contrast: The electron density contrast.
        """
        self.radius_of_gyration = radius_of_gyration  # Rg in angstroms
        self.electron_density_contrast = electron_density_contrast
        self.radius = self._calculate_radius()  # Radius in angstroms

    def _calculate_radius(self):
        """Calculate the sphere radius from the radius of gyration (Rg)."""
        return self.radius_of_gyration * np.sqrt(5)

    def calculate_iq(self, q_values):
        """
        Calculate the scattering intensity I(q) for a given range of q-values.
        
        Parameters:
        - q_values: A numpy array of q-values in inverse angstroms.

        Returns:
        - iq_values: A numpy array of I(q) values.
        """
        R = self.radius
        
        # Handle q=0 case separately to avoid division by zero
        q_values = np.where(q_values == 0, 1e-10, q_values)
        
        # Calculate the form factor
        form_factor = (3 * (np.sin(q_values * R) - q_values * R * np.cos(q_values * R)) / (q_values * R)**3) ** 2
        
        # Calculate I(q)
        iq_values = (self.electron_density_contrast ** 2) * (self.radius_of_gyration ** 6) * form_factor
        
        return iq_values

    def plot_iq(self, q_values):
        """
        Plot I(q) vs. q on a log-log scale.
        
        Parameters:
        - q_values: A numpy array of q-values in inverse angstroms.
        """
        iq_values = self.calculate_iq(q_values)
        
        # Create the plot
        plt.figure(figsize=(8, 6))
        plt.loglog(q_values, iq_values, marker='o', linestyle='-', color='b')
        plt.xlabel('q (Å⁻¹)')
        plt.ylabel('I(q)')
        plt.title('Scattering Intensity I(q) vs. Scattering Vector q')
        plt.grid(True, which="both", ls="--")
        plt.show()

class EllipsoidScattering:
    def __init__(self, a, b, c, electron_density_contrast=1.0):
        """
        Initialize the EllipsoidScattering class.
        
        Parameters:
        - a: Semi-axis length along the x-axis (angstroms)
        - b: Semi-axis length along the y-axis (angstroms)
        - c: Semi-axis length along the z-axis (angstroms)
        - electron_density_contrast: Electron density contrast (default is 1.0)
        """
        self.a = a
        self.b = b
        self.c = c
        self.electron_density_contrast = electron_density_contrast
        self.volume = self._calculate_volume()  # Volume of the ellipsoid in cubic angstroms

    def _calculate_volume(self):
        """Calculate the volume of the ellipsoid."""
        return (4/3) * np.pi * self.a * self.b * self.c

    def _calculate_R_alpha(self, q, theta, phi):
        """Calculate the effective radius R(α) of the ellipsoid."""
        numerator = self.a * self.b * self.c
        denominator = np.sqrt(
            (self.b**2 * self.c**2 * np.sin(theta)**2 * np.sin(phi)**2) +
            (self.a**2 * self.c**2 * np.sin(theta)**2 * np.cos(phi)**2) +
            (self.a**2 * self.b**2 * np.cos(theta)**2)
        )
        return numerator / denominator

    def _integrate_phi(self, q, theta):
        """Perform the phi integration for a given q and theta."""
        phi_integral = 0.0
        for phi in np.linspace(0, 2 * np.pi, 100):
            R_alpha = self._calculate_R_alpha(q, theta, phi)
            form_factor = (
                3 * (np.sin(q * R_alpha) - q * R_alpha * np.cos(q * R_alpha))
                / (q * R_alpha) ** 3
            )
            phi_integral += form_factor ** 2
        return phi_integral

    def _integrate_theta(self, q):
        """Perform the theta integration for a given q."""
        theta_integral = 0.0
        with ThreadPoolExecutor() as executor:
            phi_integrals = list(executor.map(lambda theta: self._integrate_phi(q, theta), np.linspace(0, np.pi, 100)))
        for theta, phi_integral in zip(np.linspace(0, np.pi, 100), phi_integrals):
            theta_integral += phi_integral * np.sin(theta)
        return theta_integral

    def calculate_iq(self, q_values):
        """
        Calculate the scattering intensity I(q) for a given range of q-values.
        
        Parameters:
        - q_values: A numpy array of q-values in inverse angstroms.

        Returns:
        - iq_values: A numpy array of I(q) values.
        """
        # Initialize intensity array
        iq_values = np.zeros_like(q_values)

        # Integration over all orientations of the ellipsoid
        with ThreadPoolExecutor() as executor:
            theta_integrals = list(executor.map(self._integrate_theta, q_values))
        
        iq_values = np.array(theta_integrals)

        # Normalize by the total volume and electron density contrast
        iq_values *= (self.electron_density_contrast ** 2) * (self.volume ** 2)
        return iq_values

    def plot_iq(self, q_values):
        """
        Plot I(q) vs. q on a log-log scale.
        
        Parameters:
        - q_values: A numpy array of q-values in inverse angstroms.
        """
        iq_values = self.calculate_iq(q_values)
        
        # Create the plot
        plt.figure(figsize=(8, 6))
        plt.loglog(q_values, iq_values, marker='o', linestyle='-', color='b')
        plt.xlabel('q (Å⁻¹)')
        plt.ylabel('I(q)')
        plt.title('Scattering Intensity I(q) vs. Scattering Vector q for an Ellipsoid')
        plt.grid(True, which="both", ls="--")
        plt.show()


'''import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

class SphereScattering:
    def __init__(self, radius):
        self.R = radius

    def I_q(self, Q):
        QR = Q * self.R
        return (3 * (np.sin(QR) - QR * np.cos(QR)) / QR**3)**2

    def analytical_P_r(self, r):
        D = self.R * 2  # Adjusting D = 2 * R
        if r <= D:
            return r**2 * (1 - (3/2)*(r/D) + (1/2)*(r/D)**3)
        else:
            return 0

    def P_r(self, r, q_min=0.01, q_max=1, points=500):
        # Discretize q in the finite interval
        Q = np.linspace(q_min, q_max, points)
        I_Q = self.I_q(Q)
        
        # Numerically compute P(r) using the provided integral
        def integrand(q, r):
            return q**2 * self.I_q(q) * (np.sin(q * r) / (q * r))
        
        P_r_val = (r**2 / (2 * np.pi**2)) * np.trapz([integrand(q, r) for q in Q], Q)
        return P_r_val

    def compute_P_r_range(self, r_min=0.001, r_max=None, points=500, q_min=0.01, q_max=1):
        if r_max is None:
            r_max = 2 * self.R
        r_vals = np.linspace(r_min, r_max, points)
        P_r_vals = np.array([self.P_r(r, q_min, q_max) for r in r_vals])
        P_r_analytical = np.array([self.analytical_P_r(r) for r in r_vals])
        
        # Normalize both P(r) values to 1 at their maxima
        P_r_vals /= np.max(P_r_vals)
        P_r_analytical /= np.max(P_r_analytical)
        
        return r_vals, P_r_vals, P_r_analytical

    def analytical_I_q_from_P_r(self, q, r_min=0.001, r_max=None, points=500):
        if r_max is None:
            r_max = 2 * self.R
        r_vals = np.linspace(r_min, r_max, points)
        P_r_analytical = np.array([self.analytical_P_r(r) for r in r_vals])

        # Compute I(Q) from the analytical P(r) using the integral provided
        def integrand(r, q):
            return P_r_analytical[r_vals == r][0] * (np.sin(q * r) / (q * r))

        I_q_val = 4 * np.pi * np.trapz([integrand(r, q) for r in r_vals], r_vals)
        return I_q_val

    def compute_I_q_range(self, q_min=0.01, q_max=1, points=500):
        Q_vals = np.linspace(q_min, q_max, points)
        I_q_vals = np.array([self.I_q(Q) for Q in Q_vals])
        I_q_analytical = np.array([self.analytical_I_q_from_P_r(Q) for Q in Q_vals])
        
        # Normalize both I(Q) values to 1 at their maxima
        I_q_vals /= np.max(I_q_vals)
        I_q_analytical /= np.max(I_q_analytical)
        
        return Q_vals, I_q_vals, I_q_analytical

    def compare_P_r_and_I_q(self, r_min=0.001, r_max=None, points=500, q_min=0.01, q_max=10):
        # Compare P(r)
        r_vals, P_r_vals, P_r_analytical = self.compute_P_r_range(r_min, r_max, points, q_min, q_max)
        Q_vals, I_q_vals, I_q_analytical = self.compute_I_q_range(q_min, q_max, points)

        # Plot P(r) comparison
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(r_vals, P_r_vals, label='Numerical P(r) from I(Q)', linestyle='--')
        plt.plot(r_vals, P_r_analytical, label='Analytical P(r)', linestyle='-')
        plt.xlabel('r')
        plt.ylabel('P(r)')
        plt.title('Comparison of Normalized P(r)')
        plt.legend()
        plt.grid(True)

        # Plot I(Q) comparison
        plt.subplot(1, 2, 2)
        plt.loglog(Q_vals, I_q_vals, label='Numerical I(Q)', linestyle='--')
        plt.loglog(Q_vals, I_q_analytical, label='Analytical I(Q) from P(r)', linestyle='-')
        plt.xlabel('Q')
        plt.ylabel('I(Q)')
        plt.title('Comparison of Normalized I(Q)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
        '''

'''class EllipsoidScattering:
    def __init__(self, a, b, c, electron_density_contrast=1.0):
        """
        Initialize the EllipsoidScattering class.
        
        Parameters:
        - a: Semi-axis length along the x-axis (angstroms)
        - b: Semi-axis length along the y-axis (angstroms)
        - c: Semi-axis length along the z-axis (angstroms)
        - electron_density_contrast: Electron density contrast (default is 1.0)
        """
        self.a = a
        self.b = b
        self.c = c
        self.electron_density_contrast = electron_density_contrast
        self.volume = self._calculate_volume()  # Volume of the ellipsoid in cubic angstroms

    def _calculate_volume(self):
        """Calculate the volume of the ellipsoid."""
        return (4/3) * np.pi * self.a * self.b * self.c

    def _calculate_R_alpha(self, q, theta, phi):
        """Calculate the effective radius R(α) of the ellipsoid."""
        numerator = self.a * self.b * self.c
        denominator = np.sqrt(
            (self.b**2 * self.c**2 * np.sin(theta)**2 * np.sin(phi)**2) +
            (self.a**2 * self.c**2 * np.sin(theta)**2 * np.cos(phi)**2) +
            (self.a**2 * self.b**2 * np.cos(theta)**2)
        )
        return numerator / denominator

    def calculate_iq(self, q_values):
        """
        Calculate the scattering intensity I(q) for a given range of q-values.
        
        Parameters:
        - q_values: A numpy array of q-values in inverse angstroms.

        Returns:
        - iq_values: A numpy array of I(q) values.
        """
        # Initialize intensity array
        iq_values = np.zeros_like(q_values)

        # Integration over all orientations of the ellipsoid
        for i, q in enumerate(q_values):
            theta_integral = 0.0
            for theta in np.linspace(0, np.pi, 100):
                phi_integral = 0.0
                for phi in np.linspace(0, 2 * np.pi, 100):
                    R_alpha = self._calculate_R_alpha(q, theta, phi)
                    form_factor = (
                        3 * (np.sin(q * R_alpha) - q * R_alpha * np.cos(q * R_alpha))
                        / (q * R_alpha) ** 3
                    )
                    phi_integral += form_factor ** 2
                theta_integral += phi_integral * np.sin(theta)
            iq_values[i] = theta_integral

        # Normalize by the total volume and electron density contrast
        iq_values *= (self.electron_density_contrast ** 2) * (self.volume ** 2)
        return iq_values

    def plot_iq(self, q_values):
        """
        Plot I(q) vs. q on a log-log scale.
        
        Parameters:
        - q_values: A numpy array of q-values in inverse angstroms.
        """
        iq_values = self.calculate_iq(q_values)
        
        # Create the plot
        plt.figure(figsize=(8, 6))
        plt.loglog(q_values, iq_values, marker='o', linestyle='-', color='b')
        plt.xlabel('q (Å⁻¹)')
        plt.ylabel('I(q)')
        plt.title('Scattering Intensity I(q) vs. Scattering Vector q for an Ellipsoid')
        plt.grid(True, which="both", ls="--")
        plt.show()
'''