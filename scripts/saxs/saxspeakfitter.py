import numpy as np
import scipy.optimize as optimize
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
from scipy.special import wofz

## - This is actually fitting pseudo-Voigt now, needs to be renamed.
class LorentzPeakFitter:
    def __init__(self, exp_file, qmin, qmax):
        self.exp_q, self.exp_iq = self._load_data(exp_file)
        
        # Define the fitting range in q-space
        self.qmin = qmin
        self.qmax = qmax

        self.peak_center = None  # To store the center of the peak
        self.lorentz_peak_params = None
        self.background = None  # To store the flat background

    def _load_data(self, filename):
        """
        Load q and I(q) from a .txt file with a header and two columns (q, I(q)).
        The header line is skipped.
        """
        data = np.loadtxt(filename, skiprows=1)  # Skip the header line
        q_values = data[:, 0]  # First column is q (Å⁻¹)
        iq_values = data[:, 1]  # Second column is I(q)
        return q_values, iq_values

    def _apply_q_range(self, qmin, qmax):
        """
        Apply the q-range filter to restrict data to the specified qmin and qmax.
        """
        mask = np.ones_like(self.exp_q, dtype=bool)
        if qmin is not None:
            mask &= self.exp_q >= qmin
        if qmax is not None:
            mask &= self.exp_q <= qmax
        
        exp_q = self.exp_q[mask]
        exp_iq = self.exp_iq[mask]

        return exp_q, exp_iq

    def pseudo_voigt(self, q, amplitude, center, width, eta, background):
        """
        Pseudo-Voigt function, a linear combination of a Lorentzian and a Gaussian.
        """
        lorentzian = amplitude * (width**2 / ((q - center)**2 + width**2))
        gaussian = amplitude * np.exp(-((q - center)**2) / (2 * width**2))
        return eta * lorentzian + (1 - eta) * gaussian + background

    def pseudo_voigt_residual(self, params, q_values, iq_values):
        """
        Residual function for the pseudo-Voigt peak fit, normalized by the intensity values.
        Includes the flat background as a parameter.
        """
        amplitude, center, width, eta, background = params
        voigt_fit = self.pseudo_voigt(q_values, amplitude, center, width, eta, background)
        residuals = (iq_values - voigt_fit) / iq_values
        return residuals

    def fit_lorentz_peak(self):
        """
        Fit a pseudo-Voigt peak to the experimental data over a fixed range.
        The fit is only performed on the left side of the peak center.
        """
        exp_q, exp_iq = self._apply_q_range(self.qmin, self.qmax)

        # Determine the peak center as the q-value corresponding to the maximum intensity
        max_index = np.argmax(exp_iq)
        self.peak_center = exp_q[max_index]

        # Use only the left side of the peak for fitting
        left_side_mask = exp_q <= self.peak_center
        exp_q_left = exp_q[left_side_mask]
        exp_iq_left = exp_iq[left_side_mask]

        # Initial guesses for amplitude, center, width, eta, and background
        initial_guess = [
            np.max(exp_iq_left),  # Amplitude
            self.peak_center,     # Center (determined from data)
            (self.qmax - self.qmin) / 4,  # Width (estimated as a quarter of the q-range)
            0.5,  # Eta (balance between Lorentzian and Gaussian)
            np.min(exp_iq_left)   # Flat background
        ]

        # Perform the fit using the weighted residuals
        result = optimize.least_squares(self.pseudo_voigt_residual, initial_guess, args=(exp_q_left, exp_iq_left))

        # Save the fitted pseudo-Voigt peak parameters and the background
        self.lorentz_peak_params = result.x[:-1]  # Exclude background from peak params
        self.background = result.x[-1]  # Flat background

        # Return the fitted pseudo-Voigt function
        def fitted_pseudo_voigt(q):
            return self.pseudo_voigt(q, *self.lorentz_peak_params, self.background)

        return fitted_pseudo_voigt

    def plot_fit(self):
        """
        Plot the experimental data and the fitted pseudo-Voigt peak over the fixed range.
        Also plot the fit for the left side of the peak only.
        """
        if self.lorentz_peak_params is None:
            raise ValueError("You must fit the Lorentz peak before plotting.")

        # Get the data for the fit range
        exp_q, exp_iq = self._apply_q_range(self.qmin, self.qmax)
        voigt_fit = self.pseudo_voigt(exp_q, *self.lorentz_peak_params, self.background)

        # Get the data for the left side of the peak
        left_side_mask = exp_q <= self.peak_center
        exp_q_left = exp_q[left_side_mask]
        exp_iq_left = exp_iq[left_side_mask]
        voigt_fit_left = self.pseudo_voigt(exp_q_left, *self.lorentz_peak_params, self.background)

        plt.figure(figsize=(10, 6))
        plt.loglog(exp_q, exp_iq, 'bo', label='Experimental Data (Full Range)')
        plt.loglog(exp_q_left, voigt_fit_left, 'r-', label='Pseudo-Voigt Fit (Left Side)')
        plt.xlabel('q (Å⁻¹)')
        plt.ylabel('I(q)')
        plt.title('Pseudo-Voigt Peak Fitting with Flat Background')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.show()