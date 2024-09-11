import numpy as np
import scipy.optimize as optimize
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
from scipy.special import wofz

class SAXSDataFitter:
    def __init__(self, exp_file, model_file, qmin_peak, qmax, qmin):
        self.exp_file = exp_file
        self.exp_q, self.exp_iq = self._load_data(exp_file)
        self.model_q, self.model_iq = self._load_data(model_file)
        
        # Interpolate model I(q) to match experimental q-values
        self.model_iq_interp = self._interpolate_model()
        
        # Define the fitting range in q-space
        self.qmin_peak = qmin_peak  # Minimum bound for pseudo-Voigt peak fitting
        self.qmax = qmax  # Maximum bound for both peak and model fitting
        self.qmin = qmin  # Minimum bound for complete model fitting

        self.scaling_factor = 1.0  # Initial scaling factor
        self.pseudo_voigt_peak = None  # This will hold the fitted pseudo-Voigt peak function

    def _load_data(self, filename):
        data = np.loadtxt(filename, skiprows=1)
        q_values = data[:, 0]
        iq_values = data[:, 1]
        return q_values, iq_values
    
    def _interpolate_model(self):
        interpolation_function = interpolate.interp1d(self.model_q, self.model_iq, kind='linear', fill_value="extrapolate")
        model_iq_interp = interpolation_function(self.exp_q)
        return model_iq_interp

    def _apply_q_range(self, qmin, qmax):
        mask = np.ones_like(self.exp_q, dtype=bool)
        if qmin is not None:
            mask &= self.exp_q >= qmin
        if qmax is not None:
            mask &= self.exp_q <= qmax
        
        exp_q = self.exp_q[mask]
        exp_iq = self.exp_iq[mask]
        model_iq_interp = self.model_iq_interp[mask]

        return exp_q, exp_iq, model_iq_interp

    def fit_lorentz_peak(self):
        try:
            # Initialize the LorentzPeakFitter with the provided q-range
            lorentz_fitter = LorentzPeakFitter(
                exp_file=self.exp_file, 
                qmin=self.qmin_peak, 
                qmax=self.qmax
            )
            
            # Perform the Lorentz peak fitting
            self.pseudo_voigt_peak = lorentz_fitter.fit_lorentz_peak()

            # Debug: Check if the peak was fitted
            if self.pseudo_voigt_peak is not None:
                print("Pseudo-Voigt peak fitting was successful.")
            else:
                print("Pseudo-Voigt peak fitting returned None.")
            
            # # Plot the peak fit to visualize it
            # lorentz_fitter.plot_fit()

        except ValueError as e:
            print(f"Failed to fit Lorentz peak: {e}")
            self.pseudo_voigt_peak = None

    def initial_model_scaling(self):
        exp_q, exp_iq, model_iq_interp = self._apply_q_range(self.qmin, self.qmax)
        
        # Get the model and experimental values at qmin
        model_at_qmin = model_iq_interp[0]  # Assuming the first value corresponds to qmin
        exp_at_qmin = exp_iq[0]  # Same assumption

        # Compute the initial scaling factor to match model to experimental data at qmin
        initial_scaling_factor = exp_at_qmin / model_at_qmin
        return initial_scaling_factor

    def residual(self, scaling_factor, exp_q, exp_iq, model_iq_interp):
        model_iq_scaled = scaling_factor * model_iq_interp
        
        if self.pseudo_voigt_peak is not None:
            pseudo_voigt_contribution = self.pseudo_voigt_peak(exp_q)
            model_iq_scaled += pseudo_voigt_contribution
        
        residuals = (exp_iq - model_iq_scaled) / exp_iq
        return residuals
    
    def fit_model(self):
        # Step 1: Fit the pseudo-Voigt peak
        self.fit_lorentz_peak()
        
        # Step 2: Get initial model scaling factor
        initial_scaling_factor = self.initial_model_scaling()
        
        # Step 3: Optimize the scaling factor
        exp_q, exp_iq, model_iq_interp = self._apply_q_range(self.qmin, self.qmax)
        initial_guess = [initial_scaling_factor]
        bounds = ([0], [np.inf])
        result = optimize.least_squares(self.residual, initial_guess, bounds=bounds,
                                        args=(exp_q, exp_iq, model_iq_interp))
        self.scaling_factor = result.x[0]

        # Debug: Check the optimized scaling factor
        print(f"Optimized scaling factor: {self.scaling_factor}")
    
    def plot_fit(self):
        """
        Plot the experimental data, the scaled model data, the pseudo-Voigt peak, 
        the combined model (scaled model + pseudo-Voigt peak), and the residuals.
        All plots will be displayed with log-log axes. The first panel is commented out.
        """
        exp_q, exp_iq, model_iq_interp = self._apply_q_range(self.qmin, self.qmax)
        model_iq_scaled = self.scaling_factor * model_iq_interp
        
        # Pseudo-Voigt peak contribution over the full exp_q range
        pseudo_voigt_contribution = np.zeros_like(exp_q)
        if self.pseudo_voigt_peak is not None:
            pseudo_voigt_contribution = self.pseudo_voigt_peak(exp_q)
        
        # Combined model (scaled model + pseudo-Voigt peak)
        model_iq_combined = model_iq_scaled + pseudo_voigt_contribution
        
        plt.figure(figsize=(10, 10))
        
        # Panel 1: Complete fit (experimental data with combined model) - Commented Out
        # plt.subplot(3, 1, 1)
        # plt.loglog(exp_q, exp_iq, 'bo', label='Experimental Data')
        # plt.loglog(exp_q, model_iq_combined, 'r-', label='Complete Fit')
        # plt.xlabel('q (Å⁻¹)')
        # plt.ylabel('I(q)')
        # plt.title('Complete SAXS Data Fit')
        # plt.legend()
        # plt.grid(True, which="both", ls="--")
        
        # Panel 1 (Now): Components of the fit
        plt.subplot(2, 1, 1)
        plt.loglog(exp_q, exp_iq, 'bo', label='Experimental Data')
        plt.loglog(exp_q, pseudo_voigt_contribution, 'm-.', label='Pseudo-Voigt Peak')
        plt.loglog(exp_q, model_iq_combined, 'r-', label='Scaled Model + Peak')
        plt.xlabel('q (Å⁻¹)')
        plt.ylabel('I(q)')
        plt.title('Components of the SAXS Data Fit')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        
        # Panel 2 (Now): Residuals on a log-log scale
        plt.subplot(2, 1, 2)
        residuals = np.abs((exp_iq - model_iq_combined) / exp_iq)
        plt.loglog(exp_q, residuals, 'k-')
        plt.axhline(1e-10, color='r', linestyle='--')
        plt.xlabel('q (Å⁻¹)')
        plt.ylabel('Residuals (log scale)')
        plt.grid(True, which="both", ls="--")
        
        plt.tight_layout()
        plt.show()
