import numpy as np

def calculate_homo_self_energy(nocc, nvirt, V, W):
    """
    Calculates the static correlation self-energy for the HOMO.
    """
    # 1. Isolate the Correlation part of the interaction
    # The bare V is already in DFT. We only want the *new* shielding effect.
    W_corr = W - V
    
    # 2. Identify the HOMO index (0-indexed, so if 9 occupied, HOMO is 8)
    homo_idx = nocc - 1
    
    # 3. Sum over all virtual states to get the Self-Energy
    sigma_correlation = 0.0
    
    # We need the diagonal elements W_{nm, nm}. 
    # In our 2D transition mapping, we find the row/col index for HOMO -> a
    for a in range(nocc, nocc + nvirt):
        # Calculate exactly which row this transition lives in
        transition_idx = (homo_idx * nvirt) + (a - nocc)
        
        # Extract the diagonal matrix element for this specific transition
        w_c_matrix_element = W_corr[transition_idx, transition_idx]
        
        # Add it to the self-energy (The 1/2 comes from the COHSEX analytic integration)
        sigma_correlation += 0.5 * w_c_matrix_element
        
    return np.real(sigma_correlation)

if __name__ == "__main__":
    # 1. Load data from previous steps
    dft_data = np.load("data/dft_results_o2.npz")
    gw_data = np.load("data/W_results.npz")
    chi0_data = np.load("data/chi0_V_results.npz")
    
    alpha_energies = dft_data["mo_energy_alpha"]
    nocc = int(dft_data["nelec_alpha"][0])   # 9
    nmo = alpha_energies.size   # 28
    nvirt = nmo - nocc   # 19
    
    W_matrix = gw_data["W_matrix"]
    V_matrix = chi0_data["V"]
    
    # 2. Extract the raw DFT energy for the HOMO
    homo_idx = nocc - 1
    e_dft_homo = alpha_energies[homo_idx]
    
    print(f"Raw DFT HOMO Energy: {e_dft_homo:.6f} Hartrees")
    
    # 3. Calculate the GW Self-Energy Correction
    sigma_c = calculate_homo_self_energy(nocc, nvirt, V_matrix, W_matrix)
    
    print(f"GW Correlation Self-Energy (Sigma): {sigma_c:.6f} Hartrees")
    
    # 4. Calculate Final Quasiparticle Energy
    # (Note: In a full code, we also subtract the DFT exchange-correlation Vxc 
    # and add exact Hartree-Fock exchange. Here we apply the pure correlation shift).
    e_gw_homo = e_dft_homo + sigma_c
    
    print(f"Corrected GW HOMO Energy: {e_gw_homo:.6f} Hartrees")