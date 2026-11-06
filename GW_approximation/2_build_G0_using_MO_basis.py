import numpy as np

def load_dft_data(filename="data/dft_results_o2.npz"):
    """Loads pre-calculated DFT ground state data from a saved archive."""
    try:
        data = np.load(filename)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find '{filename}'. Run your DFT script first to generate it.")
    
    return {
        "mo_energy_alpha": data["mo_energy_alpha"],
        "mo_energy_beta": data["mo_energy_beta"],
        "mo_coeff_alpha": data["mo_coeff_alpha"],
        "mo_coeff_beta": data["mo_coeff_beta"],
        "nocc_alpha": int(data["nelec_alpha"][0]),
        "nocc_beta": int(data["nelec_beta"][0]),
        "s_matrix": data["s_matrix"]
    }

def get_G0_mo(omega, mo_energy, nocc, eta=0.01):
    """
    Constructs the frequency-dependent non-interacting Green's function G0(omega) 
    in the Molecular Orbital (MO) basis for a single spin channel.
    """
    nmo = mo_energy.size   # nmo = 28
    homo_idx = nocc - 1
    lumo_idx = nocc
    mu = (mo_energy[homo_idx] + mo_energy[lumo_idx]) / 2.0
    
    G0_mo = np.zeros((nmo, nmo), dtype=complex)
    
    for p in range(nmo):
        eps_p = mo_energy[p]
        if eps_p <= mu:
            denominator = omega - eps_p + 1j * eta
        else:
            denominator = omega - eps_p - 1j * eta
            
        G0_mo[p, p] = 1.0 / denominator
        
    return G0_mo

if __name__ == "__main__":
    # Load data instantly without running DFT
    dft = load_dft_data()
    print("Loaded DFT data successfully from disk!\n")
    
    omega_test = 0.0
    
    G0_alpha = get_G0_mo(omega=omega_test, mo_energy=dft["mo_energy_alpha"], nocc=dft["nocc_alpha"])
    G0_beta = get_G0_mo(omega=omega_test, mo_energy=dft["mo_energy_beta"], nocc=dft["nocc_beta"])
    
    print(f"--- G0 at omega = {omega_test} Ha ---")
    print(f"Alpha HOMO G0 value: {G0_alpha[dft['nocc_alpha']-1, dft['nocc_alpha']-1]:.4f}")
    print(f"Beta HOMO G0 value: {G0_beta[dft['nocc_beta']-1, dft['nocc_beta']-1]:.4f}")
    
''' Output
--- G0 at omega = 0.0 Ha ---
Alpha HOMO G0 value: 4.3368-0.1884j
Beta HOMO G0 value: 2.4280-0.0590j
'''
