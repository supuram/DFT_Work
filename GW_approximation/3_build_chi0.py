import os
import numpy as np
from pyscf import gto, ao2mo

def build_polarizability_chi0(omega, mo_energy, nocc, eri_mo_full, eta=0.05):
    """
    Constructs the independent-particle polarizability chi_0(omega) in the MO basis.
    
    Parameters:
    - omega: Complex frequency (energy variable)
    - mo_energy: Kohn-Sham orbital energies
    - nocc: Number of occupied electrons
    - eri_mo_full: Molecular orbital electron repulsion integrals (physicist/chemist notation)
    - eta: Broadening factor
    
    Returns:
    - chi0: 2D matrix representing polarizability over MO index pairs (ia, jb)
    """
    nmo = mo_energy.size   # nmo = 28
    nvirt = nmo - nocc     # nvirt = 19, nvirt is the unoccupied/virtual orbitals
    
    # In MBPT, polarizability is often indexed by electron-hole pairs (i -> a, j -> b).
    # For a simplified matrix representation, we map occupied-virtual transitions.
    # Dimension of chi0 in transition space: (nocc * nvirt) x (nocc * nvirt)
    
    dim_ia = nocc * nvirt   # dim_ia = 171
    chi0 = np.zeros((dim_ia), dtype=complex)
    V = np.zeros((dim_ia, dim_ia))
    # rows are i->a transition and columns are j->b
    row_idx = 0
    for i in range(nocc):
        for a in range(nocc, nmo):
            eps_ia = mo_energy[a] - mo_energy[i]
            
            '''The subtraction you see in the code is the final, analytical result of multiplying two Green's functions together and 
            integrating over frequency.'''
            g_prod = 1.0 / (omega - eps_ia + 1j * eta) - 1.0 / (omega + eps_ia + 1j * eta)
            chi0[row_idx] = 2.0 * g_prod   # # Factor of 2 for spin degeneracy in restricted closed-shell, or handle alpha/beta explicitly
            
            col_idx = 0
            for j in range(nocc):
                for b in range(nocc, nmo):
                    V[row_idx, col_idx] = eri_mo_full[i, a, b, j]
                    col_idx += 1
            row_idx += 1
            
    return chi0, V

if __name__ == "__main__":
    # 1. Load saved DFT results
    data = np.load("data/dft_results_o2.npz")
    alpha_energies = data["mo_energy_alpha"]
    alpha_coeffs = data["mo_coeff_alpha"]
    nocc_alpha = int(data["nelec_alpha"][0])  # nocc_alpha = 9
    
    # Re-initialize PySCF molecule object quickly to compute MO integrals
    mol = gto.M(atom='O 0 0 0; O 0 0 1.208', basis='cc-pvdz', charge=0, spin=2)
    
    # 2. Transform Electron Repulsion Integrals to MO basis for Alpha spin
    eri_ao = mol.intor('int2e')  # (28, 28, 28, 28)
    eri_mo_full = ao2mo.full(eri_ao, alpha_coeffs) # (28, 28, 28, 28)
    nmo = alpha_energies.size
        
    # 3. Test building chi_0 at a specific test frequency omega = 0.1 Ha
    omega_test = 0.1
    chi0, V = build_polarizability_chi0(omega_test, alpha_energies, nocc_alpha, eri_mo_full)
    os.makedirs("data", exist_ok=True)
    np.savez(
        "data/chi0_V_results.npz",
        chi0=chi0,
        V=V,
        omega=np.array([omega_test])
    )