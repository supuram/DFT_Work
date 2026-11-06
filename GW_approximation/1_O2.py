import numpy as np
from pyscf import gto, scf

# 1. Define the O2 molecule (bond length ~ 1.208 Angstroms)
# Triplet state means spin = 2 (2 un-paired electrons: 2s+1 = 3 -> spin = 2 * S = 2)
mol = gto.M(
    atom='O 0 0 0; O 0 0 1.208',
    basis='cc-pvdz',
    charge=0,
    spin=2 
)

# 2. Run Unrestricted Kohn-Sham DFT (UKS) using PBE functional
mf = scf.UKS(mol)
mf.xc = 'Pbe,Pbe'
E_tot = mf.kernel()

# 3. Extract the non-interacting Kohn-Sham components
# For UKS, mf.mo_energy and mf.mo_coeff come as a tuple: (alpha_set, beta_set)
mo_energy_alpha, mo_energy_beta = mf.mo_energy   # Both have shapes (28, )
mo_coeff_alpha, mo_coeff_beta = mf.mo_coeff      # Both have shapes (28, 28)
nelec_alpha, nelec_beta = mol.nelec   # nelec_alpha = 9, nelec_beta = 7, nelec stands for Number of Electrons.
s_matrix = mf.get_ovlp()    # Overlap matrix S in atomic orbital (AO) basis, Has shape (28, 28)

np.savez(
    "data/dft_results_o2.npz",
    mo_energy_alpha=mo_energy_alpha,
    mo_energy_beta=mo_energy_beta,
    mo_coeff_alpha=mo_coeff_alpha,
    mo_coeff_beta=mo_coeff_beta,
    nelec_alpha=np.array([nelec_alpha]),
    nelec_beta=np.array([nelec_beta]),
    s_matrix=s_matrix
)

print(f"Number of basis functions (AO size): {s_matrix.shape[0]}")
print(f"Alpha HOMO energy: {mo_energy_alpha[mol.nelec[0]-1]:.4f} Ha")
print(f"Alpha LUMO energy: {mo_energy_alpha[mol.nelec[0]]:.4f} Ha")

''' Output
WARN: HOMO -0.234564635385581 >= LUMO -0.234564635385584

converged SCF energy = -150.193280761031  <S^2> = 2.0029745  2S+1 = 3.0019823
DFT Total Energy for O2: -150.193281 Ha
Number of basis functions (AO size): 28
Alpha HOMO energy: -0.2302 Ha
Alpha LUMO energy: 0.1612 Ha
'''