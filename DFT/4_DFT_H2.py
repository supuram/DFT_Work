# The book used here for the equations is , A chemists guide to DFT by Koch and Holthausen

import numpy as np
from pyscf import dft, gto, scf
import scipy.linalg

# 1. Define Molecule (spin=0 for closed-shell singlet H2)
mol = gto.M(
    atom="H 0 0 0; H 0 0 1",
    basis="sto-3g",
    unit="Bohr",
    spin=0,
    verbose=0
)

# Number of electrons and occupied spatial orbitals
n_electrons = mol.nelectron  # 2 for H2
n_occ = n_electrons // 2     # 1 doubly occupied orbital

# 2. Compute Constant Matrices (Independent of density)
# Overlap matrix S (Equation 7-8)
S = mol.intor('int1e_ovlp')

# Core Hamiltonian: Kinetic (T) + Nuclear attraction (V_Ne) (Equation 7-13)
T = mol.intor('int1e_kin')
V_Ne = mol.intor('int1e_nuc')
H_core = T + V_Ne

# Nuclear repulsion energy E_nuc (constant)
E_nuc = mol.energy_nuc()

# 3. Set up the numerical integration grid for V_xc (Section 7.4)
grid = dft.gen_grid.Grids(mol)
grid.level = 3
grid.build()

xc_code = 'LDA,VWN'  # Standard LDA functional (Slater + VWN5)

# 4. Initial Guess for Density Matrix P (from H_core)
# Solve H_core * C = S * C * eps
eps, C = scipy.linalg.eigh(H_core, S) # directly solves, H_core * C = S * C * eps
C_occ = C[:, :n_occ]
# Density matrix P_mu_nu = 2 * sum_i c_mu_i * c_nu_i (factor of 2 for closed-shell)
P = 2.0 * np.dot(C_occ, C_occ.T)

print("Starting manual Kohn-Sham SCF iterations...\n")

# 5. SCF Iteration Loop
max_cycles = 50
conv_tol = 1e-8
E_old = 0.0

for cycle in range(max_cycles):
    # A. Classical Coulomb Matrix J (Equation 7-16)
    J, K = scf.hf.get_jk(mol, P)

    # B. Exchange-Correlation Potential V_xc and Energy E_xc (Equation 7-17 / 7-34)
    # nr_vxc evaluates the grid integration using Libxc
    n_elec_grid, exc, vxc = dft.numint.nr_vxc(mol, grid, xc_code, P)

    # C. Build the Kohn-Sham Matrix (Equation 7-12)
    F_KS = H_core + J + vxc

    # D. Compute Total Electronic + Nuclear Energy
    # E_total = Tr(P * H_core) + 0.5 * Tr(P * J) + E_xc + E_nuc
    E_one_electron = np.einsum('ij,ji->', P, H_core)   # eqn 5-13
    E_coulomb = 0.5 * np.einsum('ij,ji->', P, J)   # eqn 5-13
    E_total = E_one_electron + E_coulomb + exc + E_nuc

    delta_E = abs(E_total - E_old)
    print(f"Cycle {cycle:2d}: Energy = {E_total:.8f} Eh, Delta E = {delta_E:.3e}")

    if delta_E < conv_tol:
        print("\nSCF successfully converged!")
        break

    E_old = E_total

    # E. Diagonalize the generalized eigenvalue problem: F_KS * C = S * C * eps
    eps, C = scipy.linalg.eigh(F_KS, S)

    # F. Update Density Matrix for next cycle (Equation 7-14 / 7-15)
    C_occ = C[:, :n_occ]
    P = 2.0 * np.dot(C_occ, C_occ.T)