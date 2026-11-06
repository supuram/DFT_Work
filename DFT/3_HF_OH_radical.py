from pyscf import gto
from pyscf import scf
import numpy as np
np.set_printoptions(precision=8, suppress=True, linewidth=300)

mol = gto.M(
    atom="O 0 0 0; H 0 0 1.8326",
    basis="sto-3g",
    unit="Bohr",
    spin=1
)

# Check from PySCF to validate my result
mf = scf.UHF(mol)
mf.kernel()

print("PySCF UHF energy =", mf.e_tot)
print("PySCF orbital energies =")
print(mf.mo_energy)
print(mol.ao_labels())

# Calculate the overlap, kinetic, potential, coulomb, exchange integrals
S = mol.intor("int1e_ovlp")
T = mol.intor("int1e_kin")
V = mol.intor("int1e_nuc")
H = T + V
eri = mol.intor("int2e")
coulomb = eri.transpose(0, 1, 3, 2)
exchange = eri.transpose(0, 2, 3, 1)

nbasis = S.shape[0]  # counts the number of orbitals

# Finding the transformatino matrix X
s, U = np.linalg.eigh(S)   
s_inv_sqrt = np.diag(1.0 / np.sqrt(s))
X = U @ s_inv_sqrt

# Initial guess for P
H_prime = X.conj().T @ H @ X
epsilon, C_prime = np.linalg.eigh(H_prime)
C = X @ C_prime
C_alpha = C.copy()
C_beta = C.copy()
P_alpha = np.zeros((nbasis, nbasis))
P_beta = np.zeros((nbasis, nbasis))
for i in range(5):
    P_alpha += np.outer(C_alpha[:, i], C_alpha[:, i])

for i in range(4):
    P_beta += np.outer(C_beta[:, i], C_beta[:, i])
    
P = P_alpha + P_beta

# SCF Procedure
convergence_threshold = 1e-14
max_iterations = 100
iteration = 0
converged = False
Fock_matrix_alpha = np.zeros((nbasis, nbasis))
Fock_matrix_beta = np.zeros((nbasis, nbasis))

while not converged and iteration < max_iterations:

    iteration += 1
    G_alpha = np.zeros((nbasis, nbasis))
    G_beta = np.zeros((nbasis, nbasis))
    
    for mu in range(nbasis):
        for nu in range(nbasis):
            G_alpha[mu, nu] = 0
            G_beta[mu, nu] = 0
            for sig in range(nbasis):
                for lam in range(nbasis):
                    G_alpha[mu, nu] += (P[sig][lam] * coulomb[mu][nu][lam][sig]) - (P_alpha[sig][lam] * exchange[mu][nu][lam][sig])
                    G_beta[mu, nu] += (P[sig][lam] * coulomb[mu][nu][lam][sig]) - (P_beta[sig][lam] * exchange[mu][nu][lam][sig])
    Fock_matrix_alpha = H + G_alpha
    Fock_matrix_beta = H + G_beta
    Fock_matrix_alpha_prime = X.conj().T @ Fock_matrix_alpha @ X
    Fock_matrix_beta_prime = X.conj().T @ Fock_matrix_beta @ X
    
    # Diagonalize F'
    epsilon_alpha, C_prime_alpha = np.linalg.eigh(Fock_matrix_alpha_prime)
    C_alpha = X @ C_prime_alpha
    epsilon_beta, C_prime_beta = np.linalg.eigh(Fock_matrix_beta_prime)
    C_beta = X @ C_prime_beta
    P_new_alpha = np.zeros((nbasis, nbasis))
    P_new_beta = np.zeros((nbasis, nbasis))
    for i in range(5):
        P_new_alpha += np.outer(C_alpha[:, i], C_alpha[:, i])
    for i in range(4):
        P_new_beta += np.outer(C_beta[:, i], C_beta[:, i])
        
    # Check convergence
    difference_alpha = np.max(np.abs(P_new_alpha - P_alpha))
    difference_beta = np.max(np.abs(P_new_beta - P_beta))
    print(f"Iteration {iteration}: "f"ΔP_alpha = {difference_alpha:.12e}, "f"ΔP_beta = {difference_beta:.12e}")
    if difference_alpha < convergence_threshold and difference_beta < convergence_threshold:
        converged = True
    else:
        P_alpha = P_new_alpha
        P_beta = P_new_beta
        P = P_alpha + P_beta
if converged:
    print("SCF converged!")
else:
    print("SCF did not converge within the maximum number of iterations.")

E_elec = 0.5 * (np.sum(P_alpha * (H + Fock_matrix_alpha)) + np.sum(P_beta * (H + Fock_matrix_beta)))
E_total = E_elec + mol.energy_nuc()

print("My UHF electronic energy =", E_elec)
print("My UHF total energy =", E_total)