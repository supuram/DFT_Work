from pyscf import gto
from pyscf import scf
import numpy as np
np.set_printoptions(precision=8, suppress=True, linewidth=300)

mol = gto.M(
    atom="C 0 0 0",
    basis="sto-3g",
    unit="Bohr"
)

# Check from PySCF to validate my result
mf = scf.RHF(mol)
mf.kernel()

print("PySCF RHF energy =", mf.e_tot)
print("PySCF orbital energies =")
print(mf.mo_energy)

# Calculate the overlap, kinetic, potential, coulomb, exchange integrals
S = mol.intor("int1e_ovlp")
T = mol.intor("int1e_kin")
V = mol.intor("int1e_nuc")
H = T + V
eri = mol.intor("int2e")
coulomb = eri.transpose(0, 1, 3, 2)
exchange = eri.transpose(0, 2, 3, 1)

# Finding the transformatino matrix X
s, U = np.linalg.eigh(S)   
s_inv_sqrt = np.diag(1.0 / np.sqrt(s))
X = U @ s_inv_sqrt

# Initial guess for P
H_prime = X.conj().T @ H @ X
epsilon, C_prime = np.linalg.eigh(H_prime)
C = X @ C_prime
P = np.zeros((5, 5))
for i in range(3):
    P += 2.0 * np.outer(C[:, i], C[:, i])

# SCF Procedure
convergence_threshold = 1e-14
max_iterations = 100
iteration = 0
converged = False
Fock_matrix = np.zeros((5, 5))

while not converged and iteration < max_iterations:

    iteration += 1
    G = np.zeros((5, 5))
    
    for mu in range(5):
        for nu in range(5):
            G[mu, nu] = 0
            for sig in range(5):
                for lam in range(5):
                    G[mu, nu] += P[sig][lam] * (coulomb[mu][nu][lam][sig] - 0.5 * exchange[mu][nu][lam][sig])
    Fock_matrix = H + G
    Fock_matrix_prime = X.conj().T @ Fock_matrix @ X
    
    # Diagonalize F'
    epsilon, C_prime = np.linalg.eigh(Fock_matrix_prime)
    C = X @ C_prime
    P_new = np.zeros((5, 5))
    for i in range(3):
        P_new += 2.0 * np.outer(C[:, i], C[:, i])
        
    # Check convergence
    difference = np.max(np.abs(P_new - P))
    print(f"Iteration {iteration}: ΔP = {difference:.12e}")
    if difference < convergence_threshold:
        converged = True
    else:
        P = P_new
if converged:
    print("SCF converged!")
else:
    print("SCF did not converge within the maximum number of iterations.")

E = 0.5 * np.sum(P * (H + Fock_matrix))
print("My RHF energy =", E)