import os
import numpy as np

def build_screened_interaction_W(chi0, V):
    """
    Constructs the RPA dielectric matrix and the screened Coulomb interaction W.
    """
    dim_ia = V.shape[0]   # dim_ia = 171

    # 1. Multiply V by chi0
    # Broadcasting chi0 (1D) across the columns of V (2D) simulates V * (diagonal chi0)
    V_chi0 = V * chi0[np.newaxis, :]   # converts chi0 from (171, ) to (1, 171). V_chi0 has shape (171, 171)
    
    # 2. Build the Dielectric Matrix: epsilon = I - V*chi0
    identity = np.eye(dim_ia, dtype=complex)   # creates an identity matrix
    epsilon = identity - V_chi0
    
    # 3. Invert the Dielectric Matrix
    epsilon_inv = np.linalg.inv(epsilon)
    
    # 4. Calculate Screened Interaction: W = epsilon^-1 * V
    # Use np.dot or the @ operator for dense matrix multiplication
    W = epsilon_inv @ V
    
    return W, epsilon

if __name__ == "__main__":
    # 1. Load data from File 3
    data = np.load("data/chi0_V_results.npz")
    chi0 = data["chi0"]
    V = data["V"]
    omega = data["omega"][0]
    
    print(f"Loaded chi0 (shape {chi0.shape}) and V (shape {V.shape}) at omega = {omega}")
    
    # 2. Execute the RPA Screening Math
    W_matrix, eps_matrix = build_screened_interaction_W(chi0, V)
    
    # 3. Save W for the final GW step
    os.makedirs("data", exist_ok=True)
    np.savez(
        "data/W_results.npz",
        W_matrix=W_matrix,
        eps_matrix=eps_matrix,
        omega=np.array([omega])
    )
    
    print(f"Screened interaction W built successfully! Shape: {W_matrix.shape}")
    print("Successfully saved W to 'data/W_results.npz'")