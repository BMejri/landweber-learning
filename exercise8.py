"""
    Learning Parameters in the Landweber Iteration
    Author: Bochra Mejri 
    Lecture: Regularization Theory: From Functional Analysis to Machine Learning 
    Level: Master students
    University of Vienna - Nov 2025
"""

import numpy as np
import matplotlib.pyplot as plt

# random number with fixed seed 
rng = np.random.default_rng(42)

# -----------------------------
# FEM 1D c-example 
# -----------------------------
def Forward(x, f, nele):
    """
        Example 1.1.1: c-example
        Solve -y"(s) + x(s) y(s) = f(s) for s in (0,1) with Dirichlet BCs y(0) = y(1) = 0
        Discretization using linear splines on a uniform grid of (nele + 1) nodes 
        Inputs: 
            x     : function - coefficient function x(s)
            f     : function - right-hand side f(s)
            nele  : int - number of elements  
        Outputs: 
            nodes : ndarray - mesh nodes in (0,1)    
            y     : ndarray - FEM solution at nodes y(s)  
    """
    # Mesh Parameters
    h = 1.0 / nele                                                 # element size
    nodes = np.linspace(0.0, 1.0, nele+1)                          # mesh points

    # Local matrices 
    Ke = (1.0/h) * np.array([[1.0, -1.0], [-1.0, 1.0]])             # local stiffness matrix 
    Me = (h/6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])               # local mass matrix 

    # Global matrices 
    K = np.zeros((nele+1, nele+1))                                  # stiffness matrix  
    M = np.zeros((nele+1, nele+1))                                  # mass matrix 
    F = np.zeros(nele+1)                                            # load vector
    
    # Assembly loop 
    for e in range(nele):
        smid = 0.5 * (nodes[e] + nodes[e+1])                        # midpoint
        K[e:e+2, e:e+2] += Ke
        M[e:e+2, e:e+2] += x(smid) * Me 
        F[e:e+2] += f(smid) * (h/2.0) * np.array([1.0, 1.0])

    # Solve system with Dirichlet BCs
    G = K + M 
    y_int = np.linalg.solve(G[1:-1, 1:-1], F[1:-1])
    y = np.hstack(([0.0], y_int, [0.0]))

    return nodes, y 

# -----------------------------
# Landweber iteration 
# -----------------------------
def Landweber(x0, f, y_delta, nele, kstar, alpha):
    """
        Iteratively Regularized Landweber iteration 
        to estimate the coefficient x(s) from observed data y_delta 
        Inputs : 
            x0       : ndarray - initial guess for the coefficient x(s) at nodal
            f        : function - right-hand side function f(s)
            y_delta  : ndarray - noisy measurements at interior nodes
            nele     : int - number of finite elements     
            kstar    : int - stopping iterations 
            alpha    : float - step size  
        Outputs :
            nodes    : ndarray - mesh nodes in (0,1)
            xk_upd   : ndarray - Landweber solution at nodes
    """ 
    # Mesh 
    nodes = np.linspace(0, 1.0, nele+1)
    sint = nodes[1:-1]                                              # interior nodes
    
    # Initial guess 
    xk_vec = x0.copy()

    # Iteration loop
    for k in range(kstar):
        # print(f"Iteration {k+1}/{kstar}")
        # Forward
        xk_func = lambda s: np.interp(s, nodes, xk_vec)
        _, yk = Forward(xk_func, f, nele)

        # Residu 
        resid_vec = yk[1:-1] - y_delta
        # print("Residual norm:", np.linalg.norm(resid_vec))

        # Adjoint
        resid_func = lambda s: np.interp(s, sint, resid_vec)
        _, pk = Forward(xk_func, resid_func, nele)

        # Gradient 
        grad_vec = - yk[1:-1] * pk[1:-1]
        grad_nodes = np.zeros_like(xk_vec)
        grad_nodes[1:-1] = grad_vec / np.linalg.norm(grad_vec)

        # update 
        xk_upd = np.maximum(xk_vec - alpha * grad_nodes , 0.0)
        xk_vec = xk_upd
        
    return nodes, xk_upd

# -----------------------------
# Training data
# -----------------------------
def Training_data(x_true, x0, f, nele, ndata, noise_levels, alpha_list, kstar_list, mixed):
    """
        Inputs : 
            ndata           : int - number of training samples
            noise_levels    : list of float - noise level delta to generate noisy measurements 
                            y_delta = y + delta \xi where \xi is Gaussian white noise
            alpha_list      : list of float - candidate step sizes
            kstar_list      : list of int - candidate stopping iterations
            mixed           : bool - if Flase : all training smaples are perturbed with the smae 
                            noise level delta. 
                                     if True : each training sample is perturbed with a randomly 
                            chosen delta from noise_levels 
        Outputs : 
            best_alpha_delta    : optimal step sizes 
            best_kstar_delta    : optimal stopping indices 
    """
    best_alpha_delta = []
    best_kstar_delta = []

    nodes = np.linspace(0.0, 1.0, nele+1)

    if not mixed:
        for delta in noise_levels:
            print(f"\n=== Noise level delta = {delta:.3f} ===")
            x_data = []
            y_data = []
            for i in range(ndata):
                x_noisy_vec = np.array([x_true(s) for s in nodes]) + 0.1 * rng.standard_normal(nele + 1)
                x_noisy_func = lambda s : np.interp(s, nodes, x_noisy_vec)
                
                _ , y_true = Forward(x_noisy_func, f, nele)
                y_noisy = y_true[1:-1] + delta * rng.standard_normal(nele - 1)                  # Gaussian noise with mean 0 and variance 1 
                
                x_data.append(x_noisy_vec)
                y_data.append(y_noisy)

            min_error = np.inf 
            best_alpha = None
            best_kstar = None 

            for alpha in alpha_list:
                for kstar in kstar_list:
                    total_error = 0.0
                    for i in range(ndata):
                        _, x_rec = Landweber(x0, f, y_data[i], nele, kstar, alpha)
                        error = np.linalg.norm(x_data[i] - x_rec)
                        total_error += error ** 2 
                    
                    if total_error < min_error:
                        min_error = total_error
                        best_alpha, best_kstar = alpha, kstar
                        print(f"delta={delta:.3f}: alpha={alpha}, kstar={kstar}, tot_err={total_error:.4e}")

            best_alpha_delta.append(best_alpha)
            best_kstar_delta.append(best_kstar)

        return best_alpha_delta, best_kstar_delta
    
    else:
        print(f"\n=== Mixed-noise level delta ===")
        chosen_delta = rng.choice(noise_levels, size=ndata, replace=True)
        print(chosen_delta)
        x_data = []
        y_data = []
        for i in range(ndata):
            delta = chosen_delta[i]
            x_noisy_vec = np.array([x_true(s) for s in nodes]) + 0.1 * rng.standard_normal(nele + 1)
            x_noisy_func = lambda s: np.interp(s, nodes, x_noisy_vec)

            _, y_true = Forward(x_noisy_func, f, nele)
            y_noisy = y_true[1:-1] + delta * rng.standard_normal(nele - 1)
            
            x_data.append(x_noisy_vec)
            y_data.append(y_noisy)

        min_error = np.inf
        best_alpha = None
        best_kstar = None

        for alpha in alpha_list:
            for kstar in kstar_list:
                total_error = 0.0
                for i in range(ndata):
                    _, x_rec = Landweber(x0, f, y_data[i], nele, kstar, alpha)
                    error = np.linalg.norm(x_data[i] - x_rec)
                    total_error += error**2

                if total_error < min_error:
                    min_error = total_error
                    best_alpha, best_kstar = alpha, kstar
                    print(f"Mixed delta: alpha={alpha}, kstar={kstar}, tot_err={min_error:.4e}")

        best_alpha_delta.append(best_alpha)
        best_kstar_delta.append(best_kstar)

        return best_alpha_delta, best_kstar_delta

# -----------------------------
# Example 
# -----------------------------
nele = 50
x0 = np.ones(nele+1)
x_true = lambda s: 1.0 + s 
f = lambda s: (np.pi**2 + 1.0 + s) * np.sin(np.pi * s)

ndata = 10 
noise_levels = [0.01, 0.02, 0.05, 0.1]
kstar_list = [100, 500, 1000, 3000]
alpha_list = [0.01, 0.02, 0.05, 0.1]

best_alpha_delta, best_kstar_delta = Training_data(x_true, x0, f, nele, ndata, noise_levels, alpha_list, kstar_list, mixed=False)
print("Best alpha:", best_alpha_delta)
print("Best kstar:", best_kstar_delta)

