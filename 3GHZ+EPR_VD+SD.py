import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import qnetvo as qnet
from scipy.stats import sem
import math
import time
from scipy.optimize import minimize
from pennylane import classical_shadow, ClassicalShadow
from functools import reduce
from collections import defaultdict
from string import ascii_letters as ABC

start_time = time.time()

matrix_ideal_uncer = [[1,1,1,2,2], [1,1,1,2,2], [1,1,1,2,2], [2,2,2,1,0], [2,2,2,0,1]]
matrix_ideal_mutualinfo = [[1,1,1,0,0], [1,1,1,0,0], [1,1,1,0,0], [0,0,0,1,1], [0,0,0,1,1]]
matrix_ideal_covar = [[1,1,1,0,0], [1,1,1,0,0], [1,1,1,0,0], [0,0,0,1,1], [0,0,0,1,1]]

B = [[1, 0, 0, 0], [0, 1/math.sqrt(2), -1/math.sqrt(2), 0], [0, 1/math.sqrt(2), 1/math.sqrt(2), 0], [0, 0, 0, 1]]
D = [[1, 0, 0, 0], [0, 1/math.sqrt(2), -1j/math.sqrt(2), 0], [0, 1/math.sqrt(2), 1j/math.sqrt(2), 0], [0, 0, 0, 1]]
sd_Uncertainty_matrices = []
sd_MutualInfo_matrices = []
sd_Covariance_matrices = []
shot_lists = [1250]
shot_lists1 = [2500]
shot_lists2 = [5000]
shot_lists3 = [10000]
record_matrix_uncer = [[[] for _ in range (5)] for _ in range(5)]
record_matrix_mutualinfo = [[[] for _ in range (5)] for _ in range(5)]
record_matrix_covar = [[[] for _ in range (5)] for _ in range(5)]
n_record_matrix_uncer = [[[] for _ in range (5)] for _ in range(5)]
n_record_matrix_mutualinfo = [[[] for _ in range (5)] for _ in range(5)]
n_record_matrix_covar = [[[] for _ in range (5)] for _ in range(5)]
sd_record_matrix_uncer = [[[] for _ in range (5)] for _ in range(5)]
sd_record_matrix_mutualinfo = [[[] for _ in range (5)] for _ in range(5)]
sd_record_matrix_covar = [[[] for _ in range (5)] for _ in range(5)]

record_dists_uncer = [[] for _ in range(len(shot_lists))]
record_dists_mutualinfo = [[] for _ in range(len(shot_lists))]
record_dists_covar = [[] for _ in range(len(shot_lists))]
n_record_dists_uncer = [[] for _ in range(len(shot_lists))]
n_record_dists_mutualinfo = [[] for _ in range(len(shot_lists))]
n_record_dists_covar = [[] for _ in range(len(shot_lists))]
sd_record_dists_uncer = []
sd_record_dists_mutualinfo = []
sd_record_dists_covar = []

def cast(matrix, dtype):
    return matrix.astype(dtype)
def partial_trace(matrix, indices, c_dtype="complex128"):
    matrix = cast(matrix, dtype=c_dtype)
    if qml.math.ndim(matrix) == 2:
        is_batched = False
        batch_dim, dim = 1, matrix.shape[1]
    else:
        is_batched = True
        batch_dim, dim = matrix.shape[:2]

    # Dimension and reshape
    num_indices = int(np.log2(dim))
    rho_dim = 2 * num_indices

    matrix = np.reshape(matrix, [batch_dim] + [2] * 2 * num_indices)
    indices = np.sort(indices)

    # For loop over wires
    for i, target_index in enumerate(indices):
        target_index = target_index - i
        state_indices = ABC[1 : rho_dim - 2 * i + 1]
        state_indices = list(state_indices)

        target_letter = state_indices[target_index]
        state_indices[target_index + num_indices - i] = target_letter
        state_indices = "".join(state_indices)

        einsum_indices = f"a{state_indices}"
        matrix = np.einsum(einsum_indices, matrix)

    number_wires_sub = num_indices - len(indices)
    reduced_density_matrix = np.reshape(
        matrix, (batch_dim, 2**number_wires_sub, 2**number_wires_sub)
    )
    return reduced_density_matrix if is_batched else reduced_density_matrix[0]

def format_sci(value, precision):
    """格式化为科学计数法，指数部分去掉前导零"""
    formatted = f"{value:.{precision}e}"  # 先格式化为科学计数法
    base, exponent = formatted.split("e")  # 拆分底数和指数
    exponent = exponent.replace("+0", "+").replace("-0", "-")  # 去掉指数前导零
    return f"{base}e{exponent}"
    
def project_to_simplex(P):
    def objective(P_prime):
        return np.sum((P_prime - P) ** 2)
    constraints = ({'type': 'eq', 'fun': lambda P_prime: np.sum(P_prime) - 1})
    bounds = [(0, None) for _ in range(len(P))]

    P0 = np.maximum(P, 0)
    P0 = P0 / np.sum(P0) if np.sum(P0) > 0 else np.ones_like(P) / len(P)

    result = minimize(objective, P0, bounds=bounds, constraints=constraints)
    return result.x
    
def Probs_VDproc(Probs_pre1, Probs_pre2, Probs_pre3, Probs_pre4):
	Probs_basis1 = []
	Probs_basis2 = []
	IVD_basis1 = Probs_pre1[0] - Probs_pre1[1] - Probs_pre1[2] + Probs_pre1[3] + Probs_pre1[4] + Probs_pre1[5] - Probs_pre1[6] - Probs_pre1[7] + Probs_pre1[8] - Probs_pre1[9] + Probs_pre1[10] - Probs_pre1[11] + Probs_pre1[13] + Probs_pre1[13] + Probs_pre1[14] + Probs_pre1[15]
	IVD_basis2 = Probs_pre2[0] - Probs_pre2[1] - Probs_pre2[2] + Probs_pre2[3] + Probs_pre2[4] + Probs_pre2[5] - Probs_pre2[6] - Probs_pre2[7] + Probs_pre2[8] - Probs_pre2[9] + Probs_pre2[10] - Probs_pre2[11] + Probs_pre2[13] + Probs_pre2[13] + Probs_pre2[14] + Probs_pre2[15]
	Z1VD = (Probs_pre1[0] - Probs_pre1[1] + Probs_pre1[4] + Probs_pre1[5] - Probs_pre1[10] + Probs_pre1[11] - Probs_pre1[14] - Probs_pre1[15]) / IVD_basis1
	Z2VD = (Probs_pre1[0] - Probs_pre1[2] - Probs_pre1[5] + Probs_pre1[7] + Probs_pre1[8] + Probs_pre1[10] - Probs_pre1[13] - Probs_pre1[15]) / IVD_basis1
	X1VD = (Probs_pre2[0] - Probs_pre2[1] + Probs_pre2[4] + Probs_pre2[5] - Probs_pre2[10] + Probs_pre2[11] - Probs_pre2[14] - Probs_pre2[15]) / IVD_basis2
	X2VD = (Probs_pre2[0] - Probs_pre2[2] - Probs_pre2[5] + Probs_pre2[6] + Probs_pre2[8] + Probs_pre2[10] - Probs_pre2[13] - Probs_pre2[15]) / IVD_basis2
	Z1Z2VD = (Probs_pre3[0] - Probs_pre3[3] - Probs_pre3[5] + Probs_pre3[6] + Probs_pre3[9] - Probs_pre3[10] - Probs_pre3[12] + Probs_pre3[15]) / IVD_basis1
	X1X2VD = (Probs_pre4[0] - Probs_pre4[3] - Probs_pre4[5] + Probs_pre4[6] + Probs_pre4[9] - Probs_pre4[10] - Probs_pre4[12] + Probs_pre4[15]) / IVD_basis2
	Probs_basis1.append((1 + Z1Z2VD + Z1VD + Z2VD) / 4) #P00_basis1
	Probs_basis1.append((1 - Z1Z2VD + Z1VD - Z2VD) / 4) #P01_basis1
	Probs_basis1.append((1 - Z1Z2VD - Z1VD + Z2VD) / 4) #P10_basis1
	Probs_basis1.append((1 + Z1Z2VD - Z1VD - Z2VD) / 4) #P11_basis1
	Probs_basis2.append((1 + X1X2VD + X1VD + X2VD) / 4) #P00_basis2
	Probs_basis2.append((1 - X1X2VD + X1VD - X2VD) / 4) #P01_basis2
	Probs_basis2.append((1 - X1X2VD - X1VD + X2VD) / 4) #P10_basis2
	Probs_basis2.append((1 + X1X2VD - X1VD - X2VD) / 4) #P11_basis2
	probs_basis1 = project_to_simplex(Probs_basis1) #project the unvalid mitigated probs to the probability simplex
	probs_basis2 = project_to_simplex(Probs_basis2)
	return probs_basis1, probs_basis2
	
def Probs_VDproc_local(Probs_pre1, Probs_pre2, Probs_pre3, Probs_pre4):
	Probs_basis1 = []
	Probs_basis2 = []
	IVD_basis1 = Probs_pre1[0] - Probs_pre1[1] + Probs_pre1[2] + Probs_pre1[3]
	IVD_basis2 = Probs_pre2[0] - Probs_pre2[1] + Probs_pre2[2] + Probs_pre2[3]
	ZVD = (Probs_pre1[0] - Probs_pre1[3]) / IVD_basis1
	XVD = (Probs_pre2[0] - Probs_pre2[3]) / IVD_basis2
	Probs_basis1.append((1 + ZVD) / 2) #P0_basis1
	Probs_basis1.append((1 - ZVD) / 4) #P1_basis1
	Probs_basis2.append((1 + XVD) / 4) #P0_basis2
	Probs_basis2.append((1 + XVD) / 4) #P1_basis2
	probs_basis1 = project_to_simplex(Probs_basis1) #project the unvalid mitigated probs to the probability simplex
	probs_basis2 = project_to_simplex(Probs_basis2)
	return probs_basis1, probs_basis2
	
def Probs_VDproc_alt(Probs_pre1, Probs_pre3):
	Probs = []
	IVD = Probs_pre1[0] - Probs_pre1[1] - Probs_pre1[2] + Probs_pre1[3] + Probs_pre1[4] + Probs_pre1[5] - Probs_pre1[6] - Probs_pre1[7] + Probs_pre1[8] - Probs_pre1[9] + Probs_pre1[10] - Probs_pre1[11] + Probs_pre1[13] + Probs_pre1[13] + Probs_pre1[14] + Probs_pre1[15]
	Z1VD = (Probs_pre1[0] - Probs_pre1[1] + Probs_pre1[4] + Probs_pre1[5] - Probs_pre1[10] + Probs_pre1[11] - Probs_pre1[14] - Probs_pre1[15]) / IVD
	Z2VD = (Probs_pre1[0] - Probs_pre1[2] - Probs_pre1[5] + Probs_pre1[7] + Probs_pre1[8] + Probs_pre1[10] - Probs_pre1[13] - Probs_pre1[15]) / IVD
	Z1Z2VD = (Probs_pre3[0] - Probs_pre3[3] - Probs_pre3[5] + Probs_pre3[6] + Probs_pre3[9] - Probs_pre3[10] - Probs_pre3[12] + Probs_pre3[15]) / IVD
	Probs.append((1 + Z1Z2VD + Z1VD + Z2VD) / 4) #P00
	Probs.append((1 - Z1Z2VD + Z1VD - Z2VD) / 4) #P01
	Probs.append((1 - Z1Z2VD - Z1VD + Z2VD) / 4) #P10
	Probs.append((1 + Z1Z2VD - Z1VD - Z2VD) / 4) #P11
	probs = project_to_simplex(Probs) #project the unvalid mitigated probs to the probability simplex
	return probs
	
def Probs_VDproc_local_alt(Probs_pre1):
	Probs = []
	IVD = Probs_pre1[0] - Probs_pre1[1] + Probs_pre1[2] + Probs_pre1[3]
	ZVD = (Probs_pre1[0] - Probs_pre1[3]) / IVD
	Probs.append((1 + ZVD) / 2) #P0
	Probs.append((1 - ZVD) / 4) #P1
	probs = project_to_simplex(Probs) #project the unvalid mitigated probs to the probability simplex
	return probs
	
def Covariance_calculation(Probs):
	return Probs[0] - Probs[1] - Probs[2] + Probs[3] - (Probs[0] + Probs[1] - Probs[2] - Probs[3]) * (Probs[0] - Probs[1] + Probs[2] - Probs[3])

def circuit_1(ns, gamma1, gamma2):
	dev = qml.device("default.mixed", wires=range(10), shots=ns)
	@qml.qnode(dev)
	def qnode(Ns, Gamma1, Gamma2):
		qml.Hadamard(wires=0)
		qml.CNOT(wires=[0, 1])
		qml.CNOT(wires=[0, 2])
		qml.Hadamard(wires=3)
		qml.CNOT(wires=[3, 4])
		qml.Hadamard(wires=5)
		qml.CNOT(wires=[5, 6])
		qml.CNOT(wires=[5, 7])
		qml.Hadamard(wires=8)
		qml.CNOT(wires=[8, 9])
		for j in [0,1,3,5,6,8]:
			qml.DepolarizingChannel(Gamma1, wires=j)
		for k in [2,4,7,9]:
			qml.DepolarizingChannel(Gamma2, wires=k)
		qml.QubitUnitary(B, wires=[0,5])
		qml.QubitUnitary(B, wires=[1,6])
		qml.QubitUnitary(B, wires=[2,7])
		qml.QubitUnitary(B, wires=[3,8])
		qml.QubitUnitary(B, wires=[4,9])
		return qml.probs(wires=[0,5]), qml.probs(wires=[1,6]), qml.probs(wires=[2,7]), qml.probs(wires=[3,8]), qml.probs(wires=[4,9]), \
		qml.probs(wires=[1,0,6,5]), qml.probs(wires=[2,0,7,5]), qml.probs(wires=[3,0,8,5]), qml.probs(wires=[4,0,9,5]),\
		qml.probs(wires=[2,1,7,6]), qml.probs(wires=[3,1,8,6]), qml.probs(wires=[4,1,9,6]),\
		qml.probs(wires=[3,2,8,7]), qml.probs(wires=[4,2,9,7]), qml.probs(wires=[4,3,9,8])
	meas_probs1 = qnode(ns, gamma1, gamma2)
	return meas_probs1
def circuit_2(ns, gamma1, gamma2):
	dev = qml.device("default.mixed", wires=range(10), shots=ns)
	@qml.qnode(dev)
	def qnode(Ns, Gamma1, Gamma2):
		qml.Hadamard(wires=0)
		qml.CNOT(wires=[0, 1])
		qml.CNOT(wires=[0, 2])
		qml.Hadamard(wires=3)
		qml.CNOT(wires=[3, 4])
		qml.Hadamard(wires=5)
		qml.CNOT(wires=[5, 6])
		qml.CNOT(wires=[5, 7])
		qml.Hadamard(wires=8)
		qml.CNOT(wires=[8, 9])
		for j in [0,1,3,5,6,8]:
			qml.DepolarizingChannel(Gamma1, wires=j)
		for k in [2,4,7,9]:
			qml.DepolarizingChannel(Gamma2, wires=k)
		qml.QubitUnitary(B, wires=[0,5])
		qml.QubitUnitary(B, wires=[1,6])
		qml.QubitUnitary(B, wires=[2,7])
		qml.QubitUnitary(B, wires=[3,8])
		qml.QubitUnitary(B, wires=[4,9])
		for l in range(10):
		    qml.Hadamard(wires=l)
		return qml.probs(wires=[0,5]), qml.probs(wires=[1,6]), qml.probs(wires=[2,7]), qml.probs(wires=[3,8]), qml.probs(wires=[4,9]), \
		qml.probs(wires=[1,0,6,5]), qml.probs(wires=[2,0,7,5]), qml.probs(wires=[3,0,8,5]), qml.probs(wires=[4,0,9,5]),\
		qml.probs(wires=[2,1,7,6]), qml.probs(wires=[3,1,8,6]), qml.probs(wires=[4,1,9,6]),\
		qml.probs(wires=[3,2,8,7]), qml.probs(wires=[4,2,9,7]), qml.probs(wires=[4,3,9,8])	
	meas_probs2 = qnode(ns, gamma1, gamma2)
	return meas_probs2
def circuit_3(ns, gamma1, gamma2):
	dev = qml.device("default.mixed", wires=range(10), shots=ns)
	@qml.qnode(dev)
	def qnode(Ns, Gamma1, Gamma2):
		qml.Hadamard(wires=0)
		qml.CNOT(wires=[0, 1])
		qml.CNOT(wires=[0, 2])
		qml.Hadamard(wires=3)
		qml.CNOT(wires=[3, 4])
		qml.Hadamard(wires=5)
		qml.CNOT(wires=[5, 6])
		qml.CNOT(wires=[5, 7])
		qml.Hadamard(wires=8)
		qml.CNOT(wires=[8, 9])
		for j in [0,1,3,5,6,8]:
			qml.DepolarizingChannel(Gamma1, wires=j)
		for k in [2,4,7,9]:
			qml.DepolarizingChannel(Gamma2, wires=k)
		qml.QubitUnitary(D, wires=[0,5])
		qml.QubitUnitary(D, wires=[1,6])
		qml.QubitUnitary(D, wires=[2,7])
		qml.QubitUnitary(D, wires=[3,8])
		qml.QubitUnitary(D, wires=[4,9])
		return qml.probs(wires=[0,5]), qml.probs(wires=[1,6]), qml.probs(wires=[2,7]), qml.probs(wires=[3,8]), qml.probs(wires=[4,9]), \
		qml.probs(wires=[1,0,6,5]), qml.probs(wires=[2,0,7,5]), qml.probs(wires=[3,0,8,5]), qml.probs(wires=[4,0,9,5]),\
		qml.probs(wires=[2,1,7,6]), qml.probs(wires=[3,1,8,6]), qml.probs(wires=[4,1,9,6]),\
		qml.probs(wires=[3,2,8,7]), qml.probs(wires=[4,2,9,7]), qml.probs(wires=[4,3,9,8])
	meas_probs3 = qnode(ns, gamma1, gamma2)
	return meas_probs3
def circuit_4(ns, gamma1, gamma2):
	dev = qml.device("default.mixed", wires=range(10), shots=ns)
	@qml.qnode(dev)
	def qnode(Ns, Gamma1, Gamma2):
		qml.Hadamard(wires=0)
		qml.CNOT(wires=[0, 1])
		qml.CNOT(wires=[0, 2])
		qml.Hadamard(wires=3)
		qml.CNOT(wires=[3, 4])
		qml.Hadamard(wires=5)
		qml.CNOT(wires=[5, 6])
		qml.CNOT(wires=[5, 7])
		qml.Hadamard(wires=8)
		qml.CNOT(wires=[8, 9])
		for j in [0,1,3,5,6,8]:
			qml.DepolarizingChannel(Gamma1, wires=j)
		for k in [2,4,7,9]:
			qml.DepolarizingChannel(Gamma2, wires=k)
		qml.QubitUnitary(D, wires=[0,5])
		qml.QubitUnitary(D, wires=[1,6])
		qml.QubitUnitary(D, wires=[2,7])
		qml.QubitUnitary(D, wires=[3,8])
		qml.QubitUnitary(D, wires=[4,9])
		for l in range(10):
		    qml.Hadamard(wires=l)
		return qml.probs(wires=[0,5]), qml.probs(wires=[1,6]), qml.probs(wires=[2,7]), qml.probs(wires=[3,8]), qml.probs(wires=[4,9]), \
		qml.probs(wires=[1,0,6,5]), qml.probs(wires=[2,0,7,5]), qml.probs(wires=[3,0,8,5]), qml.probs(wires=[4,0,9,5]),\
		qml.probs(wires=[2,1,7,6]), qml.probs(wires=[3,1,8,6]), qml.probs(wires=[4,1,9,6]),\
		qml.probs(wires=[3,2,8,7]), qml.probs(wires=[4,2,9,7]), qml.probs(wires=[4,3,9,8])	
	meas_probs4 = qnode(ns, gamma1, gamma2)
	return meas_probs4


def VD_simulation(ntrials):
	for k in range(ntrials):
		Uncertainty_matrices = []
		MutualInfo_matrices = []
		Covariance_matrices = []
		for shot in shot_lists:
			meas_probs1 = circuit_1(shot, 0.2, 0.3)
			meas_probs2 = circuit_2(shot, 0.2, 0.3)
			meas_probs3 = circuit_3(shot, 0.2, 0.3)
			meas_probs4 = circuit_4(shot, 0.2, 0.3)
			
			Marginal_Probs0_basis1 = [Probs_VDproc(meas_probs1[5], meas_probs2[5], meas_probs3[5], meas_probs4[5])[0][0] + Probs_VDproc(meas_probs1[5], meas_probs2[5], meas_probs3[5], meas_probs4[5])[0][2], Probs_VDproc(meas_probs1[5], meas_probs2[5], meas_probs3[5], meas_probs4[5])[0][1] + Probs_VDproc(meas_probs1[5], meas_probs2[5], meas_probs3[5], meas_probs4[5])[0][3]]
			Marginal_Probs0_basis2 = [Probs_VDproc(meas_probs1[5], meas_probs2[5], meas_probs3[5], meas_probs4[5])[1][0] + Probs_VDproc(meas_probs1[5], meas_probs2[5], meas_probs3[5], meas_probs4[5])[1][2], Probs_VDproc(meas_probs1[5], meas_probs2[5], meas_probs3[5], meas_probs4[5])[1][1] + Probs_VDproc(meas_probs1[5], meas_probs2[5], meas_probs3[5], meas_probs4[5])[1][3]]
			Marginal_Probs1_basis1 = [Probs_VDproc(meas_probs1[5], meas_probs2[5], meas_probs3[5], meas_probs4[5])[0][0] + Probs_VDproc(meas_probs1[5], meas_probs2[5], meas_probs3[5], meas_probs4[5])[0][1], Probs_VDproc(meas_probs1[5], meas_probs2[5], meas_probs3[5], meas_probs4[5])[0][2] + Probs_VDproc(meas_probs1[5], meas_probs2[5], meas_probs3[5], meas_probs4[5])[0][3]]
			Marginal_Probs1_basis2 = [Probs_VDproc(meas_probs1[5], meas_probs2[5], meas_probs3[5], meas_probs4[5])[1][0] + Probs_VDproc(meas_probs1[5], meas_probs2[5], meas_probs3[5], meas_probs4[5])[1][1], Probs_VDproc(meas_probs1[5], meas_probs2[5], meas_probs3[5], meas_probs4[5])[1][2] + Probs_VDproc(meas_probs1[5], meas_probs2[5], meas_probs3[5], meas_probs4[5])[1][3]]
			Marginal_Probs2_basis1 = [Probs_VDproc(meas_probs1[12], meas_probs2[12], meas_probs3[12], meas_probs4[12])[0][0] + Probs_VDproc(meas_probs1[12], meas_probs2[12], meas_probs3[12], meas_probs4[12])[0][2], Probs_VDproc(meas_probs1[12], meas_probs2[12], meas_probs3[12], meas_probs4[12])[0][1] + Probs_VDproc(meas_probs1[12], meas_probs2[12], meas_probs3[12], meas_probs4[12])[0][3]]
			Marginal_Probs2_basis2 = [Probs_VDproc(meas_probs1[12], meas_probs2[12], meas_probs3[12], meas_probs4[12])[1][0] + Probs_VDproc(meas_probs1[12], meas_probs2[12], meas_probs3[12], meas_probs4[12])[1][2], Probs_VDproc(meas_probs1[12], meas_probs2[12], meas_probs3[12], meas_probs4[12])[1][1] + Probs_VDproc(meas_probs1[12], meas_probs2[12], meas_probs3[12], meas_probs4[12])[1][3]]
			Marginal_Probs3_basis1 = [Probs_VDproc(meas_probs1[12], meas_probs2[12], meas_probs3[12], meas_probs4[12])[0][0] + Probs_VDproc(meas_probs1[12], meas_probs2[12], meas_probs3[12], meas_probs4[12])[0][1], Probs_VDproc(meas_probs1[12], meas_probs2[12], meas_probs3[12], meas_probs4[12])[0][2] + Probs_VDproc(meas_probs1[12], meas_probs2[12], meas_probs3[12], meas_probs4[12])[0][3]]
			Marginal_Probs3_basis2 = [Probs_VDproc(meas_probs1[12], meas_probs2[12], meas_probs3[12], meas_probs4[12])[1][0] + Probs_VDproc(meas_probs1[12], meas_probs2[12], meas_probs3[12], meas_probs4[12])[1][1], Probs_VDproc(meas_probs1[12], meas_probs2[12], meas_probs3[12], meas_probs4[12])[1][2] + Probs_VDproc(meas_probs1[12], meas_probs2[12], meas_probs3[12], meas_probs4[12])[1][3]]
			Marginal_Probs4_basis1 = [Probs_VDproc(meas_probs1[11], meas_probs2[11], meas_probs3[11], meas_probs4[11])[0][0] + Probs_VDproc(meas_probs1[11], meas_probs2[11], meas_probs3[11], meas_probs4[11])[0][1], Probs_VDproc(meas_probs1[11], meas_probs2[11], meas_probs3[11], meas_probs4[11])[0][2] + Probs_VDproc(meas_probs1[11], meas_probs2[11], meas_probs3[11], meas_probs4[11])[0][3]]
			Marginal_Probs4_basis2 = [Probs_VDproc(meas_probs1[11], meas_probs2[11], meas_probs3[11], meas_probs4[11])[1][0] + Probs_VDproc(meas_probs1[11], meas_probs2[11], meas_probs3[11], meas_probs4[11])[1][1], Probs_VDproc(meas_probs1[11], meas_probs2[11], meas_probs3[11], meas_probs4[11])[1][2] + Probs_VDproc(meas_probs1[11], meas_probs2[11], meas_probs3[11], meas_probs4[11])[1][3]]
			
			H0 = qnet.shannon_entropy(Marginal_Probs0_basis1) + qnet.shannon_entropy(Marginal_Probs0_basis2)
			H1 = qnet.shannon_entropy(Marginal_Probs1_basis1) + qnet.shannon_entropy(Marginal_Probs1_basis2)
			H2 = qnet.shannon_entropy(Marginal_Probs2_basis1) + qnet.shannon_entropy(Marginal_Probs2_basis2)
			H3 = qnet.shannon_entropy(Marginal_Probs3_basis1) + qnet.shannon_entropy(Marginal_Probs3_basis2)
			H4 = qnet.shannon_entropy(Marginal_Probs4_basis1) + qnet.shannon_entropy(Marginal_Probs4_basis2)
			H10 = qnet.shannon_entropy(Probs_VDproc(meas_probs1[5], meas_probs2[5], meas_probs3[5], meas_probs4[5])[0]) + qnet.shannon_entropy(Probs_VDproc(meas_probs1[5], meas_probs2[5], meas_probs3[5], meas_probs4[5])[1]) 
			H20 = qnet.shannon_entropy(Probs_VDproc(meas_probs1[6], meas_probs2[6], meas_probs3[6], meas_probs4[6])[0]) + qnet.shannon_entropy(Probs_VDproc(meas_probs1[6], meas_probs2[6], meas_probs3[6], meas_probs4[6])[1])
			H30 = qnet.shannon_entropy(Probs_VDproc(meas_probs1[7], meas_probs2[7], meas_probs3[7], meas_probs4[7])[0]) + qnet.shannon_entropy(Probs_VDproc(meas_probs1[7], meas_probs2[7], meas_probs3[7], meas_probs4[7])[1]) 
			H40 = qnet.shannon_entropy(Probs_VDproc(meas_probs1[8], meas_probs2[8], meas_probs3[8], meas_probs4[8])[0]) + qnet.shannon_entropy(Probs_VDproc(meas_probs1[8], meas_probs2[8], meas_probs3[8], meas_probs4[8])[1]) 
			H21 = qnet.shannon_entropy(Probs_VDproc(meas_probs1[9], meas_probs2[9], meas_probs3[9], meas_probs4[9])[0]) + qnet.shannon_entropy(Probs_VDproc(meas_probs1[9], meas_probs2[9], meas_probs3[9], meas_probs4[9])[1]) 
			H31 = qnet.shannon_entropy(Probs_VDproc(meas_probs1[10], meas_probs2[10], meas_probs3[10], meas_probs4[10])[0]) + qnet.shannon_entropy(Probs_VDproc(meas_probs1[10], meas_probs2[10], meas_probs3[10], meas_probs4[10])[1]) 
			H41 = qnet.shannon_entropy(Probs_VDproc(meas_probs1[11], meas_probs2[11], meas_probs3[11], meas_probs4[11])[0]) + qnet.shannon_entropy(Probs_VDproc(meas_probs1[11], meas_probs2[11], meas_probs3[11], meas_probs4[11])[1]) 
			H32 = qnet.shannon_entropy(Probs_VDproc(meas_probs1[12], meas_probs2[12], meas_probs3[12], meas_probs4[12])[0]) + qnet.shannon_entropy(Probs_VDproc(meas_probs1[12], meas_probs2[12], meas_probs3[12], meas_probs4[12])[1]) 
			H42 = qnet.shannon_entropy(Probs_VDproc(meas_probs1[13], meas_probs2[13], meas_probs3[13], meas_probs4[13])[0]) + qnet.shannon_entropy(Probs_VDproc(meas_probs1[13], meas_probs2[13], meas_probs3[13], meas_probs4[13])[1]) 
			H43 = qnet.shannon_entropy(Probs_VDproc(meas_probs1[14], meas_probs2[14], meas_probs3[14], meas_probs4[14])[0]) + qnet.shannon_entropy(Probs_VDproc(meas_probs1[14], meas_probs2[14], meas_probs3[14], meas_probs4[14])[1]) 
			
			Uncertainty_matrices.append([[H0/2, H10-H1, H20-H2, H30-H3, H40-H4], [H10-H0, H1/2, H21-H2, H31-H3, H41-H4], [H20-H0, H21-H1, H2/2, H32-H3, H42-H4], [H30-H0, H31-H1, H32-H2, H3/2, H43-H4], [H40-H0, H41-H1, H42-H2, H43-H3, H4/2]])
		for m in range(5):
			for n in range(5):
				record_matrix_uncer[m][n].append(Uncertainty_matrices[0][m][n])
		
		for shot in shot_lists1:
			meas_probs1 = circuit_1(shot, 0.2, 0.3)
			meas_probs3 = circuit_3(shot, 0.2, 0.3)

			I0 = qnet.shannon_entropy(Probs_VDproc_local_alt(meas_probs1[0]))
			I1 = qnet.shannon_entropy(Probs_VDproc_local_alt(meas_probs1[1]))
			I2 = qnet.shannon_entropy(Probs_VDproc_local_alt(meas_probs1[2]))
			I3 = qnet.shannon_entropy(Probs_VDproc_local_alt(meas_probs1[3]))
			I4 = qnet.shannon_entropy(Probs_VDproc_local_alt(meas_probs1[4]))
			I10 = I1 + I0 - qnet.shannon_entropy(Probs_VDproc_alt(meas_probs1[5], meas_probs3[5]))
			I20 = I2 + I0 - qnet.shannon_entropy(Probs_VDproc_alt(meas_probs1[6], meas_probs3[6]))
			I30 = I3 + I0 - qnet.shannon_entropy(Probs_VDproc_alt(meas_probs1[7], meas_probs3[7]))
			I40 = I4 + I0 - qnet.shannon_entropy(Probs_VDproc_alt(meas_probs1[8], meas_probs3[8]))
			I21 = I2 + I1 - qnet.shannon_entropy(Probs_VDproc_alt(meas_probs1[9], meas_probs3[9]))
			I31 = I3 + I1 - qnet.shannon_entropy(Probs_VDproc_alt(meas_probs1[10], meas_probs3[10]))
			I41 = I4 + I1 - qnet.shannon_entropy(Probs_VDproc_alt(meas_probs1[11], meas_probs3[11]))
			I32 = I3 + I2 - qnet.shannon_entropy(Probs_VDproc_alt(meas_probs1[12], meas_probs3[12]))
			I42 = I4 + I2 - qnet.shannon_entropy(Probs_VDproc_alt(meas_probs1[13], meas_probs3[13]))
			I43 = I4 + I3 - qnet.shannon_entropy(Probs_VDproc_alt(meas_probs1[14], meas_probs3[14]))
			
			C0 = 1 - (Probs_VDproc_local_alt(meas_probs1[0])[0] - Probs_VDproc_local_alt(meas_probs1[0])[1])**2
			C1 = 1 - (Probs_VDproc_local_alt(meas_probs1[1])[0] - Probs_VDproc_local_alt(meas_probs1[1])[1])**2
			C2 = 1 - (Probs_VDproc_local_alt(meas_probs1[2])[0] - Probs_VDproc_local_alt(meas_probs1[2])[1])**2
			C3 = 1 - (Probs_VDproc_local_alt(meas_probs1[3])[0] - Probs_VDproc_local_alt(meas_probs1[3])[1])**2
			C4 = 1 - (Probs_VDproc_local_alt(meas_probs1[4])[0] - Probs_VDproc_local_alt(meas_probs1[4])[1])**2
			C10 = Covariance_calculation(Probs_VDproc_alt(meas_probs1[5], meas_probs3[5]))
			C20 = Covariance_calculation(Probs_VDproc_alt(meas_probs1[6], meas_probs3[6]))
			C30 = Covariance_calculation(Probs_VDproc_alt(meas_probs1[7], meas_probs3[7]))
			C40 = Covariance_calculation(Probs_VDproc_alt(meas_probs1[8], meas_probs3[8]))
			C21 = Covariance_calculation(Probs_VDproc_alt(meas_probs1[9], meas_probs3[9]))
			C31 = Covariance_calculation(Probs_VDproc_alt(meas_probs1[10], meas_probs3[10]))
			C41 = Covariance_calculation(Probs_VDproc_alt(meas_probs1[11], meas_probs3[11]))
			C32 = Covariance_calculation(Probs_VDproc_alt(meas_probs1[12], meas_probs3[12]))
			C42 = Covariance_calculation(Probs_VDproc_alt(meas_probs1[13], meas_probs3[13]))
			C43 = Covariance_calculation(Probs_VDproc_alt(meas_probs1[14], meas_probs3[14]))
			
			MutualInfo_matrices.append([[I0, I10, I20, I30, I40], [I10, I1, I21, I31, I41], [I20, I21, I2, I32, I42], [I30, I31, I32, I3, I43], [I40, I41, I42, I43, I4]])
			Covariance_matrices.append([[C0, C10, C20, C30, C40], [C10, C1, C21, C31, C41], [C20, C21, C2, C32, C42], [C30, C31, C32, C3, C43], [C40, C41, C42, C43, C4]])
		for m in range(5):
			for n in range(5):
				record_matrix_mutualinfo[m][n].append(MutualInfo_matrices[0][m][n])
				record_matrix_covar[m][n].append(Covariance_matrices[0][m][n])
		for i in range(len(shot_lists)):
			record_dists_uncer[i].append(math.sqrt(((np.array(Uncertainty_matrices[i]) - np.array(matrix_ideal_uncer)) @ (np.array(Uncertainty_matrices[i]) - np.array(matrix_ideal_uncer)).transpose()).trace()))
			record_dists_mutualinfo[i].append(math.sqrt(((np.array(MutualInfo_matrices[i]) - np.array(matrix_ideal_mutualinfo)) @ (np.array(MutualInfo_matrices[i]) - np.array(matrix_ideal_mutualinfo)).transpose()).trace()))
			record_dists_covar[i].append(math.sqrt(((np.array(Covariance_matrices[i]) - np.array(matrix_ideal_covar)) @ (np.array(Covariance_matrices[i]) - np.array(matrix_ideal_covar)).transpose()).trace()))
	return record_dists_uncer, record_dists_mutualinfo, record_dists_covar, record_matrix_uncer, record_matrix_mutualinfo, record_matrix_covar

def Noisy_simulation(ntrials):
	def n_circuit_1(ns, gamma1, gamma2):
		dev = qml.device("default.mixed", wires=range(5), shots=ns)
		@qml.qnode(dev)
		def qnode(Ns, Gamma1, Gamma2):
			qml.Hadamard(wires=0)
			qml.CNOT(wires=[0, 1])
			qml.CNOT(wires=[0, 2])
			qml.Hadamard(wires=3)
			qml.CNOT(wires=[3, 4])
			for j in [0,1,3]:
				qml.DepolarizingChannel(Gamma1, wires=j)
			for k in [2,4]:
				qml.DepolarizingChannel(Gamma2, wires=k)
			return qml.probs(wires=[0]), qml.probs(wires=[1]), qml.probs(wires=[2]), qml.probs(wires=[3]), qml.probs(wires=[4]),\
			qml.probs(wires=[1,0]), qml.probs(wires=[2,0]), qml.probs(wires=[3,0]), qml.probs(wires=[4,0]),\
			qml.probs(wires=[2,1]), qml.probs(wires=[3,1]), qml.probs(wires=[4,1]),\
			qml.probs(wires=[3,2]), qml.probs(wires=[4,2]), qml.probs(wires=[4,3])
		n_meas_probs1 = qnode(ns, gamma1, gamma2)
		return n_meas_probs1
	def n_circuit_2(ns, gamma1, gamma2):
		dev = qml.device("default.mixed", wires=range(5), shots=ns)
		@qml.qnode(dev)
		def qnode(Ns, Gamma1, Gamma2):
			qml.Hadamard(wires=0)
			qml.CNOT(wires=[0, 1])
			qml.CNOT(wires=[0, 2])
			qml.Hadamard(wires=3)
			qml.CNOT(wires=[3, 4])
			for j in [0,1,3]:
				qml.DepolarizingChannel(Gamma1, wires=j)
			for k in [2,4]:
				qml.DepolarizingChannel(Gamma2, wires=k)
			for l in [0,1,2,3,4]:
				qml.Hadamard(wires=l)
			return qml.probs(wires=[0]), qml.probs(wires=[1]), qml.probs(wires=[2]), qml.probs(wires=[3]), qml.probs(wires=[4]),\
			qml.probs(wires=[1,0]), qml.probs(wires=[2,0]), qml.probs(wires=[3,0]), qml.probs(wires=[4,0]),\
			qml.probs(wires=[2,1]), qml.probs(wires=[3,1]), qml.probs(wires=[4,1]),\
			qml.probs(wires=[3,2]), qml.probs(wires=[4,2]), qml.probs(wires=[4,3])
		n_meas_probs2 = qnode(ns, gamma1, gamma2)
		return n_meas_probs2
	for k in range(ntrials):
		n_Uncertainty_matrices = []
		n_MutualInfo_matrices = []
		n_Covariance_matrices = []
		for shot in shot_lists2:
			n_meas_probs1 = n_circuit_1(shot, 0.2, 0.3)
			n_meas_probs2 = n_circuit_2(shot, 0.2, 0.3)
			H0 = qnet.shannon_entropy(n_meas_probs1[0]) + qnet.shannon_entropy(n_meas_probs2[0])
			H1 = qnet.shannon_entropy(n_meas_probs1[1]) + qnet.shannon_entropy(n_meas_probs2[1])
			H2 = qnet.shannon_entropy(n_meas_probs1[2]) + qnet.shannon_entropy(n_meas_probs2[2])
			H3 = qnet.shannon_entropy(n_meas_probs1[3]) + qnet.shannon_entropy(n_meas_probs2[3])
			H4 = qnet.shannon_entropy(n_meas_probs1[4]) + qnet.shannon_entropy(n_meas_probs2[4])
			H10 = qnet.shannon_entropy(n_meas_probs1[5]) + qnet.shannon_entropy(n_meas_probs2[5])
			H20 = qnet.shannon_entropy(n_meas_probs1[6]) + qnet.shannon_entropy(n_meas_probs2[6])
			H30 = qnet.shannon_entropy(n_meas_probs1[7]) + qnet.shannon_entropy(n_meas_probs2[7])
			H40 = qnet.shannon_entropy(n_meas_probs1[8]) + qnet.shannon_entropy(n_meas_probs2[8])
			H21 = qnet.shannon_entropy(n_meas_probs1[9]) + qnet.shannon_entropy(n_meas_probs2[9])
			H31 = qnet.shannon_entropy(n_meas_probs1[10]) + qnet.shannon_entropy(n_meas_probs2[10])
			H41 = qnet.shannon_entropy(n_meas_probs1[11]) + qnet.shannon_entropy(n_meas_probs2[11])
			H32 = qnet.shannon_entropy(n_meas_probs1[12]) + qnet.shannon_entropy(n_meas_probs2[12])
			H42 = qnet.shannon_entropy(n_meas_probs1[13]) + qnet.shannon_entropy(n_meas_probs2[13])
			H43 = qnet.shannon_entropy(n_meas_probs1[14]) + qnet.shannon_entropy(n_meas_probs2[14])
			n_Uncertainty_matrices.append([[H0/2, H10-H1, H20-H2, H30-H3, H40-H4], [H10-H0, H1/2, H21-H2, H31-H3, H41-H4], [H20-H0, H21-H1, H2/2, H32-H3, H42-H4], [H30-H0, H31-H1, H32-H2, H3/2, H43-H4], [H40-H0, H41-H1, H42-H2, H43-H3, H4/2]])
		for m in range(5):
			for n in range(5):
				n_record_matrix_uncer[m][n].append(n_Uncertainty_matrices[0][m][n])
		
		for shot in shot_lists3:
			n_meas_probs = n_circuit_1(shot, 0.2, 0.3)
			I0 = qnet.shannon_entropy(n_meas_probs[0])
			I1 = qnet.shannon_entropy(n_meas_probs[1])
			I2 = qnet.shannon_entropy(n_meas_probs[2])
			I3 = qnet.shannon_entropy(n_meas_probs[3])
			I4 = qnet.shannon_entropy(n_meas_probs[4])
			I10 = I1 + I0 - qnet.shannon_entropy(n_meas_probs[5])
			I20 = I2 + I0 - qnet.shannon_entropy(n_meas_probs[6])
			I30 = I3 + I0 - qnet.shannon_entropy(n_meas_probs[7])
			I40 = I4 + I0 - qnet.shannon_entropy(n_meas_probs[8])
			I21 = I2 + I1 - qnet.shannon_entropy(n_meas_probs[9])
			I31 = I3 + I1 - qnet.shannon_entropy(n_meas_probs[10])
			I41 = I4 + I1 - qnet.shannon_entropy(n_meas_probs[11])
			I32 = I3 + I2 - qnet.shannon_entropy(n_meas_probs[12])
			I42 = I4 + I2 - qnet.shannon_entropy(n_meas_probs[13])
			I43 = I4 + I3 - qnet.shannon_entropy(n_meas_probs[14])
			
			C0 = 1 - (n_meas_probs[0][0] - n_meas_probs[0][1])**2
			C1 = 1 - (n_meas_probs[1][0] - n_meas_probs[1][1])**2
			C2 = 1 - (n_meas_probs[2][0] - n_meas_probs[2][1])**2
			C3 = 1 - (n_meas_probs[3][0] - n_meas_probs[3][1])**2
			C4 = 1 - (n_meas_probs[4][0] - n_meas_probs[4][1])**2
			C10 = Covariance_calculation(n_meas_probs[5])
			C20 = Covariance_calculation(n_meas_probs[6])
			C30 = Covariance_calculation(n_meas_probs[7])
			C40 = Covariance_calculation(n_meas_probs[8])
			C21 = Covariance_calculation(n_meas_probs[9])
			C31 = Covariance_calculation(n_meas_probs[10])
			C41 = Covariance_calculation(n_meas_probs[11])
			C32 = Covariance_calculation(n_meas_probs[12])
			C42 = Covariance_calculation(n_meas_probs[13])
			C43 = Covariance_calculation(n_meas_probs[14])
			
			n_MutualInfo_matrices.append([[I0, I10, I20, I30, I40], [I10, I1, I21, I31, I41], [I20, I21, I2, I32, I42], [I30, I31, I32, I3, I43], [I40, I41, I42, I43, I4]])
			n_Covariance_matrices.append([[C0, C10, C20, C30, C40], [C10, C1, C21, C31, C41], [C20, C21, C2, C32, C42], [C30, C31, C32, C3, C43], [C40, C41, C42, C43, C4]])
		for m in range(5):
			for n in range(5):
				n_record_matrix_mutualinfo[m][n].append(n_MutualInfo_matrices[0][m][n])
				n_record_matrix_covar[m][n].append(n_Covariance_matrices[0][m][n])
		for i in range(len(shot_lists)):
			n_record_dists_uncer[i].append(math.sqrt(((np.array(n_Uncertainty_matrices[i]) - np.array(matrix_ideal_uncer)) @ (np.array(n_Uncertainty_matrices[i]) - np.array(matrix_ideal_uncer)).transpose()).trace()))
			n_record_dists_mutualinfo[i].append(math.sqrt(((np.array(n_MutualInfo_matrices[i]) - np.array(matrix_ideal_mutualinfo)) @ (np.array(n_MutualInfo_matrices[i]) - np.array(matrix_ideal_mutualinfo)).transpose()).trace()))
			n_record_dists_covar[i].append(math.sqrt(((np.array(n_Covariance_matrices[i]) - np.array(matrix_ideal_covar)) @ (np.array(n_Covariance_matrices[i]) - np.array(matrix_ideal_covar)).transpose()).trace()))
	return n_record_dists_uncer, n_record_dists_mutualinfo, n_record_dists_covar, n_record_matrix_uncer, n_record_matrix_mutualinfo, n_record_matrix_covar


zero_state = np.array([[1, 0], [0, 0]])
one_state = np.array([[0, 0], [0, 1]])

positive_state = np.array([[1/2, 1/2], [1/2, 1/2]])
negative_state = np.array([[1/2, -1/2], [-1/2, 1/2]])

phase_z = np.array([[1, 0], [0, -1j]], dtype=complex)
hadamard = qml.matrix(qml.Hadamard(0))
identity = qml.matrix(qml.Identity(0))

unitaries = [hadamard, hadamard @ phase_z, identity]

# ~ def STT(bitstring):#String to Tensor
    # ~ mapping = {'z': zero_state-one_state, 'x': positive_state-negative_state, 'i': identity}
    # ~ matrices = [mapping[c] for c in bitstring]
    # ~ return reduce(np.kron, matrices)
# ~ projection_bases = [[STT('ziiii'), STT('xiiii')], [STT('iziii'), STT('ixiii')], [STT('iizii'), STT('iixii')], [STT('iiizi'), STT('iiixi')], [STT('iiiiz'), STT('iiiix')],\
# ~ [STT('zziii'), STT('xxiii')], [STT('zizii'), STT('xixii')], [STT('ziizi'), STT('xiixi')], [STT('ziiiz'), STT('xiiix')], [STT('izzii'), STT('ixxii')], \
# ~ [STT('izizi'), STT('ixixi')], [STT('iziiz'), STT('ixiix')], [STT('iizzi'), STT('iixxi')], [STT('iiziz'), STT('iixix')], [STT('iiizz'), STT('iiixx')]]

def reconstruct_snapshots(ndarray1, ndarray2, nq):
	list2 = ndarray2.tolist()
	dictionary = {}
	for i in range(len(ndarray1)):
		key = tuple(list2[i]) 
		if key in dictionary:
			dictionary[key].append(ndarray1[i])
		else:
			dictionary[key] = [ndarray1[i]]
	dictionary_of_snapshots = {}
	for recipe in dictionary.keys():
		rho_snapshot_aver = np.zeros((2 ** nq, 2 ** nq), dtype=complex) 
		for j in range(len(dictionary[recipe])):
			rho_snapshot = [1]
			for i in range(nq):
				U = unitaries[int(list(recipe)[i])]
				state = zero_state if dictionary[recipe][j][i] == 1 else one_state
				local_rho = 3 * (U.conj().T @ state @ U) - identity
				rho_snapshot = np.kron(rho_snapshot, local_rho)
			rho_snapshot_aver += rho_snapshot 
		dictionary_of_snapshots[recipe] = rho_snapshot_aver / len(dictionary[recipe])
	# ~ print(dictionary_of_snapshots)
	return dictionary_of_snapshots

def permutation_operator(nq):
	op = qml.SWAP(wires=[0, nq])
	if nq == 1:
		permutation = qml.matrix(op)
	else:
		for i in range(1, nq):
			op = op @ qml.SWAP(wires=[i, i + nq])
			permutation = qml.matrix(op, wire_order = range(2*nq))
	return permutation
	
def get_remaining_indices(idx1, idx2=None):
    if idx2 is not None:
        return [i for i in range(5) if i not in {idx1, idx2}]
    return [i for i in range(5) if i != idx1]
def aggregate_matrices(data_dict, idx1, idx2=None):
    aggregated_data = defaultdict(list)

    for key, matrix in data_dict.items():
        if idx2 is not None:
            fixed_key = (key[idx1], key[idx2])
        else:
            fixed_key = (key[idx1])
        aggregated_data[fixed_key].append(matrix)

    averaged_data = {}
    remaining_indices = get_remaining_indices(idx1, idx2)
    for new_key, matrices in aggregated_data.items():
        averaged_matrix = np.mean(matrices, axis=0).real.numpy() # tensor to array
        averaged_data[new_key] = partial_trace(averaged_matrix, indices=remaining_indices)
    return list(averaged_data.values())
    
def calculate_expectations(list1, num_qubit):
	permutation = permutation_operator(num_qubit)
	
	mitigated_moment = 0
	for i in range(len(list1)):
		for j in range(len(list1)):
				mitigated_moment += np.trace( permutation @ np.kron(list1[i], list1[j]) ) / (len(list1)**2)
	mitigated_exp0 = 0
	mitigated_exp1 = 0
	if num_qubit == 1:
		for i in range(len(list1)):
			for j in range(len(list1)):
				mitigated_exp0 += np.trace( permutation @ np.kron((zero_state-one_state) @ list1[i], list1[j]) ) / (len(list1)**2)
				mitigated_exp1 += np.trace( permutation @ np.kron((positive_state-negative_state) @ list1[i], list1[j]) ) / (len(list1)**2)
		return [mitigated_exp0.real / mitigated_moment.real, mitigated_exp1.real / mitigated_moment.real]
	else:
		ZZ = np.kron(zero_state-one_state, zero_state-one_state)
		XX = np.kron(positive_state-negative_state, positive_state-negative_state)
		for i in range(len(list1)):
			for j in range(len(list1)):
				mitigated_exp0 += np.trace( permutation @ np.kron(ZZ @ list1[i], list1[j]) ) / (len(list1)**2)
				mitigated_exp1 += np.trace( permutation @ np.kron(XX @ list1[i], list1[j]) ) / (len(list1)**2)
		return [mitigated_exp0.real / mitigated_moment.real, mitigated_exp1.real / mitigated_moment.real]

def calculate_probs_and_matrices(dictionary_of_snapshots):
	exps = []
	exps.append(calculate_expectations(aggregate_matrices(dictionary_of_snapshots, 0, None), 1))
	exps.append(calculate_expectations(aggregate_matrices(dictionary_of_snapshots, 1, None), 1))
	exps.append(calculate_expectations(aggregate_matrices(dictionary_of_snapshots, 2, None), 1))
	exps.append(calculate_expectations(aggregate_matrices(dictionary_of_snapshots, 3, None), 1))
	exps.append(calculate_expectations(aggregate_matrices(dictionary_of_snapshots, 4, None), 1))
	exps.append(calculate_expectations(aggregate_matrices(dictionary_of_snapshots, 1, 0), 2))
	exps.append(calculate_expectations(aggregate_matrices(dictionary_of_snapshots, 2, 0), 2))
	exps.append(calculate_expectations(aggregate_matrices(dictionary_of_snapshots, 3, 0), 2))
	exps.append(calculate_expectations(aggregate_matrices(dictionary_of_snapshots, 4, 0), 2))
	exps.append(calculate_expectations(aggregate_matrices(dictionary_of_snapshots, 2, 1), 2))
	exps.append(calculate_expectations(aggregate_matrices(dictionary_of_snapshots, 3, 1), 2))
	exps.append(calculate_expectations(aggregate_matrices(dictionary_of_snapshots, 4, 1), 2))
	exps.append(calculate_expectations(aggregate_matrices(dictionary_of_snapshots, 3, 2), 2))
	exps.append(calculate_expectations(aggregate_matrices(dictionary_of_snapshots, 4, 2), 2))
	exps.append(calculate_expectations(aggregate_matrices(dictionary_of_snapshots, 4, 3), 2))
	mitigated_probs_z = []
	mitigated_probs_x = []
	for k in range(5):
		mitigated_probs_z.append(project_to_simplex([(1+exps[k][0])/2, (1-exps[k][0])/2]))
		mitigated_probs_x.append(project_to_simplex([(1+exps[k][1])/2, (1-exps[k][1])/2]))
	mitigated_probs_z.append(project_to_simplex([(1 + exps[5][0] + exps[1][0] + exps[0][0]) / 4, (1 - exps[5][0] + exps[1][0] - exps[0][0]) / 4, (1 - exps[5][0] - exps[1][0] + exps[0][0]) / 4, (1 + exps[5][0] - exps[1][0] - exps[0][0]) / 4]))
	mitigated_probs_z.append(project_to_simplex([(1 + exps[6][0] + exps[2][0] + exps[0][0]) / 4, (1 - exps[6][0] + exps[2][0] - exps[0][0]) / 4, (1 - exps[6][0] - exps[2][0] + exps[0][0]) / 4, (1 + exps[6][0] - exps[2][0] - exps[0][0]) / 4]))
	mitigated_probs_z.append(project_to_simplex([(1 + exps[7][0] + exps[3][0] + exps[0][0]) / 4, (1 - exps[7][0] + exps[3][0] - exps[0][0]) / 4, (1 - exps[7][0] - exps[3][0] + exps[0][0]) / 4, (1 + exps[7][0] - exps[3][0] - exps[0][0]) / 4]))
	mitigated_probs_z.append(project_to_simplex([(1 + exps[8][0] + exps[4][0] + exps[0][0]) / 4, (1 - exps[8][0] + exps[4][0] - exps[0][0]) / 4, (1 - exps[8][0] - exps[4][0] + exps[0][0]) / 4, (1 + exps[8][0] - exps[4][0] - exps[0][0]) / 4]))
	mitigated_probs_z.append(project_to_simplex([(1 + exps[9][0] + exps[2][0] + exps[1][0]) / 4, (1 - exps[9][0] + exps[2][0] - exps[1][0]) / 4, (1 - exps[9][0] - exps[2][0] + exps[1][0]) / 4, (1 + exps[9][0] - exps[2][0] - exps[1][0]) / 4]))
	mitigated_probs_z.append(project_to_simplex([(1 + exps[10][0] + exps[3][0] + exps[1][0]) / 4, (1 - exps[10][0] + exps[3][0] - exps[1][0]) / 4, (1 - exps[10][0] - exps[3][0] + exps[1][0]) / 4, (1 + exps[10][0] - exps[3][0] - exps[1][0]) / 4]))
	mitigated_probs_z.append(project_to_simplex([(1 + exps[11][0] + exps[4][0] + exps[1][0]) / 4, (1 - exps[11][0] + exps[4][0] - exps[1][0]) / 4, (1 - exps[11][0] - exps[4][0] + exps[1][0]) / 4, (1 + exps[11][0] - exps[4][0] - exps[1][0]) / 4]))
	mitigated_probs_z.append(project_to_simplex([(1 + exps[12][0] + exps[3][0] + exps[2][0]) / 4, (1 - exps[12][0] + exps[3][0] - exps[2][0]) / 4, (1 - exps[12][0] - exps[3][0] + exps[2][0]) / 4, (1 + exps[12][0] - exps[3][0] - exps[2][0]) / 4]))
	mitigated_probs_z.append(project_to_simplex([(1 + exps[13][0] + exps[4][0] + exps[2][0]) / 4, (1 - exps[13][0] + exps[4][0] - exps[2][0]) / 4, (1 - exps[13][0] - exps[4][0] + exps[2][0]) / 4, (1 + exps[13][0] - exps[4][0] - exps[2][0]) / 4]))
	mitigated_probs_z.append(project_to_simplex([(1 + exps[14][0] + exps[4][0] + exps[3][0]) / 4, (1 - exps[14][0] + exps[4][0] - exps[3][0]) / 4, (1 - exps[14][0] - exps[4][0] + exps[3][0]) / 4, (1 + exps[14][0] - exps[4][0] - exps[3][0]) / 4]))
	mitigated_probs_x.append(project_to_simplex([(1 + exps[5][1] + exps[1][1] + exps[0][1]) / 4, (1 - exps[5][1] + exps[1][1] - exps[0][1]) / 4, (1 - exps[5][1] - exps[1][1] + exps[0][1]) / 4, (1 + exps[5][1] - exps[1][1] - exps[0][1]) / 4]))
	mitigated_probs_x.append(project_to_simplex([(1 + exps[6][1] + exps[2][1] + exps[0][1]) / 4, (1 - exps[6][1] + exps[2][1] - exps[0][1]) / 4, (1 - exps[6][1] - exps[2][1] + exps[0][1]) / 4, (1 + exps[6][1] - exps[2][1] - exps[0][1]) / 4]))
	mitigated_probs_x.append(project_to_simplex([(1 + exps[7][1] + exps[3][1] + exps[0][1]) / 4, (1 - exps[7][1] + exps[3][1] - exps[0][1]) / 4, (1 - exps[7][1] - exps[3][1] + exps[0][1]) / 4, (1 + exps[7][1] - exps[3][1] - exps[0][1]) / 4]))
	mitigated_probs_x.append(project_to_simplex([(1 + exps[8][1] + exps[4][1] + exps[0][1]) / 4, (1 - exps[8][1] + exps[4][1] - exps[0][1]) / 4, (1 - exps[8][1] - exps[4][1] + exps[0][1]) / 4, (1 + exps[8][1] - exps[4][1] - exps[0][1]) / 4]))
	mitigated_probs_x.append(project_to_simplex([(1 + exps[9][1] + exps[2][1] + exps[1][1]) / 4, (1 - exps[9][1] + exps[2][1] - exps[1][1]) / 4, (1 - exps[9][1] - exps[2][1] + exps[1][1]) / 4, (1 + exps[9][1] - exps[2][1] - exps[1][1]) / 4]))
	mitigated_probs_x.append(project_to_simplex([(1 + exps[11][1] + exps[3][1] + exps[1][1]) / 4, (1 - exps[11][1] + exps[3][1] - exps[1][1]) / 4, (1 - exps[11][1] - exps[3][1] + exps[1][1]) / 4, (1 + exps[11][1] - exps[3][1] - exps[1][1]) / 4]))
	mitigated_probs_x.append(project_to_simplex([(1 + exps[11][1] + exps[4][1] + exps[1][1]) / 4, (1 - exps[11][1] + exps[4][1] - exps[1][1]) / 4, (1 - exps[11][1] - exps[4][1] + exps[1][1]) / 4, (1 + exps[11][1] - exps[4][1] - exps[1][1]) / 4]))
	mitigated_probs_x.append(project_to_simplex([(1 + exps[12][1] + exps[3][1] + exps[2][1]) / 4, (1 - exps[12][1] + exps[3][1] - exps[2][1]) / 4, (1 - exps[12][1] - exps[3][1] + exps[2][1]) / 4, (1 + exps[12][1] - exps[3][1] - exps[2][1]) / 4]))
	mitigated_probs_x.append(project_to_simplex([(1 + exps[13][1] + exps[4][1] + exps[2][1]) / 4, (1 - exps[13][1] + exps[4][1] - exps[2][1]) / 4, (1 - exps[13][1] - exps[4][1] + exps[2][1]) / 4, (1 + exps[13][1] - exps[4][1] - exps[2][1]) / 4]))
	mitigated_probs_x.append(project_to_simplex([(1 + exps[14][1] + exps[4][1] + exps[3][1]) / 4, (1 - exps[14][1] + exps[4][1] - exps[3][1]) / 4, (1 - exps[14][1] - exps[4][1] + exps[3][1]) / 4, (1 + exps[14][1] - exps[4][1] - exps[3][1]) / 4]))
	# ~ print("mitigated_probs_z", mitigated_probs_z)
	# ~ print("mitigated_probs_x", mitigated_probs_x)
	H0 = qnet.shannon_entropy(mitigated_probs_z[0]) + qnet.shannon_entropy(mitigated_probs_x[0])
	H1 = qnet.shannon_entropy(mitigated_probs_z[1]) + qnet.shannon_entropy(mitigated_probs_x[1])
	H2 = qnet.shannon_entropy(mitigated_probs_z[2]) + qnet.shannon_entropy(mitigated_probs_x[2])
	H3 = qnet.shannon_entropy(mitigated_probs_z[3]) + qnet.shannon_entropy(mitigated_probs_x[3])
	H4 = qnet.shannon_entropy(mitigated_probs_z[4]) + qnet.shannon_entropy(mitigated_probs_x[4])
	H10 = qnet.shannon_entropy(mitigated_probs_z[5]) + qnet.shannon_entropy(mitigated_probs_x[5])
	H20 = qnet.shannon_entropy(mitigated_probs_z[6]) + qnet.shannon_entropy(mitigated_probs_x[6])
	H30 = qnet.shannon_entropy(mitigated_probs_z[7]) + qnet.shannon_entropy(mitigated_probs_x[7])
	H40 = qnet.shannon_entropy(mitigated_probs_z[8]) + qnet.shannon_entropy(mitigated_probs_x[8])
	H21 = qnet.shannon_entropy(mitigated_probs_z[9]) + qnet.shannon_entropy(mitigated_probs_x[9])
	H31 = qnet.shannon_entropy(mitigated_probs_z[10]) + qnet.shannon_entropy(mitigated_probs_x[10])
	H41 = qnet.shannon_entropy(mitigated_probs_z[11]) + qnet.shannon_entropy(mitigated_probs_x[11])
	H32 = qnet.shannon_entropy(mitigated_probs_z[12]) + qnet.shannon_entropy(mitigated_probs_x[12])
	H42 = qnet.shannon_entropy(mitigated_probs_z[13]) + qnet.shannon_entropy(mitigated_probs_x[13])
	H43 = qnet.shannon_entropy(mitigated_probs_z[14]) + qnet.shannon_entropy(mitigated_probs_x[14])	
	sd_Uncertainty_matrices.append([[H0/2, H10-H1, H20-H2, H30-H3, H40-H4], [H10-H0, H1/2, H21-H2, H31-H3, H41-H4], [H20-H0, H21-H1, H2/2, H32-H3, H42-H4], [H30-H0, H31-H1, H32-H2, H3/2, H43-H4], [H40-H0, H41-H1, H42-H2, H43-H3, H4/2]])
	for m in range(5):
			for n in range(5):
				sd_record_matrix_uncer[m][n].append(sd_Uncertainty_matrices[-1][m][n])
	# ~ print("sd_Uncertainty_matrices", sd_Uncertainty_matrices)
	# ~ print("sd_record_matrix_uncer", sd_record_matrix_uncer)
	I0 = qnet.shannon_entropy(mitigated_probs_z[0])
	I1 = qnet.shannon_entropy(mitigated_probs_z[1])
	I2 = qnet.shannon_entropy(mitigated_probs_z[2])
	I3 = qnet.shannon_entropy(mitigated_probs_z[3])
	I4 = qnet.shannon_entropy(mitigated_probs_z[4])
	I10 = I1 + I0 - qnet.shannon_entropy(mitigated_probs_z[5])
	I20 = I2 + I0 - qnet.shannon_entropy(mitigated_probs_z[6])
	I30 = I3 + I0 - qnet.shannon_entropy(mitigated_probs_z[7])
	I40 = I4 + I0 - qnet.shannon_entropy(mitigated_probs_z[8])
	I21 = I2 + I1 - qnet.shannon_entropy(mitigated_probs_z[9])
	I31 = I3 + I1 - qnet.shannon_entropy(mitigated_probs_z[10])
	I41 = I4 + I1 - qnet.shannon_entropy(mitigated_probs_z[11])
	I32 = I3 + I2 - qnet.shannon_entropy(mitigated_probs_z[12])
	I42 = I4 + I2 - qnet.shannon_entropy(mitigated_probs_z[13])
	I43 = I4 + I3 - qnet.shannon_entropy(mitigated_probs_z[14])
			
	C0 = 1 - (mitigated_probs_z[0][0] - mitigated_probs_z[0][1])**2
	C1 = 1 - (mitigated_probs_z[1][0] - mitigated_probs_z[1][1])**2
	C2 = 1 - (mitigated_probs_z[2][0] - mitigated_probs_z[2][1])**2
	C3 = 1 - (mitigated_probs_z[3][0] - mitigated_probs_z[3][1])**2
	C4 = 1 - (mitigated_probs_z[4][0] - mitigated_probs_z[4][1])**2
	C10 = Covariance_calculation(mitigated_probs_z[5])
	C20 = Covariance_calculation(mitigated_probs_z[6])
	C30 = Covariance_calculation(mitigated_probs_z[7])
	C40 = Covariance_calculation(mitigated_probs_z[8])
	C21 = Covariance_calculation(mitigated_probs_z[9])
	C31 = Covariance_calculation(mitigated_probs_z[10])
	C41 = Covariance_calculation(mitigated_probs_z[11])
	C32 = Covariance_calculation(mitigated_probs_z[12])
	C42 = Covariance_calculation(mitigated_probs_z[13])
	C43 = Covariance_calculation(mitigated_probs_z[14])
	
	sd_MutualInfo_matrices.append([[I0, I10, I20, I30, I40], [I10, I1, I21, I31, I41], [I20, I21, I2, I32, I42], [I30, I31, I32, I3, I43], [I40, I41, I42, I43, I4]])
	sd_Covariance_matrices.append([[C0, C10, C20, C30, C40], [C10, C1, C21, C31, C41], [C20, C21, C2, C32, C42], [C30, C31, C32, C3, C43], [C40, C41, C42, C43, C4]])
	for m in range(5):
		for n in range(5):
			sd_record_matrix_mutualinfo[m][n].append(sd_MutualInfo_matrices[-1][m][n])
			sd_record_matrix_covar[m][n].append(sd_Covariance_matrices[-1][m][n])
	sd_record_dists_uncer.append(math.sqrt(((np.array(sd_Uncertainty_matrices[-1]) - np.array(matrix_ideal_uncer)) @ (np.array(sd_Uncertainty_matrices[-1]) - np.array(matrix_ideal_uncer)).transpose()).trace()))
	sd_record_dists_mutualinfo.append(math.sqrt(((np.array(sd_MutualInfo_matrices[-1]) - np.array(matrix_ideal_mutualinfo)) @ (np.array(sd_MutualInfo_matrices[-1]) - np.array(matrix_ideal_mutualinfo)).transpose()).trace()))
	sd_record_dists_covar.append(math.sqrt(((np.array(sd_Covariance_matrices[-1]) - np.array(matrix_ideal_covar)) @ (np.array(sd_Covariance_matrices[-1]) - np.array(matrix_ideal_covar)).transpose()).trace()))
	# ~ print("sd_record_dists_uncer", sd_record_dists_uncer)

def circuit_shadow(ns):
	dev = qml.device("default.mixed", wires=range(5), shots=ns)
	@qml.qnode(dev)
	def qnode(Ns, Gamma1, Gamma2):
		qml.Hadamard(wires=0)
		qml.CNOT(wires=[0, 1])
		qml.CNOT(wires=[0, 2])
		qml.Hadamard(wires=3)
		qml.CNOT(wires=[3, 4])
		for j in [0,1,3]:
			qml.DepolarizingChannel(Gamma1, wires=j)
		for k in [2,4]:
			qml.DepolarizingChannel(Gamma2, wires=k)
		return classical_shadow(wires=range(5))
	bits, recipes = qnode(ns, 0.2, 0.3)
	dictionary_of_snapshots = reconstruct_snapshots(bits, recipes, 5)
	return dictionary_of_snapshots
def Shadow_simulation(ns, trials):
	for k in range(trials):
		sd_Uncertainty_matrices = []
		sd_MutualInfo_matrices = []
		sd_Covariance_matrices = []
		dictionary_of_snapshots = circuit_shadow(ns)
		calculate_probs_and_matrices(dictionary_of_snapshots)
	return sd_record_dists_uncer, sd_record_dists_mutualinfo, sd_record_dists_covar, sd_record_matrix_uncer, sd_record_matrix_mutualinfo, sd_record_matrix_covar
mitigated_exps = []
	
def Draw_heatmap(trials):
	record_dists_uncer, record_dists_mutualinfo, record_dists_covar, record_matrix_uncer, record_matrix_mutualinfo, record_matrix_covar = VD_simulation(trials)
	print("VD_Dists_uncer:", np.mean(record_dists_uncer), math.sqrt(np.var(record_dists_uncer)))
	print("VD_Dists_mutual:", np.mean(record_dists_mutualinfo), math.sqrt(np.var(record_dists_mutualinfo)))
	print("VD_Dists_Covar:", np.mean(record_dists_covar), math.sqrt(np.var(record_dists_covar)))
	print("record_matrix_uncer:", record_matrix_uncer)

	
	n_record_dists_uncer, n_record_dists_mutualinfo, n_record_dists_covar, n_record_matrix_uncer, n_record_matrix_mutualinfo, n_record_matrix_covar = Noisy_simulation(trials)
	print("n_Dists_uncer:", np.mean(n_record_dists_uncer), math.sqrt(np.var(n_record_dists_uncer)))
	print("n_Dists_mutual:", np.mean(n_record_dists_mutualinfo), math.sqrt(np.var(n_record_dists_mutualinfo)))
	print("n_Dists_Covar:", np.mean(n_record_dists_covar), math.sqrt(np.var(n_record_dists_covar)))
	print("sd_record_matrix_uncer:", n_record_matrix_uncer)
	
	sd_record_dists_uncer, sd_record_dists_mutualinfo, sd_record_dists_covar, sd_record_matrix_uncer, sd_record_matrix_mutualinfo, sd_record_matrix_covar = Shadow_simulation(10000, trials)
	print("sd_Dists_uncer:", np.mean(sd_record_dists_uncer), math.sqrt(np.var(sd_record_dists_uncer)))	
	print("sd_Dists_mutual:", np.mean(sd_record_dists_mutualinfo), math.sqrt(np.var(sd_record_dists_mutualinfo)))
	print("sd_Dists_Covar:", np.mean(sd_record_dists_covar), math.sqrt(np.var(sd_record_dists_covar)))
	print("sd_record_matrix_uncer:", sd_record_matrix_uncer)
	
	label = ["0","1","2","3","4"]
	bar_label = ["Noisy", "VD", "SD"]
	x = np.arange(len(bar_label)) * 0.5
	fig2, ((ax13, ax10, ax11, ax12), (ax23, ax20, ax21, ax22), (ax33, ax30, ax31, ax32)) = plt.subplots(3, 4, figsize = (15,9), gridspec_kw={'width_ratios': [1, 1.3, 1.3, 1.3]})
	image10 = ax10.imshow(np.mean(n_record_matrix_uncer, axis=2), cmap='YlGnBu', vmin=0, vmax=2, interpolation='none')
	# ~ for i in range(5):
	    # ~ for j in range(5):
	        # ~ ax10.text(j, i, format_sci(np.var(n_record_matrix_uncer, axis=2)[i, j], precision=1), ha='center', va='center', color='white', fontsize=6)
	ax10.grid(which="minor", color="k", linestyle='-', linewidth=2)
	ax10.set_xticks(np.arange(5),labels=label)
	ax10.set_yticks(np.arange(5),labels=label)
	ax10.tick_params(top=True, bottom=False,labeltop=True, labelbottom=False)
	ax10.set_title("Noisy")
	image11 = ax11.imshow(np.mean(record_matrix_uncer, axis=2), cmap='YlGnBu', vmin=0, vmax=2, interpolation='none')
	# ~ for i in range(5):
	    # ~ for j in range(5):
	        # ~ ax11.text(j, i, format_sci(np.var(record_matrix_uncer, axis=2)[i, j], precision=1), ha='center', va='center', color='white', fontsize=6)
	ax11.grid(which="minor", color="k", linestyle='-', linewidth=2)
	ax11.set_xticks(np.arange(5),labels=label)
	ax11.set_yticks(np.arange(5),labels=label)
	ax11.tick_params(top=True, bottom=False,labeltop=True, labelbottom=False)
	ax11.set_title("VD")
	image12 = ax12.imshow(np.mean(sd_record_matrix_uncer, axis=2), cmap='YlGnBu', vmin=0, vmax=2, interpolation='none')
	# ~ for i in range(5):
	    # ~ for j in range(5):
	        # ~ ax12.text(j, i, format_sci(np.var(sd_record_matrix_uncer, axis=2)[i, j], precision=1), ha='center', va='center', color='white', fontsize=6)
	ax12.grid(which="minor", color="k", linestyle='-', linewidth=1)
	ax12.set_xticks(np.arange(5),labels=label)
	ax12.set_yticks(np.arange(5),labels=label)
	ax12.tick_params(top=True, bottom=False,labeltop=True, labelbottom=False)
	ax12.set_title("SD")
	cbar1 = fig2.colorbar(image12, ax=ax12)
	cbar1.set_label("Uncertainty", rotation=-90, va="bottom")
	image13 = ax13.bar(x, [np.mean(n_record_dists_uncer), np.mean(record_dists_uncer), np.mean(sd_record_dists_uncer)], yerr = [math.sqrt(np.var(n_record_dists_uncer)), math.sqrt(np.var(record_dists_uncer)), math.sqrt(np.var(sd_record_dists_uncer))], color = 'dodgerblue', width = 0.3, capsize=2)
	ax13.set_ylabel('Distances to noiseless')
	ax13.set_xlim(-0.25, 1.25)
	ax13.set_xticks(x)
	ax13.set_xticklabels(bar_label)

	
	image20 = ax20.imshow(np.mean(n_record_matrix_mutualinfo, axis=2), cmap='viridis', vmin=0, vmax=1, interpolation='none')
	# ~ for i in range(5):
	    # ~ for j in range(5):
	        # ~ ax20.text(j, i, format_sci(np.var(n_record_matrix_mutualinfo, axis=2)[i, j], precision=1), ha='center', va='center', color='white', fontsize=6)
	ax20.grid(which="minor", color="k", linestyle='-', linewidth=1)
	ax20.set_xticks(np.arange(5),labels=label)
	ax20.set_yticks(np.arange(5),labels=label)
	ax20.tick_params(top=True, bottom=False,labeltop=True, labelbottom=False)
	image21 = ax21.imshow(np.mean(record_matrix_mutualinfo, axis=2), cmap='viridis', vmin=0, vmax=1, interpolation='none')
	# ~ for i in range(5):
	    # ~ for j in range(5):
	        # ~ ax21.text(j, i, format_sci(np.var(record_matrix_mutualinfo, axis=2)[i, j], precision=1), ha='center', va='center', color='white', fontsize=6)
	ax21.grid(which="minor", color="k", linestyle='-', linewidth=1)
	ax21.set_xticks(np.arange(5),labels=label)
	ax21.set_yticks(np.arange(5),labels=label)
	ax21.tick_params(top=True, bottom=False,labeltop=True, labelbottom=False)
	image22 = ax22.imshow(np.mean(sd_record_matrix_mutualinfo, axis=2), cmap='viridis', vmin=0, vmax=1, interpolation='none')
	# ~ for i in range(5):
	    # ~ for j in range(5):
	        # ~ ax22.text(j, i, format_sci(np.var(sd_record_matrix_mutualinfo, axis=2)[i, j], precision=1), ha='center', va='center', color='white', fontsize=6)
	ax22.grid(which="minor", color="k", linestyle='-', linewidth=1)
	ax22.set_xticks(np.arange(5),labels=label)
	ax22.set_yticks(np.arange(5),labels=label)
	ax22.tick_params(top=True, bottom=False,labeltop=True, labelbottom=False)
	cbar2 = fig2.colorbar(image22, ax=ax22)
	cbar2.set_label("Characteristic", rotation=-90, va="bottom")
	image23 = ax23.bar(x, [np.mean(n_record_dists_mutualinfo), np.mean(record_dists_mutualinfo), np.mean(sd_record_dists_mutualinfo)], yerr = [math.sqrt(np.var(n_record_dists_mutualinfo)), math.sqrt(np.var(record_dists_mutualinfo)), math.sqrt(np.var(sd_record_dists_mutualinfo))], color = 'orange', width = 0.3, capsize=2)
	ax23.set_ylabel('Distances to noiseless')
	ax23.set_xlim(-0.25, 1.25)
	ax23.set_xticks(x)
	ax23.set_xticklabels(bar_label)
	
	image30 = ax30.imshow(np.mean(n_record_matrix_covar, axis=2), cmap='viridis', vmin=0, vmax=1, interpolation='none')
	# ~ for i in range(5):
	    # ~ for j in range(5):
	        # ~ ax30.text(j, i, format_sci(np.var(n_record_matrix_covar, axis=2)[i, j], precision=1), ha='center', va='center', color='white', fontsize=6)
	ax30.grid(which="minor", color="k", linestyle='-', linewidth=1)	
	ax30.set_xticks(np.arange(5),labels=label)
	ax30.set_yticks(np.arange(5),labels=label)
	ax30.tick_params(top=True, bottom=False,labeltop=True, labelbottom=False)
	image31 = ax31.imshow(np.mean(record_matrix_covar, axis=2), cmap='viridis', vmin=0, vmax=1, interpolation='none')
	# ~ for i in range(5):
	    # ~ for j in range(5):
	        # ~ ax31.text(j, i, format_sci(np.var(record_matrix_covar, axis=2)[i, j], precision=1), ha='center', va='center', color='white', fontsize=6)
	ax31.grid(which="minor", color="k", linestyle='-', linewidth=1)	
	ax31.set_xticks(np.arange(5),labels=label)
	ax31.set_yticks(np.arange(5),labels=label)
	ax31.tick_params(top=True, bottom=False,labeltop=True, labelbottom=False)
	image32 = ax32.imshow(np.mean(sd_record_matrix_covar, axis=2), cmap='viridis', vmin=0, vmax=1, interpolation='none')
	# ~ for i in range(5):
	    # ~ for j in range(5):
	        # ~ ax32.text(j, i, format_sci(np.var(sd_record_matrix_covar, axis=2)[i, j], precision=1), ha='center', va='center', color='white', fontsize=6)
	ax32.grid(which="minor", color="k", linestyle='-', linewidth=1)
	ax32.set_xticks(np.arange(5),labels=label)
	ax32.set_yticks(np.arange(5),labels=label)
	ax32.tick_params(top=True, bottom=False,labeltop=True, labelbottom=False)	
	cbar3 = fig2.colorbar(image32, ax=ax32)
	cbar3.set_label("Covariance", rotation=-90, va="bottom")
	image33 = ax33.bar(x, [np.mean(n_record_dists_covar), np.mean(record_dists_covar), np.mean(sd_record_dists_covar)], yerr = [math.sqrt(np.var(n_record_dists_covar)), math.sqrt(np.var(record_dists_covar)), math.sqrt(np.var(sd_record_dists_covar))], color = 'forestgreen', width = 0.3, capsize=2)
	ax33.set_ylabel('Distances to noiseless')
	ax33.set_xlim(-0.25, 1.25)
	ax33.set_xticks(x)
	ax33.set_xticklabels(bar_label)
Draw_heatmap(10)
	
end_time = time.time()
print("Time cost:", end_time - start_time)

plt.show()
