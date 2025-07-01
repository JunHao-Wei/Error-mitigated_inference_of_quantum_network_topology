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

matrix_ideal_uncer = [[4,2,2], [2,4,2], [2,2,4]]
matrix_ideal_mutualinfo = [[2,1,1], [1,2,1], [1,1,2]]

sd_Uncertainty_matrices = []
sd_MutualInfo_matrices = []

shot_lists2 = [5000]
shot_lists3 = [10000]
n_record_matrix_uncer = [[[] for _ in range (3)] for _ in range(3)]
n_record_matrix_mutualinfo = [[[] for _ in range (3)] for _ in range(3)]

sd_record_matrix_uncer = [[[] for _ in range (3)] for _ in range(3)]
sd_record_matrix_mutualinfo = [[[] for _ in range (3)] for _ in range(3)]

n_record_dists_uncer = [[] for _ in range(len(shot_lists2))]
n_record_dists_mutualinfo = [[] for _ in range(len(shot_lists2))]

sd_record_dists_uncer = []
sd_record_dists_mutualinfo = []

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
    

def Noisy_simulation(ntrials):
	def n_circuit_1(ns, gamma1, gamma2, gamma3):
		dev = qml.device("default.mixed", wires=range(6), shots=ns)
		@qml.qnode(dev)
		def qnode(Ns, Gamma1, Gamma2, Gamma3):
			qml.Hadamard(wires=0)
			qml.Hadamard(wires=2)
			qml.Hadamard(wires=4)
			qml.CNOT(wires=[0, 1])
			qml.CNOT(wires=[2, 3])
			qml.CNOT(wires=[4, 5])
			for j in [0,5]:
				qml.DepolarizingChannel(Gamma1, wires=j)
			for k in [1,2]:
				qml.DepolarizingChannel(Gamma2, wires=k)
			for l in [3,4]:
				qml.DepolarizingChannel(Gamma3, wires=l)
			return qml.probs(wires=[0,5]), qml.probs(wires=[1,2]), qml.probs(wires=[3,4]), qml.probs(wires=[0,5,1,2]), qml.probs(wires=[0,5,3,4]), qml.probs(wires=[1,2,3,4]) 
		n_meas_probs1 = qnode(ns, gamma1, gamma2, gamma3)
		return n_meas_probs1
	def n_circuit_2(ns, gamma1, gamma2, gamma3):
		dev = qml.device("default.mixed", wires=range(6), shots=ns)
		@qml.qnode(dev)
		def qnode(Ns, Gamma1, Gamma2, Gamma3):
			qml.Hadamard(wires=0)
			qml.Hadamard(wires=2)
			qml.Hadamard(wires=4)
			qml.CNOT(wires=[0, 1])
			qml.CNOT(wires=[2, 3])
			qml.CNOT(wires=[4, 5])
			for j in [0,5]:
				qml.DepolarizingChannel(Gamma1, wires=j)
			for k in [1,2]:
				qml.DepolarizingChannel(Gamma2, wires=k)
			for l in [3,4]:
				qml.DepolarizingChannel(Gamma3, wires=l)
			for m in [0,1,2,3,4,5]:
				qml.Hadamard(wires=m)
			return qml.probs(wires=[0,5]), qml.probs(wires=[1,2]), qml.probs(wires=[3,4]), qml.probs(wires=[0,5,1,2]), qml.probs(wires=[0,5,3,4]), qml.probs(wires=[1,2,3,4]) 
		n_meas_probs2 = qnode(ns, gamma1, gamma2, gamma3)
		return n_meas_probs2
	for k in range(ntrials):
		n_Uncertainty_matrices = []
		n_MutualInfo_matrices = []
		for shot in shot_lists2:
			n_meas_probs1 = n_circuit_1(shot, 0.05, 0.10, 0.15)
			n_meas_probs2 = n_circuit_2(shot, 0.05, 0.10, 0.15)
			H05 = qnet.shannon_entropy(n_meas_probs1[0]) + qnet.shannon_entropy(n_meas_probs2[0])
			H12 = qnet.shannon_entropy(n_meas_probs1[1]) + qnet.shannon_entropy(n_meas_probs2[1])
			H34 = qnet.shannon_entropy(n_meas_probs1[2]) + qnet.shannon_entropy(n_meas_probs2[2])
			H0512 = qnet.shannon_entropy(n_meas_probs1[3]) + qnet.shannon_entropy(n_meas_probs2[3])
			H0534 = qnet.shannon_entropy(n_meas_probs1[4]) + qnet.shannon_entropy(n_meas_probs2[4])
			H1234 = qnet.shannon_entropy(n_meas_probs1[5]) + qnet.shannon_entropy(n_meas_probs2[5])

			n_Uncertainty_matrices.append([[H05, H0512-H12, H0534-H34], [H0512-H05, H12, H1234-H34], [H0534-H05, H1234-H12, H34]])
		for m in range(3):
			for n in range(3):
				n_record_matrix_uncer[m][n].append(n_Uncertainty_matrices[0][m][n])
		
		for shot in shot_lists2:
			n_meas_probs = n_circuit_1(shot, 0.05, 0.10, 0.15)
			I05 = qnet.shannon_entropy(n_meas_probs[0])
			I12 = qnet.shannon_entropy(n_meas_probs[1])
			I34 = qnet.shannon_entropy(n_meas_probs[2])
			I0512 = I05 + I12 - qnet.shannon_entropy(n_meas_probs[3])
			I0534 = I05 + I34 - qnet.shannon_entropy(n_meas_probs[4])
			I1234 = I12 + I34 - qnet.shannon_entropy(n_meas_probs[5])
			
			n_MutualInfo_matrices.append([[I05, I0512, I0534], [I0512, I12, I1234], [I0534, I1234, I34]])
		for m in range(3):
			for n in range(3):
				n_record_matrix_mutualinfo[m][n].append(n_MutualInfo_matrices[0][m][n])
		for i in range(len(shot_lists2)):
			n_record_dists_uncer[i].append(math.sqrt(((np.array(n_Uncertainty_matrices[i]) - np.array(matrix_ideal_uncer)) @ (np.array(n_Uncertainty_matrices[i]) - np.array(matrix_ideal_uncer)).transpose()).trace()))
			n_record_dists_mutualinfo[i].append(math.sqrt(((np.array(n_MutualInfo_matrices[i]) - np.array(matrix_ideal_mutualinfo)) @ (np.array(n_MutualInfo_matrices[i]) - np.array(matrix_ideal_mutualinfo)).transpose()).trace()))
	return n_record_dists_uncer, n_record_dists_mutualinfo, n_record_matrix_uncer, n_record_matrix_mutualinfo


zero_state = np.array([[1, 0], [0, 0]])
one_state = np.array([[0, 0], [0, 1]])

positive_state = np.array([[1/2, 1/2], [1/2, 1/2]])
negative_state = np.array([[1/2, -1/2], [-1/2, 1/2]])

phase_z = np.array([[1, 0], [0, -1j]], dtype=complex)
hadamard = qml.matrix(qml.Hadamard(0))
identity = qml.matrix(qml.Identity(0))

unitaries = [hadamard, hadamard @ phase_z, identity]

def STT(bitstring):#String to Tensor
    mapping = {'0': zero_state, '1': one_state, 'p': positive_state, 'n': negative_state, 'i': identity}
    matrices = [mapping[c] for c in bitstring]
    return reduce(np.kron, matrices)
projection_bases_2qubit = [[np.kron(zero_state, zero_state), np.kron(positive_state, positive_state)], [np.kron(zero_state, one_state), np.kron(positive_state, negative_state)],\
 [np.kron(one_state, zero_state), np.kron(negative_state, positive_state)], [np.kron(one_state, one_state), np.kron(negative_state, negative_state)]]
projection_bases_4qubit = [[STT('0000'), STT('pppp')], [STT('0001'), STT('pppn')], [STT('0010'), STT('ppnp')], [STT('0011'), STT('ppnn')], [STT('0100'), STT('pnpp')], [STT('0101'), STT('pnpn')], [STT('0110'), STT('pnnp')],\
 [STT('0111'), STT('pnnn')], [STT('1000'), STT('nppp')], [STT('1001'), STT('nppn')], [STT('1010'), STT('npnp')], [STT('1011'), STT('npnn')], [STT('1100'), STT('nnpp')], [STT('1101'), STT('nnpn')], [STT('1110'), STT('nnnp')], [STT('1111'), STT('nnnn')]]

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
	
def get_remaining_indices(idx1, idx2, idx3=None, idx4=None):
    fixed_indices = {idx1, idx2}
    if idx3 is not None:
        fixed_indices.add(idx3)
    if idx4 is not None:
        fixed_indices.add(idx4)
    return [i for i in range(6) if i not in fixed_indices]
    
def aggregate_matrices(data_dict, idx1, idx2, idx3=None, idx4=None):
    aggregated_data = defaultdict(list)

    for key, matrix in data_dict.items():
        if idx3 is None and idx4 is None:
            fixed_key = (key[idx1], key[idx2])
        else:
            fixed_key = (key[idx1], key[idx2], key[idx3], key[idx4])
        aggregated_data[fixed_key].append(matrix)

    averaged_data = {}
    remaining_indices = get_remaining_indices(idx1, idx2, idx3, idx4)
    for new_key, matrices in aggregated_data.items():
        averaged_matrix = np.mean(matrices, axis=0).real.numpy() # tensor to array
        averaged_data[new_key] = partial_trace(averaged_matrix, indices=remaining_indices)
    return list(averaged_data.values())
    
def calculate_probs(list1, num_qubit):
	permutation = permutation_operator(num_qubit)

	mitigated_probs0 = []
	mitigated_probs1 = []
	
	mitigated_moment = 0
	for i in range(len(list1)):
		for j in range(len(list1)):
				mitigated_moment += np.trace( permutation @ np.kron(list1[i], list1[j]) ) / (len(list1)**2)

	if num_qubit == 2:
		for k in range(len(projection_bases_2qubit)):
			mitigated_prob0 = 0
			mitigated_prob1 = 0
			for i in range(len(list1)):
				for j in range(len(list1)):
					mitigated_prob0 += np.trace( permutation @ np.kron(projection_bases_2qubit[k][0] @ list1[i], list1[j]) ) / (len(list1)**2)
					mitigated_prob1 += np.trace( permutation @ np.kron(projection_bases_2qubit[k][1] @ list1[i], list1[j]) ) / (len(list1)**2)
			mitigated_probs0.append(mitigated_prob0.real / mitigated_moment.real)
			mitigated_probs1.append(mitigated_prob1.real / mitigated_moment.real)
	else:
		for k in range(len(projection_bases_4qubit)):
			mitigated_prob0 = 0
			mitigated_prob1 = 0
			for i in range(len(list1)):
				for j in range(len(list1)):
					mitigated_prob0 += np.trace( permutation @ np.kron(projection_bases_4qubit[k][0] @ list1[i], list1[j]) ) / (len(list1)**2)
					mitigated_prob1 += np.trace( permutation @ np.kron(projection_bases_4qubit[k][1] @ list1[i], list1[j]) ) / (len(list1)**2)
			mitigated_probs0.append(mitigated_prob0.real / mitigated_moment.real)
			mitigated_probs1.append(mitigated_prob1.real / mitigated_moment.real)
	return mitigated_probs0, mitigated_probs1

def calculate_matrices(dictionary_of_snapshots):
	prob05 = calculate_probs(aggregate_matrices(dictionary_of_snapshots, 0, 5, None, None), 2)
	prob12 = calculate_probs(aggregate_matrices(dictionary_of_snapshots, 1, 2, None, None), 2)
	prob34 = calculate_probs(aggregate_matrices(dictionary_of_snapshots, 3, 4, None, None), 2)
	prob0512 = calculate_probs(aggregate_matrices(dictionary_of_snapshots, 0, 5, 1, 2), 4)
	prob0534 = calculate_probs(aggregate_matrices(dictionary_of_snapshots, 0, 5, 3, 4), 4)
	prob1234 = calculate_probs(aggregate_matrices(dictionary_of_snapshots, 1, 2, 3, 4), 4)

	H05 = qnet.shannon_entropy(prob05[0]) + qnet.shannon_entropy(prob05[1])
	H12 = qnet.shannon_entropy(prob12[0]) + qnet.shannon_entropy(prob12[1])
	H34 = qnet.shannon_entropy(prob34[0]) + qnet.shannon_entropy(prob34[1])
	H0512 = qnet.shannon_entropy(prob0512[0]) + qnet.shannon_entropy(prob0512[1])
	H0534 = qnet.shannon_entropy(prob0534[0]) + qnet.shannon_entropy(prob0534[1])
	H1234 = qnet.shannon_entropy(prob1234[0]) + qnet.shannon_entropy(prob1234[1])
	
	sd_Uncertainty_matrices.append([[H05, H0512-H12, H0534-H34], [H0512-H05, H12, H1234-H34], [H0534-H05, H1234-H12, H34]])
	for m in range(3):
			for n in range(3):
				sd_record_matrix_uncer[m][n].append(sd_Uncertainty_matrices[-1][m][n])
	# ~ print("sd_Uncertainty_matrices", sd_Uncertainty_matrices)
	# ~ print("sd_record_matrix_uncer", sd_record_matrix_uncer)
	I05 = qnet.shannon_entropy(prob05[0])
	I12 = qnet.shannon_entropy(prob12[0])
	I34 = qnet.shannon_entropy(prob34[0])
	I0512 = I05 + I12 - qnet.shannon_entropy(prob0512[0])
	I0534 = I05 + I34 - qnet.shannon_entropy(prob0534[0])
	I1234 = I12 + I34 - qnet.shannon_entropy(prob1234[0])

	
	sd_MutualInfo_matrices.append([[I05, I0512, I0534], [I0512, I12, I1234], [I0534, I1234, I34]])
	for m in range(3):
		for n in range(3):
			sd_record_matrix_mutualinfo[m][n].append(sd_MutualInfo_matrices[-1][m][n])
	sd_record_dists_uncer.append(math.sqrt(((np.array(sd_Uncertainty_matrices[-1]) - np.array(matrix_ideal_uncer)) @ (np.array(sd_Uncertainty_matrices[-1]) - np.array(matrix_ideal_uncer)).transpose()).trace()))
	sd_record_dists_mutualinfo.append(math.sqrt(((np.array(sd_MutualInfo_matrices[-1]) - np.array(matrix_ideal_mutualinfo)) @ (np.array(sd_MutualInfo_matrices[-1]) - np.array(matrix_ideal_mutualinfo)).transpose()).trace()))
	# ~ print("sd_record_dists_uncer", sd_record_dists_uncer)

def circuit_shadow(ns):
	dev = qml.device("default.mixed", wires=range(6), shots=ns)
	@qml.qnode(dev)
	def qnode(Ns, Gamma1, Gamma2, Gamma3):
		qml.Hadamard(wires=0)
		qml.Hadamard(wires=2)
		qml.Hadamard(wires=4)
		qml.CNOT(wires=[0, 1])
		qml.CNOT(wires=[2, 3])
		qml.CNOT(wires=[4, 5])
		for j in [0,5]:
			qml.DepolarizingChannel(Gamma1, wires=j)
		for k in [1,2]:
			qml.DepolarizingChannel(Gamma2, wires=k)
		for l in [3,4]:
			qml.DepolarizingChannel(Gamma3, wires=l)
		return classical_shadow(wires=range(6))
	bits, recipes = qnode(ns, 0.05, 0.10, 0.15)
	dictionary_of_snapshots = reconstruct_snapshots(bits, recipes, 6)
	return dictionary_of_snapshots
def Shadow_simulation(ns, trials):
	for k in range(trials):
		sd_Uncertainty_matrices = []
		sd_MutualInfo_matrices = []
		dictionary_of_snapshots = circuit_shadow(ns)
		calculate_matrices(dictionary_of_snapshots)
	return sd_record_dists_uncer, sd_record_dists_mutualinfo, sd_record_matrix_uncer, sd_record_matrix_mutualinfo
mitigated_exps = []
	
def Draw_heatmap(trials):
	n_record_dists_uncer, n_record_dists_mutualinfo, n_record_matrix_uncer, n_record_matrix_mutualinfo = Noisy_simulation(trials)
	print("n_Dists_uncer:", np.mean(n_record_dists_uncer), math.sqrt(np.var(n_record_dists_uncer)))
	print("n_Dists_mutual:", np.mean(n_record_dists_mutualinfo), math.sqrt(np.var(n_record_dists_mutualinfo)))
	# ~ print("n_record_matrix_uncer:", n_record_matrix_uncer)
	# ~ print("n_record_matrix_mutual:", n_record_matrix_mutualinfo)
	print("averaged_n_matrix_uncer:", np.mean(n_record_matrix_uncer, axis=2))
	print("averaged_n_matrix_mutual:", np.mean(n_record_matrix_mutualinfo, axis=2))
	sd_record_dists_uncer, sd_record_dists_mutualinfo, sd_record_matrix_uncer, sd_record_matrix_mutualinfo = Shadow_simulation(10000, trials)
	print("sd_Dists_uncer:", np.mean(sd_record_dists_uncer), math.sqrt(np.var(sd_record_dists_uncer)))	
	print("sd_Dists_mutual:", np.mean(sd_record_dists_mutualinfo), math.sqrt(np.var(sd_record_dists_mutualinfo)))
	# ~ print("sd_record_matrix_uncer:", sd_record_matrix_uncer)
	# ~ print("sd_record_matrix_mutual:", sd_record_matrix_mutualinfo)
	print("averaged_sd_matrix_uncer:", np.mean(sd_record_matrix_uncer, axis=2))
	print("averaged_sd_matrix_mutual:", np.mean(sd_record_matrix_mutualinfo, axis=2))
	
	label = ["0","1","2"]
	bar_label = ["Noisy", "SD"]
	fig2, ((ax13, ax10, ax11), (ax23, ax20, ax21)) = plt.subplots(2, 3, figsize = (12,6), gridspec_kw={'width_ratios': [1, 1.25, 1.25]})
	image10 = ax10.imshow(np.mean(n_record_matrix_uncer, axis=2), cmap='YlGnBu', vmin=0, vmax=4, interpolation='none')
	for i in range(3):
	    for j in range(3):
	        ax10.text(j, i, f"{np.mean(n_record_matrix_uncer, axis=2)[i, j]:.2f}", ha='center', va='center', color='white', fontsize=10)
	ax10.grid(which="minor", color="k", linestyle='-', linewidth=2)
	ax10.set_xticks(np.arange(3),labels=label)
	ax10.set_yticks(np.arange(3),labels=label)
	ax10.tick_params(top=True, bottom=False,labeltop=True, labelbottom=False)
	ax10.set_title("Noisy")
	image11 = ax11.imshow(np.mean(sd_record_matrix_uncer, axis=2), cmap='YlGnBu', vmin=0, vmax=4, interpolation='none')
	for i in range(3):
	    for j in range(3):
	        ax11.text(j, i, f"{np.mean(sd_record_matrix_uncer, axis=2)[i, j]:.2f}", ha='center', va='center', color='white', fontsize=10)
	ax11.grid(which="minor", color="k", linestyle='-', linewidth=2)
	ax11.set_xticks(np.arange(3),labels=label)
	ax11.set_yticks(np.arange(3),labels=label)
	ax11.tick_params(top=True, bottom=False,labeltop=True, labelbottom=False)
	ax11.set_title("SD")
	cbar1 = fig2.colorbar(image11, ax=ax11)
	cbar1.set_label("Uncertainty", rotation=-90, va="bottom")
	image13 = ax13.bar(bar_label, [np.mean(n_record_dists_uncer), np.mean(sd_record_dists_uncer)], yerr = [math.sqrt(np.var(n_record_dists_uncer)), math.sqrt(np.var(sd_record_dists_uncer))], color = 'dodgerblue', width = 0.6, capsize=2)
	ax13.set_ylabel('Distances to noiseless')
	
	image20 = ax20.imshow(np.mean(n_record_matrix_mutualinfo, axis=2), cmap='viridis', vmin=0, vmax=2, interpolation='none')
	for i in range(3):
	    for j in range(3):
	        ax20.text(j, i, f"{np.mean(n_record_matrix_mutualinfo, axis=2)[i, j]:.2f}", ha='center', va='center', color='white', fontsize=10)
	ax20.grid(which="minor", color="k", linestyle='-', linewidth=1)
	ax20.set_xticks(np.arange(3),labels=label)
	ax20.set_yticks(np.arange(3),labels=label)
	ax20.tick_params(top=True, bottom=False,labeltop=True, labelbottom=False)
	image21 = ax21.imshow(np.mean(sd_record_matrix_mutualinfo, axis=2), cmap='viridis', vmin=0, vmax=2, interpolation='none')
	for i in range(3):
	    for j in range(3):
	        ax21.text(j, i, f"{np.mean(sd_record_matrix_mutualinfo, axis=2)[i, j]:.2f}", ha='center', va='center', color='white', fontsize=10)
	ax21.grid(which="minor", color="k", linestyle='-', linewidth=1)
	ax21.set_xticks(np.arange(3),labels=label)
	ax21.set_yticks(np.arange(3),labels=label)
	ax21.tick_params(top=True, bottom=False,labeltop=True, labelbottom=False)
	cbar2 = fig2.colorbar(image21, ax=ax21)
	cbar2.set_label("Characteristic", rotation=-90, va="bottom")
	image23 = ax23.bar(bar_label, [np.mean(n_record_dists_mutualinfo), np.mean(sd_record_dists_mutualinfo)], yerr = [math.sqrt(np.var(n_record_dists_mutualinfo)), math.sqrt(np.var(sd_record_dists_mutualinfo))], color = 'orange', width = 0.6, capsize=2)
	ax23.set_ylabel('Distances to noiseless')
	
	
Draw_heatmap(10)
	
end_time = time.time()
print("Time cost:", end_time - start_time)

plt.show()
