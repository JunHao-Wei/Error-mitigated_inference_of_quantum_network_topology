import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import qnetvo as qnet
from scipy.stats import sem
import math
import time

start_time = time.time()

start_time = time.time()

prep_nodes = [
    qnet.PrepareNode(1, [0,1,2], qnet.ghz_state, 0),
    qnet.PrepareNode(1, [3,4], qnet.ghz_state, 0),
]
def processing_ansatz(settings, wires):
	qml.Hadamard(wires=wires[5])
	qml.Hadamard(wires=wires[6])
	qml.Hadamard(wires=wires[7])
	qml.Hadamard(wires=wires[8])
	qml.Hadamard(wires=wires[9])
	qml.ControlledQubitUnitary(qml.Hadamard(wires=wires[0]), control_wires=wires[5])
	qml.ControlledQubitUnitary(qml.Hadamard(wires=wires[1]), control_wires=wires[6])
	qml.ControlledQubitUnitary(qml.Hadamard(wires=wires[2]), control_wires=wires[7])
	qml.ControlledQubitUnitary(qml.Hadamard(wires=wires[3]), control_wires=wires[8])
	qml.ControlledQubitUnitary(qml.Hadamard(wires=wires[4]), control_wires=wires[9])
proc_nodes = [
	qnet.ProcessingNode(
		wires=[0,1,2,3,4,5,6,7,8,9],
		ansatz_fn=processing_ansatz
	)
]
meas_nodes = [
    qnet.MeasureNode(1, 2, [0], ansatz_fn=qml.ArbitraryUnitary, num_settings=3),
    qnet.MeasureNode(1, 2, [1], ansatz_fn=qml.ArbitraryUnitary, num_settings=3),
    qnet.MeasureNode(1, 2, [2], ansatz_fn=qml.ArbitraryUnitary, num_settings=3),
    qnet.MeasureNode(1, 2, [3], ansatz_fn=qml.ArbitraryUnitary, num_settings=3),
    qnet.MeasureNode(1, 2, [4], ansatz_fn=qml.ArbitraryUnitary, num_settings=3),
]
	
def SHANNONENTROPY(m, ansatz):
	@qml.qnode(qml.device(**ansatz.dev_kwargs))
	def circ(settings):
		ansatz.fn(settings)
		return qml.probs([m])
	def ShannonEntropy(*settings):
		Probs = circ(settings)
		Shannonentropy = qnet.shannon_entropy(Probs)
		return Shannonentropy
	return ShannonEntropy
	
def UNCERTAINTY_local(m, ansatz):
	@qml.qnode(qml.device(**ansatz.dev_kwargs))
	def circ(settings):
		ansatz.fn(settings)
		return qml.probs([m, m+5])
	def Uncertainty_local(*settings):
		Probs = circ(settings)
		Probs_basis1 = [Probs[0]/(Probs[0]+Probs[2]), Probs[2]/(Probs[0]+Probs[2])]
		Probs_basis2 = [Probs[1]/(Probs[1]+Probs[3]), Probs[3]/(Probs[1]+Probs[3])]
		uncertainty_local = (qnet.shannon_entropy(Probs_basis1) + qnet.shannon_entropy(Probs_basis2))/2
		return uncertainty_local
	return Uncertainty_local

def UNCERTAINTY_cond_on_Q2(m, n, ansatz):
	@qml.qnode(qml.device(**ansatz.dev_kwargs))
	def circ(settings):
		ansatz.fn(settings)
		return qml.probs([m, n, m+5, n+5])
	def Uncertainty_cond_on_Q2(*settings):
		Probs = circ(settings)
		Marginal_Probs_Q2_basis1 = [(Probs[0]+Probs[8])/(Probs[0]+Probs[4]+Probs[8]+Probs[12]), (Probs[4]+Probs[12])/(Probs[0]+Probs[4]+Probs[8]+Probs[12])]
		Marginal_Probs_Q2_basis2 = [(Probs[3]+Probs[11])/(Probs[3]+Probs[7]+Probs[11]+Probs[15]), (Probs[7]+Probs[15])/(Probs[3]+Probs[7]+Probs[11]+Probs[15])]
		Joint_Probs_basis1 = [Probs[0]/(Probs[0] + Probs[4] + Probs[8] + Probs[12]), Probs[4]/(Probs[0] + Probs[4] + Probs[8] + Probs[12]), Probs[8]/(Probs[0] + Probs[4] + Probs[8] + Probs[12]), Probs[12]/(Probs[0] + Probs[4] + Probs[8] + Probs[12])]
		Joint_Probs_basis2 = [Probs[3]/(Probs[3] + Probs[7] + Probs[11] + Probs[15]), Probs[7]/(Probs[3] + Probs[7] + Probs[11] + Probs[15]), Probs[11]/(Probs[3] + Probs[7] + Probs[11] + Probs[15]), Probs[15]/(Probs[3] + Probs[7] + Probs[11] + Probs[15])]
		uncertainty_cond_on_Q2 = qnet.shannon_entropy(Joint_Probs_basis1) + qnet.shannon_entropy(Joint_Probs_basis2) - qnet.shannon_entropy(Marginal_Probs_Q2_basis1) - qnet.shannon_entropy(Marginal_Probs_Q2_basis2)
		return uncertainty_cond_on_Q2
	return Uncertainty_cond_on_Q2
	
def Mutual_information(m, n, ansatz):
	@qml.qnode(qml.device(**ansatz.dev_kwargs))
	def circ2(settings):
		ansatz.fn(settings)
		return qml.probs([m, n])
	def Mutual_info(*settings):
		Probs = circ2(settings)
		Marginal_Probs_Q1 = [Probs[0]+Probs[1], Probs[2]+Probs[3]]
		Marginal_Probs_Q2 = [Probs[0]+Probs[2], Probs[1]+Probs[3]]
		mutual_info = qnet.shannon_entropy(Probs) - qnet.shannon_entropy(Marginal_Probs_Q1) - qnet.shannon_entropy(Marginal_Probs_Q2)#Note that the mutual information is negative here as it is to be maximized
		return mutual_info
	return Mutual_info

def Covariance(m, n, ansatz):
	@qml.qnode(qml.device(**ansatz.dev_kwargs))
	def circ2(settings):
		ansatz.fn(settings)
		return qml.probs([m, n])
	def Covar(*settings):
		Probs = circ2(settings)
		Marginal_Probs_Q1 = [Probs[0]+Probs[1], Probs[2]+Probs[3]]
		Marginal_Probs_Q2 = [Probs[0]+Probs[2], Probs[1]+Probs[3]]
		covar = (Marginal_Probs_Q1[0] - Marginal_Probs_Q1[1])*(Marginal_Probs_Q2[0] - Marginal_Probs_Q2[1]) - (Probs[0] - Probs[1] - Probs[2] + Probs[3])
		return covar
	return Covar
	
def Variance(m, ansatz):
	@qml.qnode(qml.device(**ansatz.dev_kwargs))
	def circ2(settings):
		ansatz.fn(settings)
		return qml.probs([m])
	def Var(*settings):
		Probs = circ2(settings)
		var = (Probs[0] - Probs[1])*(Probs[0] - Probs[1]) - (Probs[0] + Probs[1])
		return var
	return Var


matrix_ideal_uncer = [[1,1,1,2,2], [1,1,1,2,2], [1,1,1,2,2], [2,2,2,1,0], [2,2,2,0,1]]
matrix_ideal_mutualinfo = [[1,1,1,0,0], [1,1,1,0,0], [1,1,1,0,0], [0,0,0,1,1], [0,0,0,1,1]]
matrix_ideal_covar = [[1,1,1,0,0], [1,1,1,0,0], [1,1,1,0,0], [0,0,0,1,1], [0,0,0,1,1]]

scores_matrix_uncer_100 = [[0 for _ in range (5)] for _ in range(5)]
scores_matrix_uncer_1000 = [[0 for _ in range (5)] for _ in range(5)]
scores_matrix_uncer_10000 = [[0 for _ in range (5)] for _ in range(5)]
opt_scores_matrix_uncer_100 = [[0 for _ in range (5)] for _ in range(5)]
opt_scores_matrix_uncer_1000 = [[0 for _ in range (5)] for _ in range(5)]
opt_scores_matrix_uncer_10000 = [[0 for _ in range (5)] for _ in range(5)]
scores_matrix_uncer_300 = [[0 for _ in range (5)] for _ in range(5)]
scores_matrix_uncer_3000 = [[0 for _ in range (5)] for _ in range(5)]
opt_scores_matrix_uncer_300 = [[0 for _ in range (5)] for _ in range(5)]
opt_scores_matrix_uncer_3000 = [[0 for _ in range (5)] for _ in range(5)]

list_of_error_uncer_100 = []
list_of_error_uncer_1000 = []
list_of_error_uncer_10000 = []
list2_of_error_uncer_100 = []
list2_of_error_uncer_1000 = []
list2_of_error_uncer_10000 = []
list_of_error_uncer_300 = []
list_of_error_uncer_3000 = []
list2_of_error_uncer_300 = []
list2_of_error_uncer_3000 = []

scores_matrix_mutualinfo_100 = [[0 for _ in range (5)] for _ in range(5)]
scores_matrix_mutualinfo_1000 = [[0 for _ in range (5)] for _ in range(5)]
scores_matrix_mutualinfo_10000 = [[0 for _ in range (5)] for _ in range(5)]
opt_scores_matrix_mutualinfo_100 = [[0 for _ in range (5)] for _ in range(5)]
opt_scores_matrix_mutualinfo_1000 = [[0 for _ in range (5)] for _ in range(5)]
opt_scores_matrix_mutualinfo_10000 = [[0 for _ in range (5)] for _ in range(5)]
scores_matrix_mutualinfo_300 = [[0 for _ in range (5)] for _ in range(5)]
scores_matrix_mutualinfo_3000 = [[0 for _ in range (5)] for _ in range(5)]
opt_scores_matrix_mutualinfo_300 = [[0 for _ in range (5)] for _ in range(5)]
opt_scores_matrix_mutualinfo_3000 = [[0 for _ in range (5)] for _ in range(5)]

list_of_error_mutualinfo_100 = []
list_of_error_mutualinfo_1000 = []
list_of_error_mutualinfo_10000 = []
list2_of_error_mutualinfo_100 = []
list2_of_error_mutualinfo_1000 = []
list2_of_error_mutualinfo_10000 = []
list_of_error_mutualinfo_300 = []
list_of_error_mutualinfo_3000 = []
list2_of_error_mutualinfo_300 = []
list2_of_error_mutualinfo_3000 = []

scores_matrix_covar_100 = [[0 for _ in range (5)] for _ in range(5)]
scores_matrix_covar_1000 = [[0 for _ in range (5)] for _ in range(5)]
scores_matrix_covar_10000 = [[0 for _ in range (5)] for _ in range(5)]
opt_scores_matrix_covar_100 = [[0 for _ in range (5)] for _ in range(5)]
opt_scores_matrix_covar_1000 = [[0 for _ in range (5)] for _ in range(5)]
opt_scores_matrix_covar_10000 = [[0 for _ in range (5)] for _ in range(5)]
scores_matrix_covar_300 = [[0 for _ in range (5)] for _ in range(5)]
scores_matrix_covar_3000 = [[0 for _ in range (5)] for _ in range(5)]
opt_scores_matrix_covar_300 = [[0 for _ in range (5)] for _ in range(5)]
opt_scores_matrix_covar_3000 = [[0 for _ in range (5)] for _ in range(5)]

list_of_error_covar_100 = []
list_of_error_covar_1000 = []
list_of_error_covar_10000 = []
list2_of_error_covar_100 = []
list2_of_error_covar_1000 = []
list2_of_error_covar_10000 = []
list_of_error_covar_300 = []
list_of_error_covar_3000 = []
list2_of_error_covar_300 = []
list2_of_error_covar_3000 = []

def Optimization_wrt_steps(trials, steps_uncer, steps, stepsizeUncerLocal, stepsizese, stepsizeUncer, stepsizeMutualInfo, stepsizeCovar):
#scores_sum stores the scores of every steps, while opt_scores_sum stores the optimal scores of each trial
	network_ansatz_100 = qnet.NetworkAnsatz(
		prep_nodes,
		proc_nodes,
		meas_nodes,
		dev_kwargs={
			"name": "default.qubit",
			"shots": 100,
		}
	)
	network_ansatz_300 = qnet.NetworkAnsatz(
		prep_nodes,
		proc_nodes,
		meas_nodes,
		dev_kwargs={
			"name": "default.qubit",
			"shots": 300,
		}
	)
	network_ansatz_1000 = qnet.NetworkAnsatz(
		prep_nodes,
		proc_nodes,
		meas_nodes,
		dev_kwargs={
			"name": "default.qubit",
			"shots": 1000,
		}
	)
	network_ansatz_10000 = qnet.NetworkAnsatz(
		prep_nodes,
		proc_nodes,
		meas_nodes,
		dev_kwargs={
			"name": "default.qubit",
			"shots": 10000,
		}
	)
	network_ansatz_3000 = qnet.NetworkAnsatz(
		prep_nodes,
		proc_nodes,
		meas_nodes,
		dev_kwargs={
			"name": "default.qubit",
			"shots": 3000,
		}
	)
	
	for m in range(5):
		for n in range(5):
			if m == n:
				cost_uncer_local_100 = UNCERTAINTY_local(m, network_ansatz_100)
				scores_sum_uncer_local_100 = 0
				opt_scores_sum_uncer_local_100 = []
				cost_uncer_local_300 = UNCERTAINTY_local(m, network_ansatz_300)
				scores_sum_uncer_local_300 = 0
				opt_scores_sum_uncer_local_300 = []
				cost_uncer_local_1000 = UNCERTAINTY_local(m, network_ansatz_1000)
				scores_sum_uncer_local_1000 = 0
				opt_scores_sum_uncer_local_1000 = []
				cost_uncer_local_3000 = UNCERTAINTY_local(m, network_ansatz_3000)
				scores_sum_uncer_local_3000 = 0
				opt_scores_sum_uncer_local_3000 = []
				cost_uncer_local_10000 = UNCERTAINTY_local(m, network_ansatz_10000)
				scores_sum_uncer_local_10000 = 0
				opt_scores_sum_uncer_local_10000 = []
				
				np.random.seed(73)
				for i in range(trials):
					settings = network_ansatz_100.rand_network_settings()
					opt_dict_uncer_local_100 = qnet.gradient_descent(
						cost_uncer_local_100,
						settings,
						step_size=stepsizeUncerLocal,
						num_steps=steps_uncer,
						sample_width=1,
						verbose=False,
					)
					scores_sum_uncer_local_100 -= np.array(opt_dict_uncer_local_100["scores"])
					opt_scores_sum_uncer_local_100.append(-np.array(opt_dict_uncer_local_100["opt_score"]))
					
					settings = network_ansatz_300.rand_network_settings()
					opt_dict_uncer_local_300 = qnet.gradient_descent(
						cost_uncer_local_300,
						settings,
						step_size=stepsizeUncerLocal,
						num_steps=steps_uncer,
						sample_width=1,
						verbose=False,
					)
					scores_sum_uncer_local_300 -= np.array(opt_dict_uncer_local_300["scores"])
					opt_scores_sum_uncer_local_300.append(-np.array(opt_dict_uncer_local_300["opt_score"]))
					
					settings = network_ansatz_1000.rand_network_settings()
					opt_dict_uncer_local_1000 = qnet.gradient_descent(
						cost_uncer_local_1000,
						settings,
						step_size=stepsizeUncerLocal,
						num_steps=steps_uncer,
						sample_width=1,
						verbose=False,
					)
					scores_sum_uncer_local_1000 -= np.array(opt_dict_uncer_local_1000["scores"])
					opt_scores_sum_uncer_local_1000.append(-np.array(opt_dict_uncer_local_1000["opt_score"]))
					
					settings = network_ansatz_3000.rand_network_settings()
					opt_dict_uncer_local_3000 = qnet.gradient_descent(
						cost_uncer_local_3000,
						settings,
						step_size=stepsizeUncerLocal,
						num_steps=steps_uncer,
						sample_width=1,
						verbose=False,
					)
					scores_sum_uncer_local_3000 -= np.array(opt_dict_uncer_local_3000["scores"])
					opt_scores_sum_uncer_local_3000.append(-np.array(opt_dict_uncer_local_3000["opt_score"]))
					
					settings = network_ansatz_10000.rand_network_settings()
					opt_dict_uncer_local_10000 = qnet.gradient_descent(
						cost_uncer_local_10000,
						settings,
						step_size=stepsizeUncerLocal,
						num_steps=steps_uncer,
						sample_width=1,
						verbose=False,
					)
					scores_sum_uncer_local_10000 -= np.array(opt_dict_uncer_local_10000["scores"])
					opt_scores_sum_uncer_local_10000.append(-np.array(opt_dict_uncer_local_10000["opt_score"]))
							
				opt_scores_matrix_uncer_100[m][m] = opt_scores_sum_uncer_local_100
				scores_matrix_uncer_100[m][m] = np.multiply(scores_sum_uncer_local_100, 1/trials)	
				opt_scores_matrix_uncer_300[m][m] = opt_scores_sum_uncer_local_300
				scores_matrix_uncer_300[m][m] = np.multiply(scores_sum_uncer_local_300, 1/trials)	
				opt_scores_matrix_uncer_1000[m][m] = opt_scores_sum_uncer_local_1000
				scores_matrix_uncer_1000[m][m] = np.multiply(scores_sum_uncer_local_1000, 1/trials)
				opt_scores_matrix_uncer_3000[m][m] = opt_scores_sum_uncer_local_3000
				scores_matrix_uncer_3000[m][m] = np.multiply(scores_sum_uncer_local_3000, 1/trials)
				opt_scores_matrix_uncer_10000[m][m] = opt_scores_sum_uncer_local_10000
				scores_matrix_uncer_10000[m][m] = np.multiply(scores_sum_uncer_local_10000, 1/trials)
			else:
				cost_uncer_100 = UNCERTAINTY_cond_on_Q2(m, n, network_ansatz_100)
				scores_sum_uncer_100 = 0
				opt_scores_sum_uncer_100 = []
				cost_uncer_300 = UNCERTAINTY_cond_on_Q2(m, n, network_ansatz_300)
				scores_sum_uncer_300 = 0
				opt_scores_sum_uncer_300 = []
				cost_uncer_1000 = UNCERTAINTY_cond_on_Q2(m, n, network_ansatz_1000)
				scores_sum_uncer_1000 = 0
				opt_scores_sum_uncer_1000 = []
				cost_uncer_3000 = UNCERTAINTY_cond_on_Q2(m, n, network_ansatz_3000)
				scores_sum_uncer_3000 = 0
				opt_scores_sum_uncer_3000 = []
				cost_uncer_10000 = UNCERTAINTY_cond_on_Q2(m, n, network_ansatz_10000)
				scores_sum_uncer_10000 = 0
				opt_scores_sum_uncer_10000 = []
				for i in range(trials):
					settings = network_ansatz_100.rand_network_settings()
					opt_dict_uncer_100 = qnet.gradient_descent(
						cost_uncer_100,
						settings,
						step_size=stepsizeUncer,
						num_steps=steps_uncer,
						sample_width=1,
						verbose=False,
					)
					scores_sum_uncer_100 -= np.array(opt_dict_uncer_100["scores"])
					opt_scores_sum_uncer_100.append(-np.array(opt_dict_uncer_100["opt_score"]))
					
					settings = network_ansatz_300.rand_network_settings()
					opt_dict_uncer_300 = qnet.gradient_descent(
						cost_uncer_300,
						settings,
						step_size=stepsizeUncer,
						num_steps=steps_uncer,
						sample_width=1,
						verbose=False,
					)
					scores_sum_uncer_300 -= np.array(opt_dict_uncer_300["scores"])
					opt_scores_sum_uncer_300.append(-np.array(opt_dict_uncer_300["opt_score"]))
					
					settings = network_ansatz_1000.rand_network_settings()
					opt_dict_uncer_1000 = qnet.gradient_descent(
						cost_uncer_1000,
						settings,
						step_size=stepsizeUncer,
						num_steps=steps_uncer,
						sample_width=1,
						verbose=False,
					)
					scores_sum_uncer_1000 -= np.array(opt_dict_uncer_1000["scores"])	
					opt_scores_sum_uncer_1000.append(-np.array(opt_dict_uncer_1000["opt_score"]))
					
					settings = network_ansatz_3000.rand_network_settings()
					opt_dict_uncer_3000 = qnet.gradient_descent(
						cost_uncer_3000,
						settings,
						step_size=stepsizeUncer,
						num_steps=steps_uncer,
						sample_width=1,
						verbose=False,
					)
					scores_sum_uncer_3000 -= np.array(opt_dict_uncer_3000["scores"])	
					opt_scores_sum_uncer_3000.append(-np.array(opt_dict_uncer_3000["opt_score"]))
					
					settings = network_ansatz_10000.rand_network_settings()
					opt_dict_uncer_10000 = qnet.gradient_descent(
						cost_uncer_10000,
						settings,
						step_size=stepsizeUncer,
						num_steps=steps_uncer,
						sample_width=1,
						verbose=False,
					)
					scores_sum_uncer_10000 -= np.array(opt_dict_uncer_10000["scores"])	
					opt_scores_sum_uncer_10000.append(-np.array(opt_dict_uncer_10000["opt_score"]))
					
				opt_scores_matrix_uncer_100[m][n] = opt_scores_sum_uncer_100	
				scores_matrix_uncer_100[m][n] = np.multiply(scores_sum_uncer_100, 1/trials)
				opt_scores_matrix_uncer_300[m][n] = opt_scores_sum_uncer_300	
				scores_matrix_uncer_300[m][n] = np.multiply(scores_sum_uncer_300, 1/trials)
				opt_scores_matrix_uncer_1000[m][n] = opt_scores_sum_uncer_1000
				scores_matrix_uncer_1000[m][n] = np.multiply(scores_sum_uncer_1000, 1/trials)
				opt_scores_matrix_uncer_3000[m][n] = opt_scores_sum_uncer_3000
				scores_matrix_uncer_3000[m][n] = np.multiply(scores_sum_uncer_3000, 1/trials)
				opt_scores_matrix_uncer_10000[m][n] = opt_scores_sum_uncer_10000
				scores_matrix_uncer_10000[m][n] = np.multiply(scores_sum_uncer_10000, 1/trials)

#record_matrix stores the list of scores for every step (lens of elements is lens(steps+1)), while record2_matrix scores the list of optimal scores for every trial (lens of elements is lens(trials))
#similar for list_of_error and list2_of_error
	record_matrix_uncer_100 = [[0 for _ in range (5)] for _ in range(5)]
	record_matrix_uncer_1000 = [[0 for _ in range (5)] for _ in range(5)]
	record_matrix_uncer_10000 = [[0 for _ in range (5)] for _ in range(5)]
	record2_matrix_uncer_100 = [[0 for _ in range (5)] for _ in range(5)]
	record2_matrix_uncer_1000 = [[0 for _ in range (5)] for _ in range(5)]
	record2_matrix_uncer_10000 = [[0 for _ in range (5)] for _ in range(5)]
	record_matrix_uncer_300 = [[0 for _ in range (5)] for _ in range(5)]
	record_matrix_uncer_3000 = [[0 for _ in range (5)] for _ in range(5)]
	record2_matrix_uncer_300 = [[0 for _ in range (5)] for _ in range(5)]
	record2_matrix_uncer_3000 = [[0 for _ in range (5)] for _ in range(5)]
	for a in range(steps_uncer+1):
		for m in range(5):
			for n in range(5):
				record_matrix_uncer_100[m][n] = scores_matrix_uncer_100[m][n][a]
				record_matrix_uncer_1000[m][n] = scores_matrix_uncer_1000[m][n][a]
				record_matrix_uncer_10000[m][n] = scores_matrix_uncer_10000[m][n][a]
				record_matrix_uncer_300[m][n] = scores_matrix_uncer_300[m][n][a]
				record_matrix_uncer_3000[m][n] = scores_matrix_uncer_3000[m][n][a]
		list_of_error_uncer_100.append(math.sqrt(((np.array(record_matrix_uncer_100) - np.array(matrix_ideal_uncer)) @ (np.array(record_matrix_uncer_100) - np.array(matrix_ideal_uncer)).transpose()).trace()))
		list_of_error_uncer_1000.append(math.sqrt(((np.array(record_matrix_uncer_1000) - np.array(matrix_ideal_uncer)) @ (np.array(record_matrix_uncer_1000) - np.array(matrix_ideal_uncer)).transpose()).trace()))
		list_of_error_uncer_10000.append(math.sqrt(((np.array(record_matrix_uncer_10000) - np.array(matrix_ideal_uncer)) @ (np.array(record_matrix_uncer_10000) - np.array(matrix_ideal_uncer)).transpose()).trace()))
		list_of_error_uncer_300.append(math.sqrt(((np.array(record_matrix_uncer_300) - np.array(matrix_ideal_uncer)) @ (np.array(record_matrix_uncer_300) - np.array(matrix_ideal_uncer)).transpose()).trace()))
		list_of_error_uncer_3000.append(math.sqrt(((np.array(record_matrix_uncer_3000) - np.array(matrix_ideal_uncer)) @ (np.array(record_matrix_uncer_3000) - np.array(matrix_ideal_uncer)).transpose()).trace()))
	for b in range(trials):
		for m in range(5):
			for n in range(5):
				record2_matrix_uncer_100[m][n] = opt_scores_matrix_uncer_100[m][n][b]
				record2_matrix_uncer_1000[m][n] = opt_scores_matrix_uncer_1000[m][n][b]
				record2_matrix_uncer_10000[m][n] = opt_scores_matrix_uncer_10000[m][n][b]
				record2_matrix_uncer_300[m][n] = opt_scores_matrix_uncer_300[m][n][b]
				record2_matrix_uncer_3000[m][n] = opt_scores_matrix_uncer_3000[m][n][b]
		list2_of_error_uncer_100.append(math.sqrt(((np.array(record2_matrix_uncer_100) - np.array(matrix_ideal_uncer)) @ (np.array(record2_matrix_uncer_100) - np.array(matrix_ideal_uncer)).transpose()).trace()))
		list2_of_error_uncer_1000.append(math.sqrt(((np.array(record2_matrix_uncer_1000) - np.array(matrix_ideal_uncer)) @ (np.array(record2_matrix_uncer_1000) - np.array(matrix_ideal_uncer)).transpose()).trace()))
		list2_of_error_uncer_10000.append(math.sqrt(((np.array(record2_matrix_uncer_10000) - np.array(matrix_ideal_uncer)) @ (np.array(record2_matrix_uncer_10000) - np.array(matrix_ideal_uncer)).transpose()).trace()))
		list2_of_error_uncer_300.append(math.sqrt(((np.array(record2_matrix_uncer_300) - np.array(matrix_ideal_uncer)) @ (np.array(record2_matrix_uncer_300) - np.array(matrix_ideal_uncer)).transpose()).trace()))
		list2_of_error_uncer_3000.append(math.sqrt(((np.array(record2_matrix_uncer_3000) - np.array(matrix_ideal_uncer)) @ (np.array(record2_matrix_uncer_3000) - np.array(matrix_ideal_uncer)).transpose()).trace()))
	list_of_error_uncer_mean = [np.mean(list2_of_error_uncer_100), np.mean(list2_of_error_uncer_300), np.mean(list2_of_error_uncer_1000), np.mean(list2_of_error_uncer_3000), np.mean(list2_of_error_uncer_10000)]
	list_of_error_uncer_standarderror = [sem(list2_of_error_uncer_100), sem(list2_of_error_uncer_300), sem(list2_of_error_uncer_1000), sem(list2_of_error_uncer_3000), sem(list2_of_error_uncer_10000)]

	print("100:", record_matrix_uncer_100)
	print("1000:", record_matrix_uncer_1000)
	print("10000:", record_matrix_uncer_10000)
	print("300:", record_matrix_uncer_300)
	print("3000:", record_matrix_uncer_3000)

	network_ansatz_alt_100 = qnet.NetworkAnsatz(
		prep_nodes,
		meas_nodes,
		dev_kwargs={
			"name": "default.qubit",
			"shots": 100,
		}
	)
	network_ansatz_alt_300 = qnet.NetworkAnsatz(
		prep_nodes,
		meas_nodes,
		dev_kwargs={
			"name": "default.qubit",
			"shots": 300,
		}
	)
	network_ansatz_alt_1000 = qnet.NetworkAnsatz(
		prep_nodes,
		meas_nodes,
		dev_kwargs={
			"name": "default.qubit",
			"shots": 1000,
		}
	)
	network_ansatz_alt_3000 = qnet.NetworkAnsatz(
		prep_nodes,
		meas_nodes,
		dev_kwargs={
			"name": "default.qubit",
			"shots": 3000,
		}
	)
	network_ansatz_alt_10000 = qnet.NetworkAnsatz(
		prep_nodes,
		meas_nodes,
		dev_kwargs={
			"name": "default.qubit",
			"shots": 10000,
		}
	)
	
	for m in range(5):
		for n in range(5):
			if m == n:
				cost_se_100 = SHANNONENTROPY(m, network_ansatz_alt_100)
				scores_sum_se_100 = 0
				opt_scores_sum_se_100 = []
				cost_se_300 = SHANNONENTROPY(m, network_ansatz_alt_300)
				scores_sum_se_300 = 0
				opt_scores_sum_se_300 = []
				cost_se_1000 = SHANNONENTROPY(m, network_ansatz_alt_1000)
				scores_sum_se_1000 = 0
				opt_scores_sum_se_1000 = []
				cost_se_3000 = SHANNONENTROPY(m, network_ansatz_alt_3000)
				scores_sum_se_3000 = 0
				opt_scores_sum_se_3000 = []
				cost_se_10000 = SHANNONENTROPY(m, network_ansatz_alt_10000)
				scores_sum_se_10000 = 0
				opt_scores_sum_se_10000 = []
				for i in range(trials):
					settings = network_ansatz_alt_100.rand_network_settings()
					opt_dict_se_100 = qnet.gradient_descent(
						cost_se_100,
						settings,
						step_size=stepsizese,
						num_steps=steps,
						sample_width=1,
						verbose=False,
					)
					scores_sum_se_100 -= np.array(opt_dict_se_100["scores"])
					opt_scores_sum_se_100.append(-np.array(opt_dict_se_100["opt_score"]))
					
					settings = network_ansatz_alt_300.rand_network_settings()
					opt_dict_se_300 = qnet.gradient_descent(
						cost_se_300,
						settings,
						step_size=stepsizese,
						num_steps=steps,
						sample_width=1,
						verbose=False,
					)
					scores_sum_se_300 -= np.array(opt_dict_se_300["scores"])
					opt_scores_sum_se_300.append(-np.array(opt_dict_se_300["opt_score"]))
					
					settings = network_ansatz_alt_1000.rand_network_settings()
					opt_dict_se_1000 = qnet.gradient_descent(
						cost_se_1000,
						settings,
						step_size=stepsizese,
						num_steps=steps,
						sample_width=1,
						verbose=False,
					)
					scores_sum_se_1000 -= np.array(opt_dict_se_1000["scores"])
					opt_scores_sum_se_1000.append(-np.array(opt_dict_se_1000["opt_score"]))
					
					settings = network_ansatz_alt_3000.rand_network_settings()
					opt_dict_se_3000 = qnet.gradient_descent(
						cost_se_3000,
						settings,
						step_size=stepsizese,
						num_steps=steps,
						sample_width=1,
						verbose=False,
					)
					scores_sum_se_3000 -= np.array(opt_dict_se_3000["scores"])
					opt_scores_sum_se_3000.append(-np.array(opt_dict_se_3000["opt_score"]))
					
					settings = network_ansatz_alt_10000.rand_network_settings()
					opt_dict_se_10000 = qnet.gradient_descent(
						cost_se_10000,
						settings,
						step_size=stepsizese,
						num_steps=steps,
						sample_width=1,
						verbose=False,
					)
					scores_sum_se_10000 -= np.array(opt_dict_se_10000["scores"])
					opt_scores_sum_se_10000.append(-np.array(opt_dict_se_10000["opt_score"]))
				opt_scores_matrix_mutualinfo_100[m][m] = opt_scores_sum_se_100
				scores_matrix_mutualinfo_100[m][m] = np.multiply(scores_sum_se_100, 1/trials)
				opt_scores_matrix_mutualinfo_300[m][m] = opt_scores_sum_se_300
				scores_matrix_mutualinfo_300[m][m] = np.multiply(scores_sum_se_300, 1/trials)
				opt_scores_matrix_mutualinfo_1000[m][m] = opt_scores_sum_se_1000
				scores_matrix_mutualinfo_1000[m][m] = np.multiply(scores_sum_se_1000, 1/trials)
				opt_scores_matrix_mutualinfo_3000[m][m] = opt_scores_sum_se_3000
				scores_matrix_mutualinfo_3000[m][m] = np.multiply(scores_sum_se_3000, 1/trials)
				opt_scores_matrix_mutualinfo_10000[m][m] = opt_scores_sum_se_10000
				scores_matrix_mutualinfo_10000[m][m] = np.multiply(scores_sum_se_10000, 1/trials)
			else:
				cost_mutualinfo_100 = Mutual_information(m, n, network_ansatz_alt_100)
				scores_sum_mutualinfo_100 = 0
				opt_scores_sum_mutualinfo_100 = []
				cost_mutualinfo_300 = Mutual_information(m, n, network_ansatz_alt_300)
				scores_sum_mutualinfo_300 = 0
				opt_scores_sum_mutualinfo_300 = []
				cost_mutualinfo_1000 = Mutual_information(m, n, network_ansatz_alt_1000)
				scores_sum_mutualinfo_1000 = 0
				opt_scores_sum_mutualinfo_1000 = []
				cost_mutualinfo_3000 = Mutual_information(m, n, network_ansatz_alt_3000)
				scores_sum_mutualinfo_3000 = 0
				opt_scores_sum_mutualinfo_3000 = []
				cost_mutualinfo_10000 = Mutual_information(m, n, network_ansatz_alt_10000)
				scores_sum_mutualinfo_10000 = 0
				opt_scores_sum_mutualinfo_10000 = []
				for i in range(trials):
					settings = network_ansatz_alt_100.rand_network_settings()
					opt_dict_mutualinfo_100 = qnet.gradient_descent(
						cost_mutualinfo_100,
						settings,
						step_size=stepsizeMutualInfo,
						num_steps=steps,
						sample_width=1,
						verbose=False,
					)
					scores_sum_mutualinfo_100 += np.array(opt_dict_mutualinfo_100["scores"])
					opt_scores_sum_mutualinfo_100.append(np.array(opt_dict_mutualinfo_100["opt_score"]))
					
					settings = network_ansatz_alt_300.rand_network_settings()
					opt_dict_mutualinfo_300 = qnet.gradient_descent(
						cost_mutualinfo_300,
						settings,
						step_size=stepsizeMutualInfo,
						num_steps=steps,
						sample_width=1,
						verbose=False,
					)
					scores_sum_mutualinfo_300 += np.array(opt_dict_mutualinfo_300["scores"])
					opt_scores_sum_mutualinfo_300.append(np.array(opt_dict_mutualinfo_300["opt_score"]))
					
					settings = network_ansatz_alt_1000.rand_network_settings()
					opt_dict_mutualinfo_1000 = qnet.gradient_descent(
						cost_mutualinfo_1000,
						settings,
						step_size=stepsizeMutualInfo,
						num_steps=steps,
						sample_width=1,
						verbose=False,
					)
					scores_sum_mutualinfo_1000 += np.array(opt_dict_mutualinfo_1000["scores"])
					opt_scores_sum_mutualinfo_1000.append(np.array(opt_dict_mutualinfo_1000["opt_score"]))
					
					settings = network_ansatz_alt_3000.rand_network_settings()
					opt_dict_mutualinfo_3000 = qnet.gradient_descent(
						cost_mutualinfo_3000,
						settings,
						step_size=stepsizeMutualInfo,
						num_steps=steps,
						sample_width=1,
						verbose=False,
					)
					scores_sum_mutualinfo_3000 += np.array(opt_dict_mutualinfo_3000["scores"])
					opt_scores_sum_mutualinfo_3000.append(np.array(opt_dict_mutualinfo_3000["opt_score"]))
					
					settings = network_ansatz_alt_10000.rand_network_settings()
					opt_dict_mutualinfo_10000 = qnet.gradient_descent(
						cost_mutualinfo_10000,
						settings,
						step_size=stepsizeMutualInfo,
						num_steps=steps,
						sample_width=1,
						verbose=False,
					)
					scores_sum_mutualinfo_10000 += np.array(opt_dict_mutualinfo_10000["scores"])
					opt_scores_sum_mutualinfo_10000.append(np.array(opt_dict_mutualinfo_10000["opt_score"]))
				opt_scores_matrix_mutualinfo_100[m][n] = opt_scores_sum_mutualinfo_100
				scores_matrix_mutualinfo_100[m][n] = np.multiply(scores_sum_mutualinfo_100, 1/trials)
				opt_scores_matrix_mutualinfo_300[m][n] = opt_scores_sum_mutualinfo_300
				scores_matrix_mutualinfo_300[m][n] = np.multiply(scores_sum_mutualinfo_300, 1/trials)
				opt_scores_matrix_mutualinfo_1000[m][n] = opt_scores_sum_mutualinfo_1000
				scores_matrix_mutualinfo_1000[m][n] = np.multiply(scores_sum_mutualinfo_1000, 1/trials)
				opt_scores_matrix_mutualinfo_3000[m][n] = opt_scores_sum_mutualinfo_3000
				scores_matrix_mutualinfo_3000[m][n] = np.multiply(scores_sum_mutualinfo_3000, 1/trials)
				opt_scores_matrix_mutualinfo_10000[m][n] = opt_scores_sum_mutualinfo_10000
				scores_matrix_mutualinfo_10000[m][n] = np.multiply(scores_sum_mutualinfo_10000, 1/trials)
	
	

	record_matrix_mutualinfo_100 = [[0 for _ in range (5)] for _ in range(5)]
	record_matrix_mutualinfo_1000 = [[0 for _ in range (5)] for _ in range(5)]
	record_matrix_mutualinfo_10000 = [[0 for _ in range (5)] for _ in range(5)]
	record2_matrix_mutualinfo_100 = [[0 for _ in range (5)] for _ in range(5)]
	record2_matrix_mutualinfo_1000 = [[0 for _ in range (5)] for _ in range(5)]
	record2_matrix_mutualinfo_10000 = [[0 for _ in range (5)] for _ in range(5)]
	record_matrix_mutualinfo_300 = [[0 for _ in range (5)] for _ in range(5)]
	record_matrix_mutualinfo_3000 = [[0 for _ in range (5)] for _ in range(5)]
	record2_matrix_mutualinfo_300 = [[0 for _ in range (5)] for _ in range(5)]
	record2_matrix_mutualinfo_3000 = [[0 for _ in range (5)] for _ in range(5)]
	for a in range(steps+1):
		for m in range(5):
			for n in range(5):
				record_matrix_mutualinfo_100[m][n] = scores_matrix_mutualinfo_100[m][n][a]
				record_matrix_mutualinfo_1000[m][n] = scores_matrix_mutualinfo_1000[m][n][a]
				record_matrix_mutualinfo_10000[m][n] = scores_matrix_mutualinfo_10000[m][n][a]
				record_matrix_mutualinfo_300[m][n] = scores_matrix_mutualinfo_300[m][n][a]
				record_matrix_mutualinfo_3000[m][n] = scores_matrix_mutualinfo_3000[m][n][a]
		list_of_error_mutualinfo_100.append(math.sqrt(((np.array(record_matrix_mutualinfo_100) - np.array(matrix_ideal_mutualinfo)) @ (np.array(record_matrix_mutualinfo_100) - np.array(matrix_ideal_mutualinfo)).transpose()).trace()))
		list_of_error_mutualinfo_1000.append(math.sqrt(((np.array(record_matrix_mutualinfo_1000) - np.array(matrix_ideal_mutualinfo)) @ (np.array(record_matrix_mutualinfo_1000) - np.array(matrix_ideal_mutualinfo)).transpose()).trace()))
		list_of_error_mutualinfo_10000.append(math.sqrt(((np.array(record_matrix_mutualinfo_10000) - np.array(matrix_ideal_mutualinfo)) @ (np.array(record_matrix_mutualinfo_10000) - np.array(matrix_ideal_mutualinfo)).transpose()).trace()))
		list_of_error_mutualinfo_300.append(math.sqrt(((np.array(record_matrix_mutualinfo_300) - np.array(matrix_ideal_mutualinfo)) @ (np.array(record_matrix_mutualinfo_300) - np.array(matrix_ideal_mutualinfo)).transpose()).trace()))
		list_of_error_mutualinfo_3000.append(math.sqrt(((np.array(record_matrix_mutualinfo_3000) - np.array(matrix_ideal_mutualinfo)) @ (np.array(record_matrix_mutualinfo_3000) - np.array(matrix_ideal_mutualinfo)).transpose()).trace()))
	for b in range(trials):
		for m in range(5):
			for n in range(5):
				record2_matrix_mutualinfo_100[m][n] = opt_scores_matrix_mutualinfo_100[m][n][b]
				record2_matrix_mutualinfo_1000[m][n] = opt_scores_matrix_mutualinfo_1000[m][n][b]
				record2_matrix_mutualinfo_10000[m][n] = opt_scores_matrix_mutualinfo_10000[m][n][b]
				record2_matrix_mutualinfo_300[m][n] = opt_scores_matrix_mutualinfo_300[m][n][b]
				record2_matrix_mutualinfo_3000[m][n] = opt_scores_matrix_mutualinfo_3000[m][n][b]
		list2_of_error_mutualinfo_100.append(math.sqrt(((np.array(record2_matrix_mutualinfo_100) - np.array(matrix_ideal_mutualinfo)) @ (np.array(record2_matrix_mutualinfo_100) - np.array(matrix_ideal_mutualinfo)).transpose()).trace()))
		list2_of_error_mutualinfo_1000.append(math.sqrt(((np.array(record2_matrix_mutualinfo_1000) - np.array(matrix_ideal_mutualinfo)) @ (np.array(record2_matrix_mutualinfo_1000) - np.array(matrix_ideal_mutualinfo)).transpose()).trace()))
		list2_of_error_mutualinfo_10000.append(math.sqrt(((np.array(record2_matrix_mutualinfo_10000) - np.array(matrix_ideal_mutualinfo)) @ (np.array(record2_matrix_mutualinfo_10000) - np.array(matrix_ideal_mutualinfo)).transpose()).trace()))
		list2_of_error_mutualinfo_300.append(math.sqrt(((np.array(record2_matrix_mutualinfo_300) - np.array(matrix_ideal_mutualinfo)) @ (np.array(record2_matrix_mutualinfo_300) - np.array(matrix_ideal_mutualinfo)).transpose()).trace()))
		list2_of_error_mutualinfo_3000.append(math.sqrt(((np.array(record2_matrix_mutualinfo_3000) - np.array(matrix_ideal_mutualinfo)) @ (np.array(record2_matrix_mutualinfo_3000) - np.array(matrix_ideal_mutualinfo)).transpose()).trace()))
	list_of_error_mutualinfo_mean = [np.mean(list2_of_error_mutualinfo_100), np.mean(list2_of_error_mutualinfo_300), np.mean(list2_of_error_mutualinfo_1000), np.mean(list2_of_error_mutualinfo_3000), np.mean(list2_of_error_mutualinfo_10000)]
	list_of_error_mutualinfo_standarderror = [sem(list2_of_error_mutualinfo_100), sem(list2_of_error_mutualinfo_300), sem(list2_of_error_mutualinfo_1000), sem(list2_of_error_mutualinfo_3000), sem(list2_of_error_mutualinfo_10000)]
	print("100:", record_matrix_mutualinfo_100)
	print("1000:", record_matrix_mutualinfo_1000)
	print("10000:", record_matrix_mutualinfo_10000)
	print("300:", record_matrix_mutualinfo_300)
	print("3000:", record_matrix_mutualinfo_3000)
	
	for m in range(5):
		for n in range(5):
			if m == n:
				cost_var_100 = Variance(m, network_ansatz_alt_100)
				scores_sum_var_100 = 0
				opt_scores_sum_var_100 = []
				cost_var_300 = Variance(m, network_ansatz_alt_300)
				scores_sum_var_300 = 0
				opt_scores_sum_var_300 = []
				cost_var_1000 = Variance(m, network_ansatz_alt_1000)
				scores_sum_var_1000 = 0
				opt_scores_sum_var_1000 = []
				cost_var_3000 = Variance(m, network_ansatz_alt_3000)
				scores_sum_var_3000 = 0
				opt_scores_sum_var_3000 = []
				cost_var_10000 = Variance(m, network_ansatz_alt_10000)
				scores_sum_var_10000 = 0
				opt_scores_sum_var_10000 = []
				for i in range(trials):
					settings = network_ansatz_alt_100.rand_network_settings()
					opt_dict_var_100 = qnet.gradient_descent(
						cost_var_100,
						settings,
						step_size=stepsizeCovar,
						num_steps=steps,
						sample_width=1,
						verbose=False,
					)
					scores_sum_var_100 += np.array(opt_dict_var_100["scores"])
					opt_scores_sum_var_100.append(np.array(opt_dict_var_100["opt_score"]))
					
					settings = network_ansatz_alt_300.rand_network_settings()
					opt_dict_var_300 = qnet.gradient_descent(
						cost_var_300,
						settings,
						step_size=stepsizeCovar,
						num_steps=steps,
						sample_width=1,
						verbose=False,
					)
					scores_sum_var_300 += np.array(opt_dict_var_300["scores"])
					opt_scores_sum_var_300.append(np.array(opt_dict_var_300["opt_score"]))
					
					settings = network_ansatz_alt_1000.rand_network_settings()
					opt_dict_var_1000 = qnet.gradient_descent(
						cost_var_1000,
						settings,
						step_size=stepsizeCovar,
						num_steps=steps,
						sample_width=1,
						verbose=False,
					)
					scores_sum_var_1000 += np.array(opt_dict_var_1000["scores"])
					opt_scores_sum_var_1000.append(np.array(opt_dict_var_1000["opt_score"]))
					
					settings = network_ansatz_alt_3000.rand_network_settings()
					opt_dict_var_3000 = qnet.gradient_descent(
						cost_var_3000,
						settings,
						step_size=stepsizeCovar,
						num_steps=steps,
						sample_width=1,
						verbose=False,
					)
					scores_sum_var_3000 += np.array(opt_dict_var_3000["scores"])
					opt_scores_sum_var_3000.append(np.array(opt_dict_var_3000["opt_score"]))
					
					settings = network_ansatz_alt_10000.rand_network_settings()
					opt_dict_var_10000 = qnet.gradient_descent(
						cost_var_10000,
						settings,
						step_size=stepsizeCovar,
						num_steps=steps,
						sample_width=1,
						verbose=False,
					)
					scores_sum_var_10000 += np.array(opt_dict_var_10000["scores"])
					opt_scores_sum_var_10000.append(np.array(opt_dict_var_10000["opt_score"]))
					
				opt_scores_matrix_covar_100[m][n] = opt_scores_sum_var_100
				scores_matrix_covar_100[m][n] = np.multiply(scores_sum_var_100, 1/trials)
				opt_scores_matrix_covar_300[m][n] = opt_scores_sum_var_300
				scores_matrix_covar_300[m][n] = np.multiply(scores_sum_var_300, 1/trials)
				opt_scores_matrix_covar_1000[m][n] = opt_scores_sum_var_1000
				scores_matrix_covar_1000[m][n] = np.multiply(scores_sum_var_1000, 1/trials)
				opt_scores_matrix_covar_3000[m][n] = opt_scores_sum_var_3000
				scores_matrix_covar_3000[m][n] = np.multiply(scores_sum_var_3000, 1/trials)
				opt_scores_matrix_covar_10000[m][n] = opt_scores_sum_var_10000
				scores_matrix_covar_10000[m][n] = np.multiply(scores_sum_var_10000, 1/trials)
			else:
				cost_covar_100 = Covariance(m, n, network_ansatz_alt_100)
				scores_sum_covar_100 = 0
				opt_scores_sum_covar_100 = []
				cost_covar_300 = Covariance(m, n, network_ansatz_alt_300)
				scores_sum_covar_300 = 0
				opt_scores_sum_covar_300 = []
				cost_covar_1000 = Covariance(m, n, network_ansatz_alt_1000)
				scores_sum_covar_1000 = 0
				opt_scores_sum_covar_1000 = []
				cost_covar_3000 = Covariance(m, n, network_ansatz_alt_3000)
				scores_sum_covar_3000 = 0
				opt_scores_sum_covar_3000 = []
				cost_covar_10000 = Covariance(m, n, network_ansatz_alt_10000)
				scores_sum_covar_10000 = 0
				opt_scores_sum_covar_10000 = []
				for i in range(trials):
					settings = network_ansatz_alt_100.rand_network_settings()
					opt_dict_covar_100 = qnet.gradient_descent(
						cost_covar_100,
						settings,
						step_size=stepsizeCovar,
						num_steps=steps,
						sample_width=1,
						verbose=False,
					)
					scores_sum_covar_100 += np.array(opt_dict_covar_100["scores"])
					opt_scores_sum_covar_100.append(np.array(opt_dict_covar_100["opt_score"]))
					
					settings = network_ansatz_alt_300.rand_network_settings()
					opt_dict_covar_300 = qnet.gradient_descent(
						cost_covar_300,
						settings,
						step_size=stepsizeCovar,
						num_steps=steps,
						sample_width=1,
						verbose=False,
					)
					scores_sum_covar_300 += np.array(opt_dict_covar_300["scores"])
					opt_scores_sum_covar_300.append(np.array(opt_dict_covar_300["opt_score"]))
					
					settings = network_ansatz_alt_1000.rand_network_settings()
					opt_dict_covar_1000 = qnet.gradient_descent(
						cost_covar_1000,
						settings,
						step_size=stepsizeCovar,
						num_steps=steps,
						sample_width=1,
						verbose=False,
					)
					scores_sum_covar_1000 += np.array(opt_dict_covar_1000["scores"])
					opt_scores_sum_covar_1000.append(np.array(opt_dict_covar_1000["opt_score"]))
					
					settings = network_ansatz_alt_3000.rand_network_settings()
					opt_dict_covar_3000 = qnet.gradient_descent(
						cost_covar_3000,
						settings,
						step_size=stepsizeCovar,
						num_steps=steps,
						sample_width=1,
						verbose=False,
					)
					scores_sum_covar_3000 += np.array(opt_dict_covar_3000["scores"])
					opt_scores_sum_covar_3000.append(np.array(opt_dict_covar_3000["opt_score"]))
					
					settings = network_ansatz_alt_10000.rand_network_settings()
					opt_dict_covar_10000 = qnet.gradient_descent(
						cost_covar_10000,
						settings,
						step_size=stepsizeCovar,
						num_steps=steps,
						sample_width=1,
						verbose=False,
					)
					scores_sum_covar_10000 += np.array(opt_dict_covar_10000["scores"])
					opt_scores_sum_covar_10000.append(np.array(opt_dict_covar_10000["opt_score"]))
					
				opt_scores_matrix_covar_100[m][n] = opt_scores_sum_covar_100
				scores_matrix_covar_100[m][n] = np.multiply(scores_sum_covar_100, 1/trials)
				opt_scores_matrix_covar_300[m][n] = opt_scores_sum_covar_300
				scores_matrix_covar_300[m][n] = np.multiply(scores_sum_covar_300, 1/trials)
				opt_scores_matrix_covar_1000[m][n] = opt_scores_sum_covar_1000
				scores_matrix_covar_1000[m][n] = np.multiply(scores_sum_covar_1000, 1/trials)
				opt_scores_matrix_covar_3000[m][n] = opt_scores_sum_covar_3000
				scores_matrix_covar_3000[m][n] = np.multiply(scores_sum_covar_3000, 1/trials)
				opt_scores_matrix_covar_10000[m][n] = opt_scores_sum_covar_10000
				scores_matrix_covar_10000[m][n] = np.multiply(scores_sum_covar_10000, 1/trials)
				
	record_matrix_covar_100 = [[0 for _ in range (5)] for _ in range(5)]
	record_matrix_covar_1000 = [[0 for _ in range (5)] for _ in range(5)]
	record_matrix_covar_10000 = [[0 for _ in range (5)] for _ in range(5)]
	record2_matrix_covar_100 = [[0 for _ in range (5)] for _ in range(5)]
	record2_matrix_covar_1000 = [[0 for _ in range (5)] for _ in range(5)]
	record2_matrix_covar_10000 = [[0 for _ in range (5)] for _ in range(5)]
	record_matrix_covar_300 = [[0 for _ in range (5)] for _ in range(5)]
	record_matrix_covar_3000 = [[0 for _ in range (5)] for _ in range(5)]
	record2_matrix_covar_300 = [[0 for _ in range (5)] for _ in range(5)]
	record2_matrix_covar_3000 = [[0 for _ in range (5)] for _ in range(5)]
	for a in range(steps+1):
		for m in range(5):
			for n in range(5):
				record_matrix_covar_100[m][n] = scores_matrix_covar_100[m][n][a]
				record_matrix_covar_1000[m][n] = scores_matrix_covar_1000[m][n][a]
				record_matrix_covar_10000[m][n] = scores_matrix_covar_10000[m][n][a]
				record_matrix_covar_300[m][n] = scores_matrix_covar_300[m][n][a]
				record_matrix_covar_3000[m][n] = scores_matrix_covar_3000[m][n][a]
		list_of_error_covar_100.append(math.sqrt(((np.array(record_matrix_covar_100) - np.array(matrix_ideal_covar)) @ (np.array(record_matrix_covar_100) - np.array(matrix_ideal_covar)).transpose()).trace()))
		list_of_error_covar_1000.append(math.sqrt(((np.array(record_matrix_covar_1000) - np.array(matrix_ideal_covar)) @ (np.array(record_matrix_covar_1000) - np.array(matrix_ideal_covar)).transpose()).trace()))
		list_of_error_covar_10000.append(math.sqrt(((np.array(record_matrix_covar_10000) - np.array(matrix_ideal_covar)) @ (np.array(record_matrix_covar_10000) - np.array(matrix_ideal_covar)).transpose()).trace()))
		list_of_error_covar_300.append(math.sqrt(((np.array(record_matrix_covar_300) - np.array(matrix_ideal_covar)) @ (np.array(record_matrix_covar_300) - np.array(matrix_ideal_covar)).transpose()).trace()))
		list_of_error_covar_3000.append(math.sqrt(((np.array(record_matrix_covar_3000) - np.array(matrix_ideal_covar)) @ (np.array(record_matrix_covar_3000) - np.array(matrix_ideal_covar)).transpose()).trace()))
	for b in range(trials):
		for m in range(5):
			for n in range(5):		
				record2_matrix_covar_100[m][n] = opt_scores_matrix_covar_100[m][n][b]
				record2_matrix_covar_1000[m][n] = opt_scores_matrix_covar_1000[m][n][b]
				record2_matrix_covar_10000[m][n] = opt_scores_matrix_covar_10000[m][n][b]
				record2_matrix_covar_300[m][n] = opt_scores_matrix_covar_300[m][n][b]
				record2_matrix_covar_3000[m][n] = opt_scores_matrix_covar_3000[m][n][b]
		list2_of_error_covar_100.append(math.sqrt(((np.array(record2_matrix_covar_100) - np.array(matrix_ideal_covar)) @ (np.array(record2_matrix_covar_100) - np.array(matrix_ideal_covar)).transpose()).trace()))
		list2_of_error_covar_1000.append(math.sqrt(((np.array(record2_matrix_covar_1000) - np.array(matrix_ideal_covar)) @ (np.array(record2_matrix_covar_1000) - np.array(matrix_ideal_covar)).transpose()).trace()))
		list2_of_error_covar_10000.append(math.sqrt(((np.array(record2_matrix_covar_10000) - np.array(matrix_ideal_covar)) @ (np.array(record2_matrix_covar_10000) - np.array(matrix_ideal_covar)).transpose()).trace()))
		list2_of_error_covar_300.append(math.sqrt(((np.array(record2_matrix_covar_300) - np.array(matrix_ideal_covar)) @ (np.array(record2_matrix_covar_300) - np.array(matrix_ideal_covar)).transpose()).trace()))
		list2_of_error_covar_3000.append(math.sqrt(((np.array(record2_matrix_covar_3000) - np.array(matrix_ideal_covar)) @ (np.array(record2_matrix_covar_3000) - np.array(matrix_ideal_covar)).transpose()).trace()))
	list_of_error_covar_mean = [np.mean(list2_of_error_covar_100), np.mean(list2_of_error_covar_300), np.mean(list2_of_error_covar_1000), np.mean(list2_of_error_covar_3000), np.mean(list2_of_error_covar_10000)]
	list_of_error_covar_standarderror = [sem(list2_of_error_covar_100), sem(list2_of_error_covar_300), sem(list2_of_error_covar_1000), sem(list2_of_error_covar_3000), sem(list2_of_error_covar_10000)]

	print("100:", record_matrix_covar_100)
	print("1000:", record_matrix_covar_1000)
	print("10000:", record_matrix_covar_10000)
	print("300:", record_matrix_covar_300)
	print("3000:", record_matrix_covar_3000)
	
	fig1, (ax1, ax2, ax3) = plt.subplots(figsize = (15,4), ncols=3)
	ax1.semilogy(range(0,steps_uncer+1), np.array(list_of_error_uncer_100), "o-", markersize=3.0, color='red', label="100 shots", linewidth=1.5)
	ax1.semilogy(range(0,steps_uncer+1), np.array(list_of_error_uncer_300), "o-", markersize=3.0, color='orange', label="300 shots", linewidth=1.5)
	ax1.semilogy(range(0,steps_uncer+1), np.array(list_of_error_uncer_1000), "o-", markersize=3.0, color='forestgreen', label="1000 shots", linewidth=1.5)
	ax1.semilogy(range(0,steps_uncer+1), np.array(list_of_error_uncer_3000), "o-", markersize=3.0, color='dodgerblue', label="3000 shots", linewidth=1.5)
	ax1.semilogy(range(0,steps_uncer+1), np.array(list_of_error_uncer_10000), "o-", markersize=3.0, color='blueviolet', label="10000 shots", linewidth=1.5)
	ax1.set_title('Uncertainty')
	ax1.set_ylabel('Distance to ideal matrix')
	ax1.set_xlabel('Optimization Step')
	ax1.set_xlim(0,steps_uncer)
	ax1.set_xticks(range(0, steps_uncer+1, 10))
	ax1.grid()
	ax2.semilogy(range(0,steps+1), np.array(list_of_error_mutualinfo_100), "o-", markersize=3.0, color='red', label="100 shots", linewidth=1.5)
	ax2.semilogy(range(0,steps+1), np.array(list_of_error_mutualinfo_300), "o-", markersize=3.0, color='orange', label="300 shots", linewidth=1.5)
	ax2.semilogy(range(0,steps+1), np.array(list_of_error_mutualinfo_1000), "o-", markersize=3.0, color='forestgreen', label="1000 shots", linewidth=1.5)
	ax2.semilogy(range(0,steps+1), np.array(list_of_error_mutualinfo_3000), "o-", markersize=3.0, color='dodgerblue', label="3000 shots", linewidth=1.5)
	ax2.semilogy(range(0,steps+1), np.array(list_of_error_mutualinfo_10000), "o-", markersize=3.0, color='blueviolet', label="10000 shots", linewidth=1.5)
	ax2.set_title('Characteristic')
	ax2.set_xlabel('Optimization Step')
	ax2.set_xlim(0,steps)
	ax2.set_xticks(range(0, steps+1, 10))
	ax2.grid()
	ax3.semilogy(range(0,steps+1), np.array(list_of_error_covar_100), "o-", markersize=3.0, color='red', label="100 shots", linewidth=1.5)
	ax3.semilogy(range(0,steps+1), np.array(list_of_error_covar_300), "o-", markersize=3.0, color='orange', label="300 shots", linewidth=1.5)
	ax3.semilogy(range(0,steps+1), np.array(list_of_error_covar_1000), "o-", markersize=3.0, color='forestgreen', label="1000 shots", linewidth=1.5)
	ax3.semilogy(range(0,steps+1), np.array(list_of_error_covar_3000), "o-", markersize=3.0, color='dodgerblue', label="3000 shots", linewidth=1.5)
	ax3.semilogy(range(0,steps+1), np.array(list_of_error_covar_10000), "o-", markersize=3.0, color='blueviolet', label="10000 shots", linewidth=1.5)
	ax3.set_title('Covariance')
	ax3.set_xlabel('Optimization Step')
	ax3.set_xlim(0,steps)
	ax3.set_xticks(range(0, steps+1, 10))
	ax3.grid()

	
	label = ["0","1","2","3","4"]
	fig2, ((ax10, ax11, ax12, ax13), (ax20, ax21, ax22, ax23), (ax30, ax31, ax32, ax33)) = plt.subplots(3, 4, figsize = (15,9))
	image10 = ax10.imshow(matrix_ideal_uncer, cmap='YlGnBu', vmin=0, vmax=2, interpolation='none')
	ax10.grid(which="minor", color="k", linestyle='-', linewidth=2)
	ax10.set_xticks(np.arange(5),labels=label)
	ax10.set_yticks(np.arange(5),labels=label)
	ax10.tick_params(top=True, bottom=False,labeltop=True, labelbottom=False)
	ax10.set_title("Ideal")
	image11 = ax11.imshow(record_matrix_uncer_100, cmap='YlGnBu', vmin=0, vmax=2, interpolation='none')
	ax11.grid(which="minor", color="k", linestyle='-', linewidth=2)
	ax11.set_xticks(np.arange(5),labels=label)
	ax11.set_yticks(np.arange(5),labels=label)
	ax11.tick_params(top=True, bottom=False,labeltop=True, labelbottom=False)
	ax11.set_title("100 shots")
	image12 = ax12.imshow(record_matrix_uncer_1000, cmap='YlGnBu', vmin=0, vmax=2, interpolation='none')
	ax12.grid(which="minor", color="k", linestyle='-', linewidth=1)
	ax12.set_xticks(np.arange(5),labels=label)
	ax12.set_yticks(np.arange(5),labels=label)
	ax12.tick_params(top=True, bottom=False,labeltop=True, labelbottom=False)
	ax12.set_title("1000 shots")	
	image13 = ax13.imshow(record_matrix_uncer_10000, cmap='YlGnBu', vmin=0, vmax=2, interpolation='none')
	ax13.grid(which="minor", color="k", linestyle='-', linewidth=1)
	ax13.set_xticks(np.arange(5),labels=label)
	ax13.set_yticks(np.arange(5),labels=label)
	ax13.tick_params(top=True, bottom=False,labeltop=True, labelbottom=False)
	ax13.set_title("10000 shots")
	cbar1 = fig2.colorbar(image13, ax=ax13)
	cbar1.set_label("Uncertainty", rotation=-90, va="bottom")
	
	image20 = ax20.imshow(matrix_ideal_mutualinfo, cmap='viridis', vmin=0, vmax=1, interpolation='none')
	ax20.grid(which="minor", color="k", linestyle='-', linewidth=1)
	ax20.set_xticks(np.arange(5),labels=label)
	ax20.set_yticks(np.arange(5),labels=label)
	ax20.tick_params(top=True, bottom=False,labeltop=True, labelbottom=False)
	image21 = ax21.imshow(record_matrix_mutualinfo_100, cmap='viridis', vmin=0, vmax=1, interpolation='none')
	ax21.grid(which="minor", color="k", linestyle='-', linewidth=1)
	ax21.set_xticks(np.arange(5),labels=label)
	ax21.set_yticks(np.arange(5),labels=label)
	ax21.tick_params(top=True, bottom=False,labeltop=True, labelbottom=False)
	image22 = ax22.imshow(record_matrix_mutualinfo_1000, cmap='viridis', vmin=0, vmax=1, interpolation='none')
	ax22.grid(which="minor", color="k", linestyle='-', linewidth=1)
	ax22.set_xticks(np.arange(5),labels=label)
	ax22.set_yticks(np.arange(5),labels=label)
	ax22.tick_params(top=True, bottom=False,labeltop=True, labelbottom=False)
	image23 = ax23.imshow(record_matrix_mutualinfo_10000, cmap='viridis', vmin=0, vmax=1, interpolation='none')
	ax23.grid(which="minor", color="k", linestyle='-', linewidth=1)
	ax23.set_xticks(np.arange(5),labels=label)
	ax23.set_yticks(np.arange(5),labels=label)
	ax23.tick_params(top=True, bottom=False,labeltop=True, labelbottom=False)
	cbar2 = fig2.colorbar(image23, ax=ax23)
	cbar2.set_label("Characteristic", rotation=-90, va="bottom")
	
	image30 = ax30.imshow(matrix_ideal_covar, cmap='viridis', vmin=0, vmax=1, interpolation='none')
	ax30.grid(which="minor", color="k", linestyle='-', linewidth=1)	
	ax30.set_xticks(np.arange(5),labels=label)
	ax30.set_yticks(np.arange(5),labels=label)
	ax30.tick_params(top=True, bottom=False,labeltop=True, labelbottom=False)
	image31 = ax31.imshow(record_matrix_covar_100, cmap='viridis', vmin=0, vmax=1, interpolation='none')
	ax31.grid(which="minor", color="k", linestyle='-', linewidth=1)	
	ax31.set_xticks(np.arange(5),labels=label)
	ax31.set_yticks(np.arange(5),labels=label)
	ax31.tick_params(top=True, bottom=False,labeltop=True, labelbottom=False)
	image32 = ax32.imshow(record_matrix_covar_1000, cmap='viridis', vmin=0, vmax=1, interpolation='none')
	ax32.grid(which="minor", color="k", linestyle='-', linewidth=1)
	ax32.set_xticks(np.arange(5),labels=label)
	ax32.set_yticks(np.arange(5),labels=label)
	ax32.tick_params(top=True, bottom=False,labeltop=True, labelbottom=False)	
	image33 = ax33.imshow(record_matrix_covar_10000, cmap='viridis', vmin=0, vmax=1, interpolation='none')
	ax33.grid(which="minor", color="k", linestyle='-', linewidth=1)
	ax33.set_xticks(np.arange(5),labels=label)
	ax33.set_yticks(np.arange(5),labels=label)
	ax33.tick_params(top=True, bottom=False,labeltop=True, labelbottom=False)
	cbar3 = fig2.colorbar(image33, ax=ax33)
	cbar3.set_label("Covariance", rotation=-90, va="bottom")
	
	list_of_error_uncer_min = [np.array(list2_of_error_uncer_100).min(), np.array(list2_of_error_uncer_300).min(), np.array(list2_of_error_uncer_1000).min(), np.array(list2_of_error_uncer_3000).min(), np.array(list2_of_error_uncer_10000).min()]
	list_of_error_mutualinfo_min = [np.array(list2_of_error_mutualinfo_100).min(), np.array(list2_of_error_mutualinfo_300).min(), np.array(list2_of_error_mutualinfo_1000).min(), np.array(list2_of_error_mutualinfo_3000).min(), np.array(list2_of_error_mutualinfo_10000).min()]
	list_of_error_covar_min = [np.array(list2_of_error_covar_100).min(), np.array(list2_of_error_covar_300).min(), np.array(list2_of_error_covar_1000).min(), np.array(list2_of_error_covar_3000).min(), np.array(list2_of_error_covar_10000).min()]
	
	fig3 = plt.figure(figsize = (7,6)).add_subplot(111)
	fig3.loglog([100,300,1000,3000,10000], np.array(list_of_error_uncer_mean), "o-", color='dodgerblue', label="Uncertainty", linewidth=2)
	fig3.fill_between([100,300,1000,3000,10000], np.array(list_of_error_uncer_mean) + np.array(list_of_error_uncer_standarderror), np.array(list_of_error_uncer_mean) - np.array(list_of_error_uncer_standarderror), alpha=.4, linewidth=1)
	fig3.loglog([100,300,1000,3000,10000], np.array(list_of_error_mutualinfo_mean), "o-",  color='orange', label="Characteristic", linewidth=2)
	fig3.fill_between([100,300,1000,3000,10000], np.array(list_of_error_mutualinfo_mean) + np.array(list_of_error_mutualinfo_standarderror), np.array(list_of_error_mutualinfo_mean) - np.array(list_of_error_mutualinfo_standarderror), alpha=.4, linewidth=1)
	fig3.loglog([100,300,1000,3000,10000], np.array(list_of_error_covar_mean), "o-",  color='forestgreen', label="Covariance", linewidth=2)
	fig3.fill_between([100,300,1000,3000,10000], np.array(list_of_error_covar_mean) + np.array(list_of_error_covar_standarderror), np.array(list_of_error_covar_mean) - np.array(list_of_error_covar_standarderror), alpha=.4, linewidth=1)
	fig3.loglog([100,300,1000,3000,10000], np.array(list_of_error_uncer_min), "v--", color='dodgerblue', label='Minimum deviance for uncertainty', linewidth=2)
	fig3.loglog([100,300,1000,3000,10000], np.array(list_of_error_mutualinfo_min), "v--", color='orange', label='Minimum deviance for mutual information', linewidth=2)
	fig3.loglog([100,300,1000,3000,10000], np.array(list_of_error_covar_min), "v--", color='forestgreen', label='Minimum deviance for covariance', linewidth=2)
	fig3.set_ylabel('Distance to ideal matrix', size=16)
	fig3.set_xlabel('Number of measurement shots', size=16)
	fig3.grid()
	
	fig4 = plt.figure(figsize = (7,6)).add_subplot(111)
	fig4.loglog([100,300,1000,3000,10000], np.array(list_of_error_uncer_mean), "o-", color='dodgerblue', label="Uncertainty", linewidth=2)
	fig4.fill_between([100,300,1000,3000,10000], np.array(list_of_error_uncer_mean) + np.array(list_of_error_uncer_standarderror), np.array(list_of_error_uncer_mean) - np.array(list_of_error_uncer_standarderror), alpha=.4, linewidth=1)
	fig4.loglog([100,300,1000,3000,10000], np.array(list_of_error_mutualinfo_mean), "o-",  color='orange', label="Characteristic", linewidth=2)
	fig4.fill_between([100,300,1000,3000,10000], np.array(list_of_error_mutualinfo_mean) + np.array(list_of_error_mutualinfo_standarderror), np.array(list_of_error_mutualinfo_mean) - np.array(list_of_error_mutualinfo_standarderror), alpha=.4, linewidth=1)
	fig4.loglog([100,300,1000,3000,10000], np.array(list_of_error_covar_mean), "o-",  color='forestgreen', label="Covariance", linewidth=2)
	fig4.fill_between([100,300,1000,3000,10000], np.array(list_of_error_covar_mean) + np.array(list_of_error_covar_standarderror), np.array(list_of_error_covar_mean) - np.array(list_of_error_covar_standarderror), alpha=.4, linewidth=1)
	fig4.scatter([100,300,1000,3000,10000], [np.array(list2_of_error_uncer_100).min(), np.array(list2_of_error_uncer_300).min(), np.array(list2_of_error_uncer_1000).min(), np.array(list2_of_error_uncer_3000).min(), np.array(list2_of_error_uncer_10000).min()], s=30, c='dodgerblue', label='Minimum deviance for uncertainty', marker='v')
	fig4.scatter([100,300,1000,3000,10000], [np.array(list2_of_error_mutualinfo_100).min(), np.array(list2_of_error_mutualinfo_300).min(), np.array(list2_of_error_mutualinfo_1000).min(), np.array(list2_of_error_mutualinfo_3000).min(), np.array(list2_of_error_mutualinfo_10000).min()], s=30, c='orange', label='Minimum deviance for mutual information', marker='v')
	fig4.scatter([100,300,1000,3000,10000], [np.array(list2_of_error_covar_100).min(), np.array(list2_of_error_covar_300).min(), np.array(list2_of_error_covar_1000).min(), np.array(list2_of_error_covar_3000).min(), np.array(list2_of_error_covar_10000).min()], s=30, c='forestgreen', label='Minimum deviance for covariance', marker='v')
	fig4.set_ylabel('Distance to ideal matrix', size=16)
	fig4.set_xlabel('Number of measurement shots', size=16)
	fig4.grid()
	
	
Optimization_wrt_steps(20, 70, 70, 0.10, 0.10, 0.10, 0.25, 0.3)


end_time = time.time()
print("Time cost:", end_time - start_time)

plt.show()
