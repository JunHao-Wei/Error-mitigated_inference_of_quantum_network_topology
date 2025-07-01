import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import qnetvo as qnet
from scipy.stats import sem

#Preparing an EPR pair on wire 0&1 and an maximally entangled pair after one-sided rotation
ghz_prep_node = [
    qnet.PrepareNode(1, [0,1], qnet.ghz_state, 0)
]
max_entangled_prep_node = [
    qnet.PrepareNode(1, [0,1], qnet.max_entangled_state, 3)
]
reduced_ghz_prep_node = [
     qnet.PrepareNode(1, [0,1,6], qnet.ghz_state, 0)
]

#Using one ancilla for each wire to implement probabilistic operation (basis choice)
def processing_ansatz(settings, wires):
	qml.Hadamard(wires=wires[2])
	qml.Hadamard(wires=wires[3])
	qml.ControlledQubitUnitary(qml.Hadamard(wires=wires[0]), control_wires=wires[2])
	qml.ControlledQubitUnitary(qml.Hadamard(wires=wires[1]), control_wires=wires[3])
proc_nodes = [
	qnet.ProcessingNode(
		wires=[0,1,2,3],
		ansatz_fn=processing_ansatz
	)
]

#Arbitrary single qubit measurement basis for optimization
meas_nodes = [
    qnet.MeasureNode(1, 2, [0], ansatz_fn=qml.ArbitraryUnitary, num_settings=3),
    qnet.MeasureNode(1, 2, [1], ansatz_fn=qml.ArbitraryUnitary, num_settings=3),
    #qnet.MeasureNode(1, 2, [2], num_settings=0),
    #qnet.MeasureNode(1, 2, [3], num_settings=0),
]

#Defining the cost function: uncertainty conditioned on measurement results of qubit1
def UNCERTAINTY_cond_on_Q1(ansatz):#the parameter of the cost function needs to be the network ansatz
	@qml.qnode(qml.device(**ansatz.dev_kwargs))
	def circ(settings):#define the quantum circuit function without specifying the network ansatz
		ansatz.fn(settings)
		return qml.probs([0,1,2,3])#Care!
	def Uncertainty_cond_on_Q1(*settings):
		Probs = circ(settings)#claim the probability (which are not real numbers yet) throught the above defined function 
		Marginal_Probs_Q1_basis1 = [(Probs[0]+Probs[4])/(Probs[0]+Probs[4]+Probs[8]+Probs[12]), (Probs[8]+Probs[12])/(Probs[0]+Probs[4]+Probs[8]+Probs[12])]
		Marginal_Probs_Q1_basis2 = [(Probs[3]+Probs[7])/(Probs[3]+Probs[7]+Probs[11]+Probs[15]), (Probs[11]+Probs[15])/(Probs[3]+Probs[7]+Probs[11]+Probs[15])]
		Joint_Probs_basis1 = [Probs[0]/(Probs[0] + Probs[4] + Probs[8] + Probs[12]), Probs[4]/(Probs[0] + Probs[4] + Probs[8] + Probs[12]), Probs[8]/(Probs[0] + Probs[4] + Probs[8] + Probs[12]), Probs[12]/(Probs[0] + Probs[4] + Probs[8] + Probs[12])]
		Joint_Probs_basis2 = [Probs[3]/(Probs[3] + Probs[7] + Probs[11] + Probs[15]), Probs[7]/(Probs[3] + Probs[7] + Probs[11] + Probs[15]), Probs[11]/(Probs[3] + Probs[7] + Probs[11] + Probs[15]), Probs[15]/(Probs[3] + Probs[7] + Probs[11] + Probs[15])]
		uncertainty_cond_on_Q1 = qnet.shannon_entropy(Joint_Probs_basis1) + qnet.shannon_entropy(Joint_Probs_basis2) - qnet.shannon_entropy(Marginal_Probs_Q1_basis1) - qnet.shannon_entropy(Marginal_Probs_Q1_basis2)
		return uncertainty_cond_on_Q1
	return Uncertainty_cond_on_Q1#return the secondly defined function which is parametrized on settings (specified in "qnet.gradient_descent" function later)
#Defining the cost function: uncertainty conditioned on measurement results of qubit2
def UNCERTAINTY_cond_on_Q2(ansatz):
	@qml.qnode(qml.device(**ansatz.dev_kwargs))
	def circ(settings):
		ansatz.fn(settings)
		return qml.probs([0,1,2,3])
	def Uncertainty_cond_on_Q2(*settings):
		Probs = circ(settings)
		Marginal_Probs_Q2_basis1 = [(Probs[0]+Probs[8])/(Probs[0]+Probs[4]+Probs[8]+Probs[12]), (Probs[4]+Probs[12])/(Probs[0]+Probs[4]+Probs[8]+Probs[12])]
		Marginal_Probs_Q2_basis2 = [(Probs[3]+Probs[11])/(Probs[3]+Probs[7]+Probs[11]+Probs[15]), (Probs[7]+Probs[15])/(Probs[3]+Probs[7]+Probs[11]+Probs[15])]
		Joint_Probs_basis1 = [Probs[0]/(Probs[0] + Probs[4] + Probs[8] + Probs[12]), Probs[4]/(Probs[0] + Probs[4] + Probs[8] + Probs[12]), Probs[8]/(Probs[0] + Probs[4] + Probs[8] + Probs[12]), Probs[12]/(Probs[0] + Probs[4] + Probs[8] + Probs[12])]
		Joint_Probs_basis2 = [Probs[3]/(Probs[3] + Probs[7] + Probs[11] + Probs[15]), Probs[7]/(Probs[3] + Probs[7] + Probs[11] + Probs[15]), Probs[11]/(Probs[3] + Probs[7] + Probs[11] + Probs[15]), Probs[15]/(Probs[3] + Probs[7] + Probs[11] + Probs[15])]
		uncertainty_cond_on_Q2 = qnet.shannon_entropy(Joint_Probs_basis1) + qnet.shannon_entropy(Joint_Probs_basis2) - qnet.shannon_entropy(Marginal_Probs_Q2_basis1) - qnet.shannon_entropy(Marginal_Probs_Q2_basis2)
		return uncertainty_cond_on_Q2
	return Uncertainty_cond_on_Q2


#Defining the cost function: mutual information
def Mutual_information(ansatz):
	@qml.qnode(qml.device(**ansatz.dev_kwargs))
	def circ2(settings):
		ansatz.fn(settings)
		return qml.probs([0,1])
	def Mutual_info(*settings):
		Probs = circ2(settings)
		Marginal_Probs_Q1 = [Probs[0]+Probs[1], Probs[2]+Probs[3]]
		Marginal_Probs_Q2 = [Probs[0]+Probs[2], Probs[1]+Probs[3]]
		mutual_info = qnet.shannon_entropy(Probs) - qnet.shannon_entropy(Marginal_Probs_Q1) - qnet.shannon_entropy(Marginal_Probs_Q2)#Note that the mutual information is negative here as it is to be maximized
		return mutual_info
	return Mutual_info

#Defining the cost function: covariance
def Covariance(ansatz):
	@qml.qnode(qml.device(**ansatz.dev_kwargs))
	def circ2(settings):
		ansatz.fn(settings)
		return qml.probs([0,1])
	def Covar(*settings):
		Probs = circ2(settings)
		Marginal_Probs_Q1 = [Probs[0]+Probs[1], Probs[2]+Probs[3]]
		Marginal_Probs_Q2 = [Probs[0]+Probs[2], Probs[1]+Probs[3]]
		covar = (Marginal_Probs_Q1[0] - Marginal_Probs_Q1[1])*(Marginal_Probs_Q2[0] - Marginal_Probs_Q2[1]) - (Probs[0] - Probs[1] - Probs[2] + Probs[3])
		return covar
	return Covar


def Optimization_EPR(trials, shots, steps):
	EPR_network_ansatz = qnet.NetworkAnsatz(
		ghz_prep_node,
		proc_nodes,
		meas_nodes,
		dev_kwargs={
			"name": "default.qubit",
			"shots": shots,}
	)
	EPR_alt_network_ansatz = qnet.NetworkAnsatz(
		ghz_prep_node,
		meas_nodes,
		dev_kwargs={
			"name": "default.qubit",
			"shots": shots,}
	)	

	cost_condQ1_EPR = UNCERTAINTY_cond_on_Q1(EPR_network_ansatz)

	scores_condQ1_EPR = []
	scores_sum_condQ1_EPR = 0

	
	for i in range(trials):
		settings = EPR_network_ansatz.rand_network_settings()#randomize the initial setting for every trial of optimization
		opt_dict_condQ1_EPR = qnet.gradient_descent(
			cost_condQ1_EPR,
			settings,
			step_size=0.10,
			num_steps=steps,
			sample_width=1,
			verbose=False,
		)
		print(opt_dict_condQ1_EPR["opt_score"])
		scores_condQ1_EPR.append(-np.array(opt_dict_condQ1_EPR["scores"]))
		scores_sum_condQ1_EPR -= np.array(opt_dict_condQ1_EPR["scores"])

		
	scores_aver_condQ1_EPR  = np.multiply(scores_sum_condQ1_EPR, 1/trials) 
	
	#Calculating the standard error of scores of every iteration step
	#convert the list of tensors into 2D lists
	scores_list_condQ1_EPR = []
	for i in range(trials):
		a = scores_condQ1_EPR[i].tolist()
		scores_list_condQ1_EPR.append(a)
	#calculating the standard error of each column (index j) of the 2D lists (i.e. scores of each iteration step averaged over trials)
	scores_condQ1_EPR_column = []
	scores_condQ1_EPR_standarderror = []
	for j in range(steps+1):
		for i in range(trials):
			scores_condQ1_EPR_column.append(scores_list_condQ1_EPR[i][j])
		scores_condQ1_EPR_standarderror.append(sem(scores_condQ1_EPR_column))
		scores_condQ1_EPR_column = []
	
	#Optimization for the mutual information and covariance of EPR through gradient descent
	cost_mutual_info_EPR = Mutual_information(EPR_alt_network_ansatz)
	cost_covar_EPR = Covariance(EPR_alt_network_ansatz)
	
	scores_mutual_info_EPR = []
	scores_sum_mutual_info_EPR = 0
	for i in range(trials):
		settings = EPR_alt_network_ansatz.rand_network_settings()
		opt_dict_mutual_info_EPR = qnet.gradient_descent(
		    cost_mutual_info_EPR,
		    settings,
		    step_size=0.1,
		    num_steps=steps,
		    sample_width=1,
		    verbose=False,
		)
		print(opt_dict_mutual_info_EPR["opt_score"])
		scores_mutual_info_EPR.append(np.array(opt_dict_mutual_info_EPR["scores"]))
		scores_sum_mutual_info_EPR += np.array(opt_dict_mutual_info_EPR["scores"])
	scores_aver_mutual_info_EPR = np.multiply(scores_sum_mutual_info_EPR, 1/trials) 
	
	scores_covar_EPR = []
	scores_sum_covar_EPR = 0
	for i in range(trials):
		settings = EPR_alt_network_ansatz.rand_network_settings()
		opt_dict_covar_EPR = qnet.gradient_descent(
		    cost_covar_EPR,
		    settings,
		    step_size=0.2,#Doubled
		    num_steps=steps,
		    sample_width=1,
		    verbose=False,
		)
		print(opt_dict_covar_EPR["opt_score"])
		scores_covar_EPR.append(np.array(opt_dict_covar_EPR["scores"]))
		scores_sum_covar_EPR += np.array(opt_dict_covar_EPR["scores"])
	scores_aver_covar_EPR  = np.multiply(scores_sum_covar_EPR, 1/trials)
	 
	#Calculating the standard error of scores of every iteration step
	scores_list_mutual_info_EPR = []
	scores_list_covar_EPR = []
	for i in range(trials):
		a = scores_mutual_info_EPR[i].tolist()
		b = scores_covar_EPR[i].tolist()
		scores_list_mutual_info_EPR.append(a)
		scores_list_covar_EPR.append(b)
	scores_mutual_info_EPR_column = []
	scores_mutual_info_EPR_standarderror = []
	scores_covar_EPR_column = []
	scores_covar_EPR_standarderror = []
	for j in range(steps+1):
		for i in range(trials):
			scores_mutual_info_EPR_column.append(scores_list_mutual_info_EPR[i][j])
			scores_covar_EPR_column.append(scores_list_covar_EPR[i][j])
		scores_mutual_info_EPR_standarderror.append(sem(scores_mutual_info_EPR_column))
		scores_covar_EPR_standarderror.append(sem(scores_covar_EPR_column))
		scores_mutual_info_EPR_column = []
		scores_covar_EPR_column = []
		
	fig1 = plt.figure('Optimization for EPR', figsize = (8,7)).add_subplot(111)
	fig1.plot(range(0,steps+1), np.array(np.multiply(scores_aver_condQ1_EPR, 0.5)), label="Uncertainty", linewidth=2)
	fig1.plot(range(0,steps+1), 1 - np.array(scores_aver_mutual_info_EPR), label="Mutual information", linewidth=2)
	fig1.plot(range(0,steps+1), 0.5 - np.array(np.multiply(scores_aver_covar_EPR, 0.5)), label="Covariance", linewidth=2)#Halved, since covariance ranges from -1 to 1.
	fig1.fill_between(range(0,steps+1), np.array(np.multiply(scores_aver_condQ1_EPR, 0.5)) + np.array(np.multiply(scores_condQ1_EPR_standarderror, 0.5)), np.array(np.multiply(scores_aver_condQ1_EPR, 0.5)) - np.array(np.multiply(scores_condQ1_EPR_standarderror, 0.5)), alpha=.5, linewidth=0)
	fig1.fill_between(range(0,steps+1), 1 - (np.array(scores_aver_mutual_info_EPR) + np.array(scores_mutual_info_EPR_standarderror)), 1- (np.array(scores_aver_mutual_info_EPR) - np.array(scores_mutual_info_EPR_standarderror)), alpha=.5, linewidth=0)
	fig1.fill_between(range(0,steps+1), 0.5 - (np.array(np.multiply(scores_aver_covar_EPR, 0.5)) + np.array(np.multiply(scores_covar_EPR_standarderror, 0.5))), 0.5 - (np.array(np.multiply(scores_aver_covar_EPR, 0.5)) - np.array(np.multiply(scores_covar_EPR_standarderror, 0.5))), alpha=.5, linewidth=0)
	fig1.set_title('EPR State', size=24)
	fig1.set_ylabel('Relative Deviation from Ideal Value', size=16)
	fig1.set_xlabel('Gradient Descent Iteration', size=16)
	fig1.grid()
	fig1.legend(fontsize=20)

def Optimization_ReducedGHZ(trials, shots, steps):
	ReducedGHZ_network_ansatz = qnet.NetworkAnsatz(
		reduced_ghz_prep_node,
		proc_nodes,
		meas_nodes,
		dev_kwargs={
			"name": "default.qubit",
			"shots": shots,}
	)
	ReducedGHZ_alt_network_ansatz = qnet.NetworkAnsatz(
		reduced_ghz_prep_node,
		meas_nodes,
		dev_kwargs={
			"name": "default.qubit",
			"shots": shots,}
	)
	cost_condQ1_ReducedGHZ = UNCERTAINTY_cond_on_Q1(ReducedGHZ_network_ansatz)
	cost_condQ2_ReducedGHZ = UNCERTAINTY_cond_on_Q2(ReducedGHZ_network_ansatz)	
	
	scores_condQ1_ReducedGHZ = []
	scores_sum_condQ1_ReducedGHZ = 0
	scores_condQ2_ReducedGHZ = []
	scores_sum_condQ2_ReducedGHZ = 0
	
	for i in range(trials):
		settings = ReducedGHZ_network_ansatz.rand_network_settings()#randomize the initial setting for every trial of optimization
		opt_dict_condQ1_ReducedGHZ = qnet.gradient_descent(
			cost_condQ1_ReducedGHZ,
			settings,
			step_size=0.1,
			num_steps=steps,
			sample_width=1,
			verbose=False,
		)
		print(opt_dict_condQ1_ReducedGHZ["opt_score"])
		scores_condQ1_ReducedGHZ.append(-np.array(opt_dict_condQ1_ReducedGHZ["scores"]))
		scores_sum_condQ1_ReducedGHZ -= np.array(opt_dict_condQ1_ReducedGHZ["scores"])
		
	scores_aver_condQ1_ReducedGHZ  = np.multiply(scores_sum_condQ1_ReducedGHZ, 1/trials) 
	
	scores_list_condQ1_ReducedGHZ = []
	for i in range(trials):
		a = scores_condQ1_ReducedGHZ[i].tolist()
		scores_list_condQ1_ReducedGHZ.append(a)
	scores_condQ1_ReducedGHZ_column = []
	scores_condQ1_ReducedGHZ_standarderror = []
	for j in range(steps+1):
		for i in range(trials):
			scores_condQ1_ReducedGHZ_column.append(scores_list_condQ1_ReducedGHZ[i][j])
		scores_condQ1_ReducedGHZ_standarderror.append(sem(scores_condQ1_ReducedGHZ_column))
		scores_condQ1_ReducedGHZ_column = []
	
	#Optimization for the mutual information and covariance of ReducedGHZ through gradient descent
	cost_mutual_info_ReducedGHZ = Mutual_information(ReducedGHZ_alt_network_ansatz)
	cost_covar_ReducedGHZ = Covariance(ReducedGHZ_alt_network_ansatz)
	
	scores_mutual_info_ReducedGHZ = []
	scores_sum_mutual_info_ReducedGHZ = 0
	for i in range(trials):
		settings = ReducedGHZ_alt_network_ansatz.rand_network_settings()
		opt_dict_mutual_info_ReducedGHZ = qnet.gradient_descent(
		    cost_mutual_info_ReducedGHZ,
		    settings,
		    step_size=0.25,
		    num_steps=steps,
		    sample_width=1,
		    verbose=False,
		)
		print(opt_dict_mutual_info_ReducedGHZ["opt_score"])
		scores_mutual_info_ReducedGHZ.append(np.array(opt_dict_mutual_info_ReducedGHZ["scores"]))
		scores_sum_mutual_info_ReducedGHZ += np.array(opt_dict_mutual_info_ReducedGHZ["scores"])
	scores_aver_mutual_info_ReducedGHZ = np.multiply(scores_sum_mutual_info_ReducedGHZ, 1/trials) 
	
	scores_covar_ReducedGHZ = []
	scores_sum_covar_ReducedGHZ = 0
	for i in range(trials):
		settings = ReducedGHZ_alt_network_ansatz.rand_network_settings()
		opt_dict_covar_ReducedGHZ = qnet.gradient_descent(
		    cost_covar_ReducedGHZ,
		    settings,
		    step_size=0.3,#Doubled
		    num_steps=steps,
		    sample_width=1,
		    verbose=False,
		)
		print(opt_dict_covar_ReducedGHZ["opt_score"])
		scores_covar_ReducedGHZ.append(np.array(opt_dict_covar_ReducedGHZ["scores"]))
		scores_sum_covar_ReducedGHZ += np.array(opt_dict_covar_ReducedGHZ["scores"])
	scores_aver_covar_ReducedGHZ  = np.multiply(scores_sum_covar_ReducedGHZ, 1/trials)
	 
	#Calculating the standard error of scores of every iteration step
	scores_list_mutual_info_ReducedGHZ = []
	scores_list_covar_ReducedGHZ = []
	for i in range(trials):
		a = scores_mutual_info_ReducedGHZ[i].tolist()
		b = scores_covar_ReducedGHZ[i].tolist()
		scores_list_mutual_info_ReducedGHZ.append(a)
		scores_list_covar_ReducedGHZ.append(b)
	scores_mutual_info_ReducedGHZ_column = []
	scores_mutual_info_ReducedGHZ_standarderror = []
	scores_covar_ReducedGHZ_column = []
	scores_covar_ReducedGHZ_standarderror = []
	for j in range(steps+1):
		for i in range(trials):
			scores_mutual_info_ReducedGHZ_column.append(scores_list_mutual_info_ReducedGHZ[i][j])
			scores_covar_ReducedGHZ_column.append(scores_list_covar_ReducedGHZ[i][j])
		scores_mutual_info_ReducedGHZ_standarderror.append(sem(scores_mutual_info_ReducedGHZ_column))
		scores_covar_ReducedGHZ_standarderror.append(sem(scores_covar_ReducedGHZ_column))
		scores_mutual_info_ReducedGHZ_column = []
		scores_covar_ReducedGHZ_column = []
	
	fig2 = plt.figure('Optimization for Reduced GHZ state', figsize = (8,7)).add_subplot(111)
	fig2.plot(range(0,steps+1), -0.5 + np.array(np.multiply(scores_aver_condQ1_ReducedGHZ, 0.5)), label="Uncertainty", linewidth=2)
	fig2.plot(range(0,steps+1), 1 - np.array(scores_aver_mutual_info_ReducedGHZ), label="Mutual information", linewidth=2)
	fig2.plot(range(0,steps+1), 0.5 - np.array(np.multiply(scores_aver_covar_ReducedGHZ, 0.5)), label="Covariance", linewidth=2)#Halved, since covariance ranges from -1 to 1.
	fig2.fill_between(range(0,steps+1), -0.5 + (np.array(np.multiply(scores_aver_condQ1_ReducedGHZ, 0.5)) + np.array(np.multiply(scores_condQ1_ReducedGHZ_standarderror, 0.5))), -0.5 + (np.array(np.multiply(scores_aver_condQ1_ReducedGHZ, 0.5)) - np.array(np.multiply(scores_condQ1_ReducedGHZ_standarderror, 0.5))), alpha=.5, linewidth=0)
	fig2.fill_between(range(0,steps+1), 1 - (np.array(scores_aver_mutual_info_ReducedGHZ) + np.array(scores_mutual_info_ReducedGHZ_standarderror)), 1 - (np.array(scores_aver_mutual_info_ReducedGHZ) - np.array(scores_mutual_info_ReducedGHZ_standarderror)), alpha=.5, linewidth=0)
	fig2.fill_between(range(0,steps+1), 0.5 - (np.array(np.multiply(scores_aver_covar_ReducedGHZ, 0.5)) + np.array(np.multiply(scores_covar_ReducedGHZ_standarderror, 0.5))), 0.5 - (np.array(np.multiply(scores_aver_covar_ReducedGHZ, 0.5)) - np.array(np.multiply(scores_covar_ReducedGHZ_standarderror, 0.5))), alpha=.5, linewidth=0)
	fig2.set_title('GHZ State', size=24)
	fig2.set_ylabel('Relative Deviation from Ideal Value', size=16)
	fig2.set_xlabel('Gradient Descent Iteration', size=16)
	fig2.grid()
	fig2.legend(fontsize=20)

Optimization_EPR(100, 10000, 30)
Optimization_ReducedGHZ(100, 10000, 50)
plt.show()
