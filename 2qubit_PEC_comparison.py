import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import qnetvo as qnet
from scipy.stats import sem
import math
from sympy import *
from sympy.codegen.cfunctions import log2
import time
import sympy as sp


start_time = time.time()

ghz_prep_nodes_PEC = [
    qnet.PrepareNode(1, [0,1], qnet.ghz_state, 0),
    qnet.PrepareNode(1, [2,3], qnet.ghz_state, 0),
    qnet.PrepareNode(1, [4,5], qnet.ghz_state, 0),
    qnet.PrepareNode(1, [6,7], qnet.ghz_state, 0),
]
max_entangled_prep_nodes_PEC = [
    qnet.PrepareNode(1, [0,1], qnet.max_entangled_state, 3),
    qnet.PrepareNode(1, [2,3], qnet.max_entangled_state, 3),
    qnet.PrepareNode(1, [4,5], qnet.max_entangled_state, 3),
    qnet.PrepareNode(1, [6,7], qnet.max_entangled_state, 3),
]

def qubit_phase_damping_nodes_PEC_alt2(gamma1):
	qubit_phase_damping_nodes_alt2 = [
	    qnet.NoiseNode(wires=[0], ansatz_fn=lambda settings, wires: qml.PhaseDamping(gamma1, wires=wires)), 
	    qnet.NoiseNode(wires=[1], ansatz_fn=lambda settings, wires: qml.PhaseDamping(gamma1, wires=wires)),
	    qnet.NoiseNode(wires=[2], ansatz_fn=lambda settings, wires: qml.PhaseDamping(gamma1, wires=wires)), 
	    qnet.NoiseNode(wires=[3], ansatz_fn=lambda settings, wires: qml.PhaseDamping(gamma1, wires=wires)),
	    qnet.NoiseNode(wires=[4], ansatz_fn=lambda settings, wires: qml.PhaseDamping(gamma1, wires=wires)), 
	    qnet.NoiseNode(wires=[5], ansatz_fn=lambda settings, wires: qml.PhaseDamping(gamma1, wires=wires)),
	    qnet.NoiseNode(wires=[6], ansatz_fn=lambda settings, wires: qml.PhaseDamping(gamma1, wires=wires)), 
	    qnet.NoiseNode(wires=[7], ansatz_fn=lambda settings, wires: qml.PhaseDamping(gamma1, wires=wires)),
	]
	return qubit_phase_damping_nodes_alt2


#Arbitrary single qubit measurement basis for optimization
meas_nodes_PEC = [
    qnet.MeasureNode(1, 2, [0], ansatz_fn=None, num_settings=0),
    qnet.MeasureNode(1, 2, [1], ansatz_fn=None, num_settings=0),
    qnet.MeasureNode(1, 2, [2], ansatz_fn=None, num_settings=0),
    qnet.MeasureNode(1, 2, [3], ansatz_fn=None, num_settings=0),
    qnet.MeasureNode(1, 2, [4], ansatz_fn=None, num_settings=0),
    qnet.MeasureNode(1, 2, [5], ansatz_fn=None, num_settings=0),
    qnet.MeasureNode(1, 2, [6], ansatz_fn=None, num_settings=0),
    qnet.MeasureNode(1, 2, [7], ansatz_fn=None, num_settings=0),
]
	
#Define the post-processed probability of the PEC process
def Probs_PECproc(Probs_pre, gamma2, n):
	Probs = []
	for i in range(n):
		prob_proc_0 = (1 + math.sqrt(1-gamma2))* (1 + math.sqrt(1-gamma2)) * Probs_pre[0][i] / (4 * math.sqrt((1-gamma2)*(1-gamma2)))
		prob_proc_1 = (1 + math.sqrt(1-gamma2))* (1 - math.sqrt(1-gamma2)) * Probs_pre[1][i] / (4 * math.sqrt((1-gamma2)*(1-gamma2)))
		prob_proc_2 = (1 - math.sqrt(1-gamma2))* (1 + math.sqrt(1-gamma2)) * Probs_pre[2][i] / (4 * math.sqrt((1-gamma2)*(1-gamma2)))
		prob_proc_3 = (1 - math.sqrt(1-gamma2))* (1 - math.sqrt(1-gamma2)) * Probs_pre[3][i] / (4 * math.sqrt((1-gamma2)*(1-gamma2)))
		Probs.append(prob_proc_0 - prob_proc_1 - prob_proc_2 + prob_proc_3)
	return Probs
	
def UNCERTAINTY_cond_on_Q1_PEC(ansatz, gamma2):
	@qml.qnode(qml.device(**ansatz.dev_kwargs))
	def circ(settings):
		ansatz.fn(settings)
		return qml.probs([0,1,8,9]), qml.probs([2,3,8,9]), qml.probs([4,5,8,9]), qml.probs([6,7,8,9])
	def Uncertainty_cond_on_Q1_PEC(*settings):
		Probs = Probs_PECproc(circ(settings),gamma2, 16)
		Marginal_Probs_Q1_basis1 = [(Probs[0]+Probs[4])/(Probs[0]+Probs[4]+Probs[8]+Probs[12]), (Probs[8]+Probs[12])/(Probs[0]+Probs[4]+Probs[8]+Probs[12])]
		Marginal_Probs_Q1_basis2 = [(Probs[3]+Probs[7])/(Probs[3]+Probs[7]+Probs[11]+Probs[15]), (Probs[11]+Probs[15])/(Probs[3]+Probs[7]+Probs[11]+Probs[15])]
		Joint_Probs_basis1 = [Probs[0]/(Probs[0] + Probs[4] + Probs[8] + Probs[12]), Probs[4]/(Probs[0] + Probs[4] + Probs[8] + Probs[12]), Probs[8]/(Probs[0] + Probs[4] + Probs[8] + Probs[12]), Probs[12]/(Probs[0] + Probs[4] + Probs[8] + Probs[12])]
		Joint_Probs_basis2 = [Probs[3]/(Probs[3] + Probs[7] + Probs[11] + Probs[15]), Probs[7]/(Probs[3] + Probs[7] + Probs[11] + Probs[15]), Probs[11]/(Probs[3] + Probs[7] + Probs[11] + Probs[15]), Probs[15]/(Probs[3] + Probs[7] + Probs[11] + Probs[15])]
		uncertainty_cond_on_Q1_PEC = qnet.shannon_entropy(Joint_Probs_basis1) + qnet.shannon_entropy(Joint_Probs_basis2) - qnet.shannon_entropy(Marginal_Probs_Q1_basis1) - qnet.shannon_entropy(Marginal_Probs_Q1_basis2)
		return uncertainty_cond_on_Q1_PEC
	return Uncertainty_cond_on_Q1_PEC

def Mutual_information_PEC(ansatz, gamma2):
	@qml.qnode(qml.device(**ansatz.dev_kwargs))
	def circ2(settings):
		ansatz.fn(settings)
		return qml.probs([0,1]), qml.probs([2,3]), qml.probs([4,5]), qml.probs([6,7])
	def Mutual_info_PEC(*settings):
		Probs = Probs_PECproc(circ2(settings), gamma2, 4)
		Marginal_Probs_Q1 = [Probs[0]+Probs[1], Probs[2]+Probs[3]]
		Marginal_Probs_Q2 = [Probs[0]+Probs[2], Probs[1]+Probs[3]]
		mutual_info_PEC = qnet.shannon_entropy(Probs) - qnet.shannon_entropy(Marginal_Probs_Q1) - qnet.shannon_entropy(Marginal_Probs_Q2)
		return mutual_info_PEC
	return Mutual_info_PEC
def Covariance_PEC(ansatz, gamma2):
	@qml.qnode(qml.device(**ansatz.dev_kwargs))
	def circ2(settings):
		ansatz.fn(settings)
		return qml.probs([0,1]), qml.probs([2,3]), qml.probs([4,5]), qml.probs([6,7])
	def Covar_PEC(*settings):
		Probs = Probs_PECproc(circ2(settings), gamma2, 4)
		Marginal_Probs_Q1 = [Probs[0]+Probs[1], Probs[2]+Probs[3]]
		Marginal_Probs_Q2 = [Probs[0]+Probs[2], Probs[1]+Probs[3]]
		covar_PEC = (Marginal_Probs_Q1[0] - Marginal_Probs_Q1[1])*(Marginal_Probs_Q2[0] - Marginal_Probs_Q2[1]) - (Probs[0] - Probs[1] - Probs[2] + Probs[3])
		return covar_PEC
	return Covar_PEC

H = [[1/math.sqrt(2), 1/math.sqrt(2)],[1/math.sqrt(2), -1/math.sqrt(2)]]
H4 = np.kron(H, np.kron(H, np.kron(H, H)))

opt_score_aver_condQ1_EPR_PEC = []
opt_score_condQ1_EPR_PEC_standarderror = []
opt_score_aver_mutual_info_EPR_PEC = []
opt_score_mutual_info_EPR_PEC_standarderror = []
opt_score_aver_covar_EPR_PEC = []
opt_score_covar_EPR_PEC_standarderror = []

def Optimization_EPR_PEC(trials, steps, gamma1, gamma2, stepsizeUncern, stepsizeMutualInfo, stepsizeCovar):
	def processing_ansatz_PEC(settings, wires):
		qml.PauliZ(wires=wires[2])
		qml.PauliZ(wires=wires[5])
		qml.PauliZ(wires=wires[6])
		qml.PauliZ(wires=wires[7])
		qml.Hadamard(wires=wires[8])
		qml.Hadamard(wires=wires[9])
		qml.ControlledQubitUnitary(H4, control_wires=wires[8], wires=wires[0:8:2])
		qml.ControlledQubitUnitary(H4, control_wires=wires[9], wires=wires[1:9:2])
		qml.Rot(*settings[0:3], wires=wires[0])
		qml.Rot(*settings[3:6], wires=wires[1])
		qml.Rot(*settings[0:3], wires=wires[2])
		qml.Rot(*settings[3:6], wires=wires[3])
		qml.Rot(*settings[0:3], wires=wires[4])
		qml.Rot(*settings[3:6], wires=wires[5])
		qml.Rot(*settings[0:3], wires=wires[6])
		qml.Rot(*settings[3:6], wires=wires[7])
	proc_nodes_PEC = [
		qnet.ProcessingNode(
			wires=[0,1,2,3,4,5,6,7,8,9],
			ansatz_fn=processing_ansatz_PEC,
			num_settings = 6
		)
	]
	EPR_network_ansatz_PEC = qnet.NetworkAnsatz(
		ghz_prep_nodes_PEC,
		qubit_phase_damping_nodes_PEC_alt2(gamma1),
		proc_nodes_PEC,
		meas_nodes_PEC,
		dev_kwargs={
			"name": "default.mixed",
		}
	)
	def processing_ansatz_PEC_alt(settings, wires):
		qml.PauliZ(wires=wires[2])
		qml.PauliZ(wires=wires[5])
		qml.PauliZ(wires=wires[6])
		qml.PauliZ(wires=wires[7])
		qml.Rot(*settings[0:3], wires=wires[0])
		qml.Rot(*settings[3:6], wires=wires[1])
		qml.Rot(*settings[0:3], wires=wires[2])
		qml.Rot(*settings[3:6], wires=wires[3])
		qml.Rot(*settings[0:3], wires=wires[4])
		qml.Rot(*settings[3:6], wires=wires[5])
		qml.Rot(*settings[0:3], wires=wires[6])
		qml.Rot(*settings[3:6], wires=wires[7])
	proc_nodes_PEC_alt = [
		qnet.ProcessingNode(
			wires=[0,1,2,3,4,5,6,7],
			ansatz_fn=processing_ansatz_PEC_alt,
			num_settings = 6
		)
	]
	EPR_alt_network_ansatz_PEC = qnet.NetworkAnsatz(
		ghz_prep_nodes_PEC,
		qubit_phase_damping_nodes_PEC_alt2(gamma1),
		proc_nodes_PEC_alt,
		meas_nodes_PEC,
		dev_kwargs={
			"name": "default.mixed",
		}
	)
		
	
	cost_condQ1_EPR_PEC = UNCERTAINTY_cond_on_Q1_PEC(EPR_network_ansatz_PEC, gamma2)

	opt_score_condQ1_EPR_PEC = []
	opt_score_sum_condQ1_EPR_PEC = 0
	
	for i in range(trials):
		settings = EPR_network_ansatz_PEC.rand_network_settings()#randomize the initial setting for every trial of optimization
		opt_dict_condQ1_EPR_PEC = qnet.gradient_descent(
			cost_condQ1_EPR_PEC,
			settings,
			step_size=stepsizeUncern,
			num_steps=steps,
			sample_width=1,
			verbose=True,
		)
		print(opt_dict_condQ1_EPR_PEC["opt_score"])
		opt_score_condQ1_EPR_PEC.append(-np.array(opt_dict_condQ1_EPR_PEC["opt_score"]))
		opt_score_sum_condQ1_EPR_PEC -= opt_dict_condQ1_EPR_PEC["opt_score"]
		
	opt_score_aver_condQ1_EPR_PEC.append(opt_score_sum_condQ1_EPR_PEC * (1/trials))
	opt_score_condQ1_EPR_PEC_standarderror.append(sem(opt_score_condQ1_EPR_PEC))
	
	
	#Optimization for the mutual information and covariance of EPR through gradient descent
	cost_mutual_info_EPR_PEC = Mutual_information_PEC(EPR_alt_network_ansatz_PEC, gamma2)
	cost_covar_EPR_PEC = Covariance_PEC(EPR_alt_network_ansatz_PEC, gamma2)
	
	opt_score_mutual_info_EPR_PEC = []
	opt_score_sum_mutual_info_EPR_PEC = 0
	for i in range(trials):
		settings = EPR_alt_network_ansatz_PEC.rand_network_settings()
		opt_dict_mutual_info_EPR_PEC = qnet.gradient_descent(
		    cost_mutual_info_EPR_PEC,
		    settings,
		    step_size=stepsizeMutualInfo,
		    num_steps=steps,
		    sample_width=1,
		    verbose=False,
		)
		print(opt_dict_mutual_info_EPR_PEC["opt_score"])
		opt_score_mutual_info_EPR_PEC.append(np.array(opt_dict_mutual_info_EPR_PEC["opt_score"]))
		opt_score_sum_mutual_info_EPR_PEC += opt_dict_mutual_info_EPR_PEC["opt_score"]

	opt_score_aver_mutual_info_EPR_PEC.append(opt_score_sum_mutual_info_EPR_PEC * (1/trials)) 
	opt_score_mutual_info_EPR_PEC_standarderror.append(sem(opt_score_mutual_info_EPR_PEC))
	
	opt_score_covar_EPR_PEC = []
	opt_score_sum_covar_EPR_PEC = 0
	for i in range(trials):
		settings = EPR_alt_network_ansatz_PEC.rand_network_settings()
		opt_dict_covar_EPR_PEC = qnet.gradient_descent(
		    cost_covar_EPR_PEC,
		    settings,
		    step_size=stepsizeCovar,
		    num_steps=steps,
		    sample_width=1,
		    verbose=False,
		)
		print(opt_dict_covar_EPR_PEC["opt_score"])
		opt_score_covar_EPR_PEC.append(np.array(opt_dict_covar_EPR_PEC["opt_score"]))
		opt_score_sum_covar_EPR_PEC += opt_dict_covar_EPR_PEC["opt_score"]
		
	opt_score_aver_covar_EPR_PEC.append(opt_score_sum_covar_EPR_PEC * (1/trials))
	opt_score_covar_EPR_PEC_standarderror.append(sem(opt_score_covar_EPR_PEC))


#Optimization without PEC

ghz_prep_node = [
    qnet.PrepareNode(1, [0,1], qnet.ghz_state, 0)
]


def qubit_phase_damping_nodes_alt2(gamma1):
	qubit_phase_damping_nodes_alt2 = [
	    qnet.NoiseNode(wires=[0], ansatz_fn=lambda settings, wires: qml.PhaseDamping(gamma1, wires=wires)), 
	    qnet.NoiseNode(wires=[1], ansatz_fn=lambda settings, wires: qml.PhaseDamping(gamma1, wires=wires)), 
	]
	return qubit_phase_damping_nodes_alt2
	
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
meas_nodes = [
    qnet.MeasureNode(1, 2, [0], ansatz_fn=qml.ArbitraryUnitary, num_settings=3),
    qnet.MeasureNode(1, 2, [1], ansatz_fn=qml.ArbitraryUnitary, num_settings=3),
]

def UNCERTAINTY_cond_on_Q1(ansatz):
	@qml.qnode(qml.device(**ansatz.dev_kwargs))
	def circ(settings):
		ansatz.fn(settings)
		return qml.probs([0,1,2,3])
	def Uncertainty_cond_on_Q1(*settings):
		Probs = circ(settings)
		Marginal_Probs_Q1_basis1 = [(Probs[0]+Probs[1]+Probs[4]+Probs[5])/(Probs[0]+Probs[1]+Probs[4]+Probs[5]+Probs[8]+Probs[9]+Probs[12]+Probs[13]), (Probs[8]+Probs[9]+Probs[12]+Probs[13])/(Probs[0]+Probs[1]+Probs[4]+Probs[5]+Probs[8]+Probs[9]+Probs[12]+Probs[13])]
		Marginal_Probs_Q1_basis2 = [(Probs[2]+Probs[3]+Probs[6]+Probs[7])/(Probs[2]+Probs[3]+Probs[6]+Probs[7]+Probs[10]+Probs[11]+Probs[14]+Probs[15]), (Probs[10]+Probs[11]+Probs[14]+Probs[15])/(Probs[2]+Probs[3]+Probs[6]+Probs[7]+Probs[10]+Probs[11]+Probs[14]+Probs[15])]
		Joint_Probs_basis1 = [Probs[0]/(Probs[0] + Probs[4] + Probs[8] + Probs[12]), Probs[4]/(Probs[0] + Probs[4] + Probs[8] + Probs[12]), Probs[8]/(Probs[0] + Probs[4] + Probs[8] + Probs[12]), Probs[12]/(Probs[0] + Probs[4] + Probs[8] + Probs[12])]
		Joint_Probs_basis2 = [Probs[3]/(Probs[3] + Probs[7] + Probs[11] + Probs[15]), Probs[7]/(Probs[3] + Probs[7] + Probs[11] + Probs[15]), Probs[11]/(Probs[3] + Probs[7] + Probs[11] + Probs[15]), Probs[15]/(Probs[3] + Probs[7] + Probs[11] + Probs[15])]
		uncertainty_cond_on_Q1 = qnet.shannon_entropy(Joint_Probs_basis1) + qnet.shannon_entropy(Joint_Probs_basis2) - qnet.shannon_entropy(Marginal_Probs_Q1_basis1) - qnet.shannon_entropy(Marginal_Probs_Q1_basis2)
		return uncertainty_cond_on_Q1
	return Uncertainty_cond_on_Q1
def UNCERTAINTY_cond_on_Q2(ansatz):
	@qml.qnode(qml.device(**ansatz.dev_kwargs))
	def circ(settings):
		ansatz.fn(settings)
		return qml.probs([0,1,2,3])
	def Uncertainty_cond_on_Q2(*settings):
		Probs = circ(settings)
		Marginal_Probs_Q2_basis1 = [(Probs[0]+Probs[2]+Probs[8]+Probs[10])/(Probs[0]+Probs[2]+Probs[8]+Probs[10]+Probs[4]+Probs[6]+Probs[12]+Probs[14]), (Probs[4]+Probs[6]+Probs[12]+Probs[14])/(Probs[0]+Probs[2]+Probs[8]+Probs[10]+Probs[4]+Probs[6]+Probs[12]+Probs[14])]
		Marginal_Probs_Q2_basis2 = [(Probs[1]+Probs[3]+Probs[9]+Probs[11])/(Probs[1]+Probs[3]+Probs[9]+Probs[11]+Probs[5]+Probs[7]+Probs[13]+Probs[15]), (Probs[5]+Probs[7]+Probs[13]+Probs[15])/(Probs[1]+Probs[3]+Probs[9]+Probs[11]+Probs[5]+Probs[7]+Probs[13]+Probs[15])]
		Joint_Probs_basis1 = [Probs[0]/(Probs[0] + Probs[4] + Probs[8] + Probs[12]), Probs[4]/(Probs[0] + Probs[4] + Probs[8] + Probs[12]), Probs[8]/(Probs[0] + Probs[4] + Probs[8] + Probs[12]), Probs[12]/(Probs[0] + Probs[4] + Probs[8] + Probs[12])]
		Joint_Probs_basis2 = [Probs[3]/(Probs[3] + Probs[7] + Probs[11] + Probs[15]), Probs[7]/(Probs[3] + Probs[7] + Probs[11] + Probs[15]), Probs[11]/(Probs[3] + Probs[7] + Probs[11] + Probs[15]), Probs[15]/(Probs[3] + Probs[7] + Probs[11] + Probs[15])]
		uncertainty_cond_on_Q2 = qnet.shannon_entropy(Joint_Probs_basis1) + qnet.shannon_entropy(Joint_Probs_basis2) - qnet.shannon_entropy(Marginal_Probs_Q2_basis1) - qnet.shannon_entropy(Marginal_Probs_Q2_basis2)
		return uncertainty_cond_on_Q2
	return Uncertainty_cond_on_Q2

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

opt_score_aver_condQ1_EPR = []
opt_score_condQ1_EPR_standarderror = []
opt_score_aver_mutual_info_EPR = []
opt_score_mutual_info_EPR_standarderror = []
opt_score_aver_covar_EPR = []
opt_score_covar_EPR_standarderror = []

def Optimization_EPR(trials, steps, gamma1, gamma2, stepsizeUncern, stepsizeMutualInfo, stepsizeCovar):
	EPR_network_ansatz = qnet.NetworkAnsatz(
		ghz_prep_node,
		qubit_phase_damping_nodes_alt2(gamma1),
		proc_nodes,
		meas_nodes,
		dev_kwargs={
			"name": "default.mixed",
		}
	)
	EPR_alt_network_ansatz = qnet.NetworkAnsatz(
		ghz_prep_node,
		qubit_phase_damping_nodes_alt2(gamma1),
		meas_nodes,
		dev_kwargs={
			"name": "default.mixed",
		}
	)
	
	cost_condQ1_EPR = UNCERTAINTY_cond_on_Q1(EPR_network_ansatz)	

	opt_score_condQ1_EPR = []
	opt_score_sum_condQ1_EPR = 0

	for i in range(trials):
		settings = EPR_network_ansatz.rand_network_settings()#randomize the initial setting for every trial of optimization
		opt_dict_condQ1_EPR = qnet.gradient_descent(
			cost_condQ1_EPR,
			settings,
			step_size=stepsizeUncern,
			num_steps=steps,
			sample_width=1,
			verbose=False,
		)
		print(opt_dict_condQ1_EPR["opt_score"])
		opt_score_condQ1_EPR.append(-np.array(opt_dict_condQ1_EPR["opt_score"]))
		opt_score_sum_condQ1_EPR -= opt_dict_condQ1_EPR["opt_score"]
		
	opt_score_aver_condQ1_EPR.append(opt_score_sum_condQ1_EPR * (1/trials))
	opt_score_condQ1_EPR_standarderror.append(sem(opt_score_condQ1_EPR))
	
	
	#Optimization for the mutual information and covariance of EPR through gradient descent
	cost_mutual_info_EPR = Mutual_information(EPR_alt_network_ansatz)
	cost_covar_EPR = Covariance(EPR_alt_network_ansatz)
	
	opt_score_mutual_info_EPR = []
	opt_score_sum_mutual_info_EPR = 0
	for i in range(trials):
		settings = EPR_alt_network_ansatz.rand_network_settings()
		opt_dict_mutual_info_EPR = qnet.gradient_descent(
		    cost_mutual_info_EPR,
		    settings,
		    step_size=stepsizeMutualInfo,
		    num_steps=steps,
		    sample_width=1,
		    verbose=False,
		)
		print(opt_dict_mutual_info_EPR["opt_score"])
		opt_score_mutual_info_EPR.append(np.array(opt_dict_mutual_info_EPR["opt_score"]))
		opt_score_sum_mutual_info_EPR += opt_dict_mutual_info_EPR["opt_score"]

	opt_score_aver_mutual_info_EPR.append(opt_score_sum_mutual_info_EPR * (1/trials)) 
	opt_score_mutual_info_EPR_standarderror.append(sem(opt_score_mutual_info_EPR))
	
	opt_score_covar_EPR = []
	opt_score_sum_covar_EPR = 0
	for i in range(trials):
		settings = EPR_alt_network_ansatz.rand_network_settings()
		opt_dict_covar_EPR = qnet.gradient_descent(
		    cost_covar_EPR,
		    settings,
		    step_size=stepsizeCovar,
		    num_steps=steps,
		    sample_width=1,
		    verbose=False,
		)
		print(opt_dict_covar_EPR["opt_score"])
		opt_score_covar_EPR.append(np.array(opt_dict_covar_EPR["opt_score"]))
		opt_score_sum_covar_EPR += opt_dict_covar_EPR["opt_score"]
		
	opt_score_aver_covar_EPR.append(opt_score_sum_covar_EPR * (1/trials))
	opt_score_covar_EPR_standarderror.append(sem(opt_score_covar_EPR))

for gamma in np.arange(0,1,0.05):
	Optimization_EPR_PEC(50, 20, gamma, gamma, 0.1, 0.1, 0.1)
	Optimization_EPR(50, 30, gamma, gamma, 0.1, 0.1, 0.1)

a = Symbol("a", postive = True)
a_vals = np.linspace(0.01,0.95,95)
Uncern_dephasing_withoutVD = 0.5*(-1 - (a/2) * log2(a/4) - ((2-a)/2) * log2((2-a)/4))
Uncern_dephasing_withoutVD_values = [sp.N(Uncern_dephasing_withoutVD.subs(a, val)) for val in a_vals]

fig1 = plt.figure('Optimization for Noisy EPR with error band', figsize = (8,6)).add_subplot(111)
fig1.plot(np.arange(0,1,0.05), np.array(np.multiply(opt_score_aver_condQ1_EPR, 0.5)), "o:", color='dodgerblue', label="Uncertainty without PEC", linewidth=2)
fig1.plot(np.arange(0,1,0.05), 1 - np.array(opt_score_aver_mutual_info_EPR), "v:", color='orange', label="Mutual information without PEC", linewidth=2)
fig1.plot(np.arange(0,1,0.05), 0.5 - np.array(np.multiply(opt_score_aver_covar_EPR, 0.5)), "s:", color='forestgreen', label="Covariance without PEC", linewidth=2)
fig1.fill_between(np.arange(0,1,0.05), np.array(np.multiply(opt_score_aver_condQ1_EPR, 0.5)) + np.array(np.multiply(opt_score_condQ1_EPR_standarderror, 0.5)), np.array(np.multiply(opt_score_aver_condQ1_EPR, 0.5)) - np.array(np.multiply(opt_score_condQ1_EPR_standarderror, 0.5)), facecolor='dodgerblue', alpha=.5, linewidth=0)
fig1.fill_between(np.arange(0,1,0.05), 1 - (np.array(opt_score_aver_mutual_info_EPR) + np.array(opt_score_mutual_info_EPR_standarderror)), 1- (np.array(opt_score_aver_mutual_info_EPR) - np.array(opt_score_mutual_info_EPR_standarderror)), facecolor='orange', alpha=.5, linewidth=0)
fig1.fill_between(np.arange(0,1,0.05), 0.5 - (np.array(np.multiply(opt_score_aver_covar_EPR, 0.5)) + np.array(np.multiply(opt_score_covar_EPR_standarderror, 0.5))), 0.5 - (np.array(np.multiply(opt_score_aver_covar_EPR, 0.5)) - np.array(np.multiply(opt_score_covar_EPR_standarderror, 0.5))), facecolor='forestgreen', alpha=.5, linewidth=0)
fig1.plot(np.arange(0,1,0.05), np.array(np.multiply(opt_score_aver_condQ1_EPR_PEC, 0.5)), "o-", color='dodgerblue', label="Uncertainty with PEC", linewidth=2)
fig1.plot(np.arange(0,1,0.05), 1 - np.array(opt_score_aver_mutual_info_EPR_PEC), "v-", color='orange', label="Mutual information with PEC", linewidth=2)
fig1.plot(np.arange(0,1,0.05), 0.5 - np.array(np.multiply(opt_score_aver_covar_EPR_PEC, 0.5)), "s-", color='forestgreen', label="Covariance with PEC", linewidth=2)
fig1.fill_between(np.arange(0,1,0.05), np.array(np.multiply(opt_score_aver_condQ1_EPR_PEC, 0.5)) + np.array(np.multiply(opt_score_condQ1_EPR_PEC_standarderror, 0.5)), np.array(np.multiply(opt_score_aver_condQ1_EPR_PEC, 0.5)) - np.array(np.multiply(opt_score_condQ1_EPR_PEC_standarderror, 0.5)), facecolor='dodgerblue', alpha=.5, linewidth=0)
fig1.fill_between(np.arange(0,1,0.05), 1 - (np.array(opt_score_aver_mutual_info_EPR_PEC) + np.array(opt_score_mutual_info_EPR_PEC_standarderror)), 1- (np.array(opt_score_aver_mutual_info_EPR_PEC) - np.array(opt_score_mutual_info_EPR_PEC_standarderror)), facecolor='orange', alpha=.5, linewidth=0)
fig1.fill_between(np.arange(0,1,0.05), 0.5 - (np.array(np.multiply(opt_score_aver_covar_EPR_PEC, 0.5)) + np.array(np.multiply(opt_score_covar_EPR_PEC_standarderror, 0.5))), 0.5 - (np.array(np.multiply(opt_score_aver_covar_EPR_PEC, 0.5)) - np.array(np.multiply(opt_score_covar_EPR_PEC_standarderror, 0.5))), facecolor='forestgreen', alpha=.5, linewidth=0)
fig1.plot(np.linspace(0,0.95,96), np.insert(Uncern_dephasing_withoutVD_values,0,0), color='deepskyblue', label = 'Theoretical value for uncerntainty')
fig1.set_title('Optimization for Noisy EPR with and without PEC', size=20)
fig1.set_ylabel('Relative deviation from ideal value', size=16)
fig1.set_xlabel('Noise parameter', size=16)
fig1.grid()
fig1.legend(fontsize=10)

end_time = time.time()
print("Time cost:", end_time - start_time)

plt.show()
