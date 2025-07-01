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
ghz_prep_nodes_VD = [
    qnet.PrepareNode(1, [0,1], qnet.ghz_state, 0),
    qnet.PrepareNode(1, [2,3], qnet.ghz_state, 0),
    qnet.PrepareNode(1, [4,5], qnet.ghz_state, 0),
    qnet.PrepareNode(1, [6,7], qnet.ghz_state, 0),
]

def qubit_depolarizing_nodes_VD_alt2(gamma):
	qubit_phase_damping_nodes_alt2 = [
	    qnet.NoiseNode(wires=[0], ansatz_fn=lambda settings, wires: qml.DepolarizingChannel(gamma, wires=wires)), 
	    qnet.NoiseNode(wires=[1], ansatz_fn=lambda settings, wires: qml.DepolarizingChannel(gamma, wires=wires)),
	    qnet.NoiseNode(wires=[2], ansatz_fn=lambda settings, wires: qml.DepolarizingChannel(gamma, wires=wires)), 
	    qnet.NoiseNode(wires=[3], ansatz_fn=lambda settings, wires: qml.DepolarizingChannel(gamma, wires=wires)),
	    qnet.NoiseNode(wires=[4], ansatz_fn=lambda settings, wires: qml.DepolarizingChannel(gamma, wires=wires)), 
	    qnet.NoiseNode(wires=[5], ansatz_fn=lambda settings, wires: qml.DepolarizingChannel(gamma, wires=wires)),
	    qnet.NoiseNode(wires=[6], ansatz_fn=lambda settings, wires: qml.DepolarizingChannel(gamma, wires=wires)), 
	    qnet.NoiseNode(wires=[7], ansatz_fn=lambda settings, wires: qml.DepolarizingChannel(gamma, wires=wires)),
	]
	return qubit_phase_damping_nodes_alt2
	
meas_nodes_VD = [
    qnet.MeasureNode(1, 2, [0], ansatz_fn=None, num_settings=0),
    qnet.MeasureNode(1, 2, [1], ansatz_fn=None, num_settings=0),
    qnet.MeasureNode(1, 2, [2], ansatz_fn=None, num_settings=0),
    qnet.MeasureNode(1, 2, [3], ansatz_fn=None, num_settings=0),
    qnet.MeasureNode(1, 2, [4], ansatz_fn=None, num_settings=0),
    qnet.MeasureNode(1, 2, [5], ansatz_fn=None, num_settings=0),
    qnet.MeasureNode(1, 2, [6], ansatz_fn=None, num_settings=0),
    qnet.MeasureNode(1, 2, [7], ansatz_fn=None, num_settings=0),
    #qnet.MeasureNode(1, 2, [2], num_settings=0),
    #qnet.MeasureNode(1, 2, [3], num_settings=0),
]
	
#Define the post-processed probability of the VD process
def Probs_VDproc_uncern(Probs_pre):
	Probs = []
	IVD_basis1 = Probs_pre[0][0] - Probs_pre[0][4] - Probs_pre[0][8] + Probs_pre[0][12] + Probs_pre[0][16] + Probs_pre[0][20] - Probs_pre[0][24] - Probs_pre[0][28] + Probs_pre[0][32] - Probs_pre[0][36] + Probs_pre[0][40] - Probs_pre[0][44] + Probs_pre[0][48] + Probs_pre[0][52] + Probs_pre[0][56] + Probs_pre[0][60]
	IVD_basis2 = Probs_pre[0][3] - Probs_pre[0][7] - Probs_pre[0][11] + Probs_pre[0][15] + Probs_pre[0][19] + Probs_pre[0][23] - Probs_pre[0][27] - Probs_pre[0][31] + Probs_pre[0][35] - Probs_pre[0][39] + Probs_pre[0][43] - Probs_pre[0][47] + Probs_pre[0][51] + Probs_pre[0][55] + Probs_pre[0][59] + Probs_pre[0][63]
	Z1VD = (Probs_pre[0][0] - Probs_pre[0][4] + Probs_pre[0][16] + Probs_pre[0][20] - Probs_pre[0][40] + Probs_pre[0][44] - Probs_pre[0][56] - Probs_pre[0][60]) / IVD_basis1
	Z2VD = (Probs_pre[0][0] - Probs_pre[0][8] - Probs_pre[0][20] + Probs_pre[0][28] + Probs_pre[0][32] + Probs_pre[0][40] - Probs_pre[0][52] - Probs_pre[0][60]) / IVD_basis1
	X1VD = (Probs_pre[0][3] - Probs_pre[0][7] + Probs_pre[0][19] + Probs_pre[0][23] - Probs_pre[0][43] + Probs_pre[0][47] - Probs_pre[0][59] - Probs_pre[0][63]) / IVD_basis2
	X2VD = (Probs_pre[0][3] - Probs_pre[0][11] - Probs_pre[0][23] + Probs_pre[0][31] + Probs_pre[0][35] + Probs_pre[0][43] - Probs_pre[0][55] - Probs_pre[0][63]) / IVD_basis2
	Z1Z2VD = (Probs_pre[1][0] - Probs_pre[1][12] - Probs_pre[1][20] + Probs_pre[1][24] + Probs_pre[1][36] - Probs_pre[1][40] - Probs_pre[1][48] + Probs_pre[1][60]) / IVD_basis1
	X1X2VD = (Probs_pre[1][3] - Probs_pre[1][15] - Probs_pre[1][23] + Probs_pre[1][27] + Probs_pre[1][39] - Probs_pre[1][43] - Probs_pre[1][51] + Probs_pre[1][63]) / IVD_basis2
	Probs.append((1 + Z1Z2VD + Z1VD + Z2VD) / 4) #P00_basis1
	Probs.append((1 - Z1Z2VD + Z1VD - Z2VD) / 4) #P01_basis1
	Probs.append((1 - Z1Z2VD - Z1VD + Z2VD) / 4) #P10_basis1
	Probs.append((1 + Z1Z2VD - Z1VD - Z2VD) / 4) #P11_basis1
	Probs.append((1 + X1X2VD + X1VD + X2VD) / 4) #P00_basis2
	Probs.append((1 - X1X2VD + X1VD - X2VD) / 4) #P01_basis2
	Probs.append((1 - X1X2VD - X1VD + X2VD) / 4) #P10_basis2
	Probs.append((1 + X1X2VD - X1VD - X2VD) / 4) #P11_basis2
	return Probs

def UNCERTAINTY_cond_on_Q1_VD(ansatz, gamma):
	@qml.qnode(qml.device(**ansatz.dev_kwargs))
	def circ(settings):
		ansatz.fn(settings)
		return qml.probs([0,1,2,3,8,9]), qml.probs([4,5,6,7,8,9])
	def Uncertainty_cond_on_Q1_VD(*settings):
		Probs = Probs_VDproc_uncern(circ(settings))
		Marginal_Probs_Q1_basis1 = [(Probs[0]+Probs[1])/(Probs[0]+Probs[1]+Probs[2]+Probs[3]), (Probs[2]+Probs[3])/(Probs[0]+Probs[1]+Probs[2]+Probs[3])]
		Marginal_Probs_Q1_basis2 = [(Probs[4]+Probs[5])/(Probs[4]+Probs[5]+Probs[6]+Probs[7]), (Probs[6]+Probs[7])/(Probs[4]+Probs[5]+Probs[6]+Probs[7])]
		Joint_Probs_basis1 = [Probs[0]/(Probs[0]+Probs[1]+Probs[2]+Probs[3]), Probs[1]/(Probs[0]+Probs[1]+Probs[2]+Probs[3]), Probs[2]/(Probs[0]+Probs[1]+Probs[2]+Probs[3]), Probs[3]/(Probs[0]+Probs[1]+Probs[2]+Probs[3])]
		Joint_Probs_basis2 = [Probs[4]/(Probs[4]+Probs[5]+Probs[6]+Probs[7]), Probs[5]/(Probs[4]+Probs[5]+Probs[6]+Probs[7]), Probs[6]/(Probs[4]+Probs[5]+Probs[6]+Probs[7]), Probs[7]/(Probs[4]+Probs[5]+Probs[6]+Probs[7])]
		uncertainty_cond_on_Q1_VD = qnet.shannon_entropy(Joint_Probs_basis1) + qnet.shannon_entropy(Joint_Probs_basis2) - qnet.shannon_entropy(Marginal_Probs_Q1_basis1) - qnet.shannon_entropy(Marginal_Probs_Q1_basis2)
		return uncertainty_cond_on_Q1_VD
	return Uncertainty_cond_on_Q1_VD



H = [[1/math.sqrt(2), 1/math.sqrt(2)],[1/math.sqrt(2), -1/math.sqrt(2)]]
H4 = np.kron(H, np.kron(H, np.kron(H, H)))

B = [[1, 0, 0, 0], [0, 1/math.sqrt(2), -1/math.sqrt(2), 0], [0, 1/math.sqrt(2), 1/math.sqrt(2), 0], [0, 0, 0, 1]]
C = np.array([[0, 0, 0, 1], [0, 1j/math.sqrt(2), 1/math.sqrt(2),0], [0, -1j/math.sqrt(2), 1/math.sqrt(2), 0], [1,0,0,0]])


opt_score_aver_condQ1_EPR_VD = []
opt_score_condQ1_EPR_VD_standarderror = []
opt_score_condQ1_EPR_VD_min = []

def Optimization_EPR_VD(trials, steps, gamma, stepsizeUncern):
	def processing_ansatz_VD(settings, wires):
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
		qml.QubitUnitary(B, wires=[0,2])
		qml.QubitUnitary(B, wires=[1,3])
		qml.QubitUnitary(C, wires=[4,6])
		qml.QubitUnitary(C, wires=[5,7])
	proc_nodes_VD = [
		qnet.ProcessingNode(
			wires=[0,1,2,3,4,5,6,7,8,9],
			ansatz_fn=processing_ansatz_VD,
			num_settings = 6
		)
	]
	EPR_network_ansatz_VD = qnet.NetworkAnsatz(
		ghz_prep_nodes_VD,
		qubit_depolarizing_nodes_VD_alt2(gamma),
		proc_nodes_VD,
		meas_nodes_VD,
		dev_kwargs={
			"name": "default.mixed",
#			"shots": shots,
		}
	)
		
	
	cost_condQ1_EPR_VD = UNCERTAINTY_cond_on_Q1_VD(EPR_network_ansatz_VD, gamma)

	opt_score_condQ1_EPR_VD = []
	opt_score_sum_condQ1_EPR_VD = 0
	
	for i in range(trials):
		settings = EPR_network_ansatz_VD.rand_network_settings()#randomize the initial setting for every trial of optimization
		opt_dict_condQ1_EPR_VD = qnet.gradient_descent(
			cost_condQ1_EPR_VD,
			settings,
			step_size=stepsizeUncern,
			num_steps=steps,
			sample_width=1,
			verbose=True,
		)
		print(opt_dict_condQ1_EPR_VD["opt_score"])
		opt_score_condQ1_EPR_VD.append(-np.array(opt_dict_condQ1_EPR_VD["opt_score"]))
		opt_score_sum_condQ1_EPR_VD -= opt_dict_condQ1_EPR_VD["opt_score"]
		
	opt_score_aver_condQ1_EPR_VD.append(opt_score_sum_condQ1_EPR_VD * (1/trials))
	opt_score_condQ1_EPR_VD_standarderror.append(sem(opt_score_condQ1_EPR_VD))
	opt_score_condQ1_EPR_VD_min.append(np.array(opt_score_condQ1_EPR_VD).min())


#Optimization without VD

ghz_prep_node = [
    qnet.PrepareNode(1, [0,1], qnet.ghz_state, 0)
]

def qubit_depolarizing_nodes_withoutVD(gamma1, gamma2):
	qubit_depolarizing_nodes_withoutVD = [
	    qnet.NoiseNode(wires=[0], ansatz_fn=lambda settings, wires: qml.DepolarizingChannel(gamma1, wires=wires)), 
	    qnet.NoiseNode(wires=[1], ansatz_fn=lambda settings, wires: qml.DepolarizingChannel(gamma2, wires=wires)), 
	]
	return qubit_depolarizing_nodes_withoutVD
	
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
		Marginal_Probs_Q1_basis1 = [(Probs[0]+Probs[4])/(Probs[0]+Probs[4]+Probs[8]+Probs[12]), (Probs[8]+Probs[12])/(Probs[0]+Probs[4]+Probs[8]+Probs[12])]
		Marginal_Probs_Q1_basis2 = [(Probs[3]+Probs[7])/(Probs[3] + Probs[7] + Probs[11] + Probs[15]), (Probs[11]+Probs[15])/(Probs[3] + Probs[7] + Probs[11] + Probs[15])]
		Joint_Probs_basis1 = [Probs[0]/(Probs[0] + Probs[4] + Probs[8] + Probs[12]), Probs[4]/(Probs[0] + Probs[4] + Probs[8] + Probs[12]), Probs[8]/(Probs[0] + Probs[4] + Probs[8] + Probs[12]), Probs[12]/(Probs[0] + Probs[4] + Probs[8] + Probs[12])]
		Joint_Probs_basis2 = [Probs[3]/(Probs[3] + Probs[7] + Probs[11] + Probs[15]), Probs[7]/(Probs[3] + Probs[7] + Probs[11] + Probs[15]), Probs[11]/(Probs[3] + Probs[7] + Probs[11] + Probs[15]), Probs[15]/(Probs[3] + Probs[7] + Probs[11] + Probs[15])]
		uncertainty_cond_on_Q1 = qnet.shannon_entropy(Joint_Probs_basis1) + qnet.shannon_entropy(Joint_Probs_basis2) - qnet.shannon_entropy(Marginal_Probs_Q1_basis1) - qnet.shannon_entropy(Marginal_Probs_Q1_basis2)
		return uncertainty_cond_on_Q1
	return Uncertainty_cond_on_Q1


opt_score_aver_condQ1_EPR = []
opt_score_condQ1_EPR_standarderror = []
opt_score_condQ1_EPR_min = []

def Optimization_EPR(trials, steps, gamma1, gamma2, stepsizeUncern):
	EPR_network_ansatz = qnet.NetworkAnsatz(
		ghz_prep_node,
		qubit_depolarizing_nodes_withoutVD(gamma1, gamma2),
		proc_nodes,
		meas_nodes,
		dev_kwargs={
			"name": "default.mixed",
#			"shots": shots,
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
	opt_score_condQ1_EPR_min.append(np.array(opt_score_condQ1_EPR).min())

a = Symbol("a", postive = True)
a_vals = np.linspace(0.01,1,98)
Z1Z2VD = 1 - ( 2 * (4*a/3)**2 * ((4*a/3)-2)**2 )/(6*(4*a/3)**4 - 24*(4*a/3)**3 + 36*(4*a/3)**2 -24*(4*a/3) + 8)
X1X2VD = 4*( (4*a/3)**4 - 4*(4*a/3)**3 + 7*(4*a/3)**2 - 6*(4*a/3) +2)/(6*(4*a/3)**4 - 24*(4*a/3)**3 + 36*(4*a/3)**2 -24*(4*a/3) + 8)
Uncern_depolarizing_withoutVD = -2 - (2- 2*(4*a/3) + (4*a/3)**2) * log2((2- 2*(4*a/3) + (4*a/3)**2)/4) - (2*(4*a/3)- (4*a/3)**2) * log2((2*(4*a/3)- (4*a/3)**2)/4)
Uncern_depolarizing_withVD = -2 - ((1+Z1Z2VD)/2) * log2((1+Z1Z2VD)/4) - ((1-Z1Z2VD)/2) * log2((1-Z1Z2VD)/4) - ((1+X1X2VD)/2) * log2((1+X1X2VD)/4) - ((1-X1X2VD)/2) * log2((1-X1X2VD)/4)
coherent_info_depolarizing = - ((3* (4*a/3)**2 - 6*(4*a/3) + 4) / 4) * log2((3* (4*a/3)**2 - 6*(4*a/3) + 4) / 4) - 3 * ((2*(4*a/3)- (4*a/3)**2)/4)*log2((2*(4*a/3)- (4*a/3)**2)/4)
Uncern_depolarizing_withoutVD_values = [sp.N(Uncern_depolarizing_withoutVD.subs(a, val)) for val in a_vals]
Uncern_depolarizing_withVD_values = [sp.N(Uncern_depolarizing_withVD.subs(a, val)) for val in a_vals]
coherent_info_depolarizing_values = [sp.N(coherent_info_depolarizing.subs(a, val)) for val in a_vals]

for gamma in np.arange(0,0.76,0.05):
	Optimization_EPR_VD(20, 20, gamma, 0.05+gamma)
	Optimization_EPR(20, 20, gamma, gamma, 0.05+gamma*0.5)


fig1 = plt.figure('Optimization for Noisy EPR with error band', figsize = (8,7)).add_subplot(111)
fig1.plot(np.arange(0,0.76,0.05), np.array(opt_score_aver_condQ1_EPR), "o:", label="Uncertainty without VD", linewidth=2)
fig1.fill_between(np.arange(0,0.76,0.05), np.array(opt_score_aver_condQ1_EPR) + np.array(opt_score_condQ1_EPR_standarderror), np.array(opt_score_aver_condQ1_EPR) - np.array(opt_score_condQ1_EPR_standarderror), alpha=.5, linewidth=1)
fig1.plot(np.arange(0,0.76,0.05), np.array(opt_score_aver_condQ1_EPR_VD), "o:", label="Uncertainty with VD", linewidth=2)
fig1.fill_between(np.arange(0,0.76,0.05), np.array(opt_score_aver_condQ1_EPR_VD) + np.array(opt_score_condQ1_EPR_VD_standarderror), np.array(opt_score_aver_condQ1_EPR_VD) - np.array(opt_score_condQ1_EPR_VD_standarderror), alpha=.5, linewidth=1)
# ~ fig1.scatter(np.arange(0,0.76,0.05), np.array(opt_score_condQ1_EPR_min), s=24, c='#1F77B4', label='Minimum uncertainty without VD', marker='v')
# ~ fig1.scatter(np.arange(0,0.76,0.05), np.array(opt_score_condQ1_EPR_VD_min), s=24, c='#FF7F0E', label='Minimum uncertainty with VD', marker='v')
fig1.plot(np.linspace(0,1,99), np.insert(Uncern_depolarizing_withoutVD_values,0,0), color='deepskyblue', label = 'Theoretical value without VD')
fig1.plot(np.linspace(0,1,99), np.insert(Uncern_depolarizing_withVD_values,0,0), color='orange', label = 'Theoretical value with VD')
fig1.plot(np.linspace(0,1,99), np.insert(coherent_info_depolarizing_values,0,0), color='forestgreen', label = '1+H(Qi|Qj)')
fig1.set_title('Optimization for Noisy EPR with and without VD', size=26)
fig1.set_ylabel('Value of uncertainty', size=16)
fig1.set_xlabel('Noise parameter', size=16)
fig1.grid()
fig1.legend(fontsize=20)
plt.xlim(0,0.75)
plt.ylim(bottom=0)

fig2 = plt.figure('Optimization for Noisy EPR without error band', figsize = (8,7)).add_subplot(111)
fig2.plot(np.arange(0,0.76,0.05), np.array(opt_score_aver_condQ1_EPR), "o:", label="Uncertainty without VD", linewidth=2)
fig2.plot(np.arange(0,0.76,0.05), np.array(opt_score_aver_condQ1_EPR_VD), "o:", label="Uncertainty with VD", linewidth=2)
# ~ fig2.scatter(np.arange(0,0.76,0.05), np.array(opt_score_condQ1_EPR_min), s=24, label='Minimum uncertainty without VD', marker='v')
# ~ fig2.scatter(np.arange(0,0.76,0.05), np.array(opt_score_condQ1_EPR_VD_min), s=24, label='Minimum uncertainty with VD', marker='v')
fig2.plot(np.linspace(0,1,99), np.insert(Uncern_depolarizing_withoutVD_values,0,0), color='deepskyblue', label = 'Theoretical value without VD')
fig2.plot(np.linspace(0,1,99), np.insert(Uncern_depolarizing_withVD_values,0,0), color='orange', label = 'Theoretical value with VD')
fig2.plot(np.linspace(0,1,99), np.insert(coherent_info_depolarizing_values,0,0), color='forestgreen', label = '1+H(Qi|Qj)')
fig2.set_title('Optimization for Noisy EPR with and without VD', size=26)
fig2.set_ylabel('Value of uncertainty', size=16)
fig2.set_xlabel('Noise parameter', size=16)
fig2.grid()
fig2.legend(fontsize=20)
plt.xlim(0,0.75)
plt.ylim(bottom=0)


end_time = time.time()
print("Time cost:", end_time - start_time)

plt.show()
