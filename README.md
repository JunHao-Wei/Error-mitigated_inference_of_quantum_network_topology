# Error-mitigated Inference of Quantum Network Topology

This repository contains the code used in the paper **"[Error-mitigated inference of quantum network topology](http://dx.doi.org/10.1103/sx7r-1npt)"**.

The code is based on the Python packages [PennyLane](https://pennylane.ai/) and [qNetVO](https://chitambarlab.github.io/qNetVO/quantum_networks/index.html). It has been tested with **NumPy version 1.26.3**; using a later version may lead to compatibility issues. For best results, please ensure that NumPy is pinned to version 1.26.3.

Note that, since the initial measurement directions are randomly chosen, different random seeds may lead to different outcomes in the variational optimization.

## File Descriptions

- **`2qubit_comparison.py`**  
  Performs variational quantum optimization of uncertainty, mutual information, and covariance for an EPR state.  
  Reproduces **Figure 4** of the paper.

- **`3GHZ+EPR_noiseless_varying_shots.py`**  
  Performs variational optimization of correlation matrices for a network consisting of a three-qubit GHZ state and an EPR state.  
  Reproduces **Figures 5 and 6**.

- **`2qubit_PEC_comparison.py`**  
  Performs variational optimization of the correlations for an EPR state with and without **probabilistic error cancellation (PEC)**.  
  Reproduces **Figure 7**.

- **`2qubit_VD_EPR_depolarizing.py`**  
  Performs variational optimization of the correlations for a **depolarized EPR state**, with and without **virtual distillation (VD)**.  
  Reproduces the **left panel of Figure 8**.

- **`2qubit_VD_ReducedGHZ_depolarizing.py`**  
  Performs variational optimization of the correlations for the **reduced state of a GHZ state**, with and without **virtual distillation**.  
  Reproduces the **right panel of Figure 8**.

- **`W+nonmaxEntangledEPR_noisy_varying_shots.py`**  
  Performs variational optimization of correlation matrices for a network consisting of a **three-qubit W state and a generalized EPR state**.  
  Reproduces **Figures 10 and 11(a)**.

- **`nonmaxEntangledEPR_comparison.py`**  
  Performs variational optimization of **mutual information** for the **generalized EPR state**.  
  Reproduces **Figure 11(b)**.

- **`3GHZ+EPR_VD+SD.py`**  
  Simulates **qubitwise correlation matrices** for a noisy network with a three-qubit GHZ state and an EPR state, applying **virtual distillation (VD)** and **shadow distillation (SD)**.  
  Reproduces **Figure 12**.

- **`EPR3_SD.py`**  
  Simulates **node-wise correlation matrices** for a noisy network of three EPR states, using **shadow distillation**.  
  Reproduces **Figure 13**.
