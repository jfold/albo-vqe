# ALBO-VQE
This repository contains the code and examples for the corresponding paper entitled *Foldager, Jonathan, et al. "Algorithmic agency can improve quantum machine learning"* currently in review.

We combine active learning and bayesian optimization (ALBO) to perform algorithmic agency for the variational quantum eigensolver (VQE). The aim is to do hyperparameter transfer learning in variational quantum circuits.


## Install packages
Using Google Colab, the following two lines at the very top of the notebook should be sufficient for the algorithm to work:
```bash
!pip install tensorflow==2.3.1
!pip install tensorflow-quantum
!pip install cirq 
!pip install qsimcirq
!pip install scikit-optimize

```

## Usage 

```python
N_QUBITS    = 5
THETA       = {"L": {"type":"int",
                     "constraints":(1,8), 
                     "prior":"exp",
                     "prior_params":1.0}}
albo = ALBO(N_QUBITS=N_QUBITS,
          THETA=THETA,
          n_a_epochs=10, 
          n_b_epochs=3,
          n_c_epochs=100,
          search="AL+BO",
          seed=0)
albo.run()
```

## Example: Optimizing for the number of QAOA layers in VQE for Ising chain
Expected loss choosing optimal number of circuit layers:
![](https://github.com/jfold/albo-vqe/blob/main/loss.png  =250x250)
Iteration 4 of BO procedure for number of layers:
![](https://github.com/jfold/albo-vqe/blob/main/BO_layers.png  =250x250)
