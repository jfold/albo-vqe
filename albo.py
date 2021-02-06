%matplotlib inline
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
from cirq import Simulator
import qsimcirq
from itertools import product
import os
import pickle
from scipy.stats import norm
import sympy
import sys
from skopt.learning import GaussianProcessRegressor as GP
from sklearn.gaussian_process.kernels import *
from sklearn.base import clone
import itertools
import matplotlib.pyplot as plt
#import warnings
#warnings.filterwarnings("ignore")
import time
from tqdm.notebook import tqdm
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['font.size'] = 16

class p:
  # =============================================================================
  # Default setting
  # =============================================================================
  # INTEGERS
  N                       = 3
  seed                    = 0
  ints                    = ["N","seed"]
  # FLOATS
  Lprior 									= 0.1
  floats                  = ["Lprior"]
  # BOOLEANS
  two_D_ex                = True
  bools                   = ["two_D_ex"]
  # STRINGS
  H_J_terms               = 'ZZ'
  H_b_terms               = 'Z'
  savepth 								= "results/"
  strings                 = ["H_J_terms","H_b_terms","savepth"]

class QC(object):
  def __init__(self, 
                H,
                N_system = 1, 
                N_ancilla = 0,
                L = 1, 
                expectation_method = "analytical", 
                gammas=np.array([]),
                gamma_lr = 0.02,
                gamma_std = 1,
                n_expectation_samples = 50,
                seed = 0,
                ansatz="qaoa-r",
                epsilon=1e-1,
                use_qsim=True,
                savepth="results/",
                H_fd_order=0,
                verbøgse=False):
    """
    Quantum Circuit class

    Parameters
    ----------
    H  : Hamiltonian (PauliSum,cirq)

    N_SYSTEM  : number of system qubits (int)

    N_SYSTEM  : number of ancilla qubits (int)
    
    L        : number of circuit layers (int)

    expectation_method : {"analytical","sampled"} (string)
    
    gamma_lr  : learning rate of gamma gradient descent (float)
    
    seed      : Random number generator seed (int)

    """

    self.H                      = H
    self.seed                   = seed
    self.verbøgse               = verbøgse
    self.N_system               = N_system
    self.N_ancilla              = N_ancilla
    self.N_total                = N_system+N_ancilla
    self.d                      = 2**(self.N_total) 
    self.L                      = L
    self.ansatz                 = ansatz
    self.expectation_method     = expectation_method
    self.n_expectation_samples  = n_expectation_samples
    self.qubits                 = cirq.GridQubit.rect(self.N_total, 1)
    self.system_qubits          = cirq.GridQubit.rect(self.N_system, 1)
    self.q_map                  = {q:i for i,q in enumerate(self.system_qubits)}
    self.gamma_lr               = gamma_lr
    self.savepth                = savepth
    self.use_qsim               = use_qsim
    self.epsilon                = epsilon
    self.grad_diff              = []
    np.random.seed(self.seed)

    # Make backends
    self.simulator              = qsimcirq.QSimSimulator() if use_qsim else Simulator()
    self.construct_circuit()

    # Make expectation layer
    if self.expectation_method == "analytical":
      self.calc_pure_expectation = tfq.layers.Expectation(differentiator=tfq.differentiators.Adjoint(),backend=self.simulator) 
    if self.expectation_method == "sampled":
      self.calc_pure_expectation = tfq.layers.SampledExpectation(differentiator=tfq.differentiators.ParameterShift(),backend=self.simulator)
    
    # Initialize parameters
    self.gammas                 = np.random.randn((self.n_gammas))*gamma_std if gammas.size == 0 else gammas
    self.gammas_tf              = tf.Variable(tf.convert_to_tensor([self.gammas],dtype=tf.float32))

    # Metric
    self.energy                 = [self.get_H_qsim_expectation() if self.use_qsim else self.get_H_tfq_expectation().numpy()[0][0]]

    # Parameters
    self.best_energy            = np.inf
    self.best_gammas            = self.gammas.copy()
    self.optimizer              = tf.optimizers.Adam(self.gamma_lr)

    self.settings               = "Ns-"+str(self.N_system)+"Nt-"+str(self.N_total)+"--L-"+str(self.L)+"--seed-"+str(self.seed)

    # Finite difference coefficients taken from: https://en.wikipedia.org/wiki/Finite_difference_coefficient
    self.fd_coeffs              = [[1/2],[2/3,-1/12],[3/4,-3/20,1/60],[4/5,-1/5,4/105,-1/280]]
    self.H_fd_order             = H_fd_order

  def plot_history(self,save=False,figsize=(8,6)):
    f = plt.figure(figsize=figsize)

    plt.subplot(1,1,1)
    plt.plot(self.energy,'--.',label="System")
    plt.xlabel("Epochs");
    plt.legend()
    plt.ylabel(EXPECTATION_STR);

    if save:
      f.savefig(self.savepth+"history--"+self.settings+".pdf")

    plt.show()

  def construct_circuit(self,incl_rz=False):
    self.symbols   = []
    self.circuit   = cirq.Circuit()
    if self.ansatz == "entangl":
      for l in range(self.L):
        block = []
        for n in range(self.N_total):
          theta_        = 'theta_' + str(n) + "_0_"+ str(l)
          theta         = sympy.Symbol(theta_)
          block.append([cirq.rz(theta)(self.qubits[n])])
          self.symbols.append(theta_)
        self.circuit.append(cirq.Moment(block)) 

        block = []
        for n in range(self.N_total):
          theta_        = 'theta_' + str(n) + "_1_"+ str(l)
          theta         = sympy.Symbol(theta_)
          block.append([cirq.rx(theta)(self.qubits[n])])
          self.symbols.append(theta_)
        self.circuit.append(cirq.Moment(block)) 

        block = []
        for n in range(self.N_total):
          theta_        = 'theta_' + str(n) + "_2_"+ str(l)
          theta         = sympy.Symbol(theta_)
          block.append([cirq.rz(theta)(self.qubits[n])])
          self.symbols.append(theta_)
        self.circuit.append(cirq.Moment(block)) 

        self.circuit  += cirq.Circuit(cirq.CNOT(self.qubits[n],self.qubits[n+1]) for n in range(self.N_total-1))
        self.circuit  += cirq.Circuit(cirq.CNOT(self.qubits[-1],self.qubits[0]))

    elif self.ansatz == "qaoa-r":

      self.circuit    = cirq.Circuit(cirq.H.on_each(self.qubits))

      for l in range(self.L):
        Rzz         = sympy.Symbol('gamma_Rzz' + str(l+1))
        Rz          = sympy.Symbol('gamma_Rz' + str(l+1))
        Rx          = sympy.Symbol('gamma_Rx' + str(l+1))

        self.circuit  += cirq.Circuit((cirq.ZZPowGate(exponent=Rzz).on(self.qubits[i], self.qubits[i+1]) for i in range(self.N_total - 1)))
        if self.N_total > 2:
          self.circuit  += cirq.Circuit(cirq.ZZPowGate(exponent=Rzz).on(self.qubits[-1], self.qubits[0])) 

        if incl_rz:
          self.circuit  += cirq.Circuit((cirq.rz(Rz)(self.qubits[i]) for i in range(self.N_total)))
        self.circuit  += cirq.Circuit((cirq.rx(Rx)(self.qubits[i]) for i in range(self.N_total)))

        self.symbols.append('gamma_Rzz' + str(l+1))
        if incl_rz:
          self.symbols.append('gamma_Rz' + str(l+1))
        self.symbols.append('gamma_Rx' + str(l+1))
        
    elif self.ansatz == "qaoa-f": 
      self.circuit    = cirq.Circuit(cirq.H.on_each(self.qubits))

      for l in range(self.L):
        for n in range(self.N_total-1):
          sym           = 'gamma' + str(l+1) + '_' + str(n+1)
          gamma         = sympy.Symbol(sym)
          self.symbols.append(sym)
          self.circuit  += cirq.Circuit(cirq.ZZPowGate(exponent=gamma).on(self.qubits[n],self.qubits[n+1]))

        if self.N_total > 2:
          sym           = 'gamma' + str(l+1) + '_' + str(self.N_total)
          gamma         = sympy.Symbol(sym)
          self.symbols.append(sym)
          self.circuit  += cirq.Circuit(cirq.ZZPowGate(exponent=gamma).on(self.qubits[-1], self.qubits[0])) 

        gates = []
        for n in range(self.N_total):
          sym           = 'eta' + str(l+1) + '_' + str(n+1)
          eta           = sympy.Symbol(sym)
          gates.append(cirq.Circuit(cirq.rz(eta)(self.qubits[n])))
          self.symbols.append(sym)
        self.circuit.append(cirq.Moment(gates))

        gates = []
        for n in range(self.N_total):
          sym           = 'beta' + str(l+1) + '_' + str(n+1)
          beta          = sympy.Symbol(sym)
          gates.append(cirq.Circuit(cirq.rx(beta)(self.qubits[n])))
          self.symbols.append(sym)
        self.circuit.append(cirq.Moment(gates))

    self.symbols  = list(self.symbols)
    self.n_gammas = len(self.symbols)

  def forward(self, gammas = np.array([])):
    if gammas.size == 0:
      gammas    = self.gammas.copy()
    psi_gamma   = self.simulator.simulate(self.circuit,cirq.ParamResolver({symbol: gammas[i] for i,symbol in enumerate(self.symbols)})).final_state_vector
    return psi_gamma

  def get_H_qsim_expectation(self,gammas=np.array([])):
    if gammas.size == 0:
      gammas    = self.gammas.copy()
    psi         = self.forward(gammas=gammas)
    expectation = self.H.expectation_from_state_vector(psi,self.q_map,check_preconditions=False).real
    return expectation

  def get_H_tfq_expectation(self):

    if "analytical" in self.expectation_method:
        expectation = self.calc_pure_expectation(self.circuit,
                                            operators=self.H,
                                            symbol_names=self.symbols,
                                            symbol_values=self.gammas_tf)
    elif "sample" in self.expectation_method:
        expectation = self.calc_pure_expectation(self.circuit,
                                            operators=self.H,
                                            symbol_names=self.symbols,
                                            symbol_values=self.gammas_tf,
                                            repetitions=self.n_expectation_samples)
    return expectation

  def get_fd_ith_gradient(self,i):
    gradient      = 0
    gamma         = self.gammas[i]
    for k,c in enumerate(self.fd_coeffs[self.H_fd_order]):
      gammas      = self.gammas.copy()
      gammas[i]   = (k+1)*self.epsilon + gamma
      forward     = self.get_H_qsim_expectation(gammas=gammas)

      gammas      = self.gammas.copy()
      gammas[i]   = -(k+1)*self.epsilon + gamma
      backward    = self.get_H_qsim_expectation(gammas=gammas)
      
      gradient    += c*(forward-backward)/self.epsilon
    
    return (i,gradient)

  def fd_gradient(self):
    gradients       = []
    for g_i,gamma in enumerate(self.gammas):
      gradient = self.get_fd_ith_gradient(g_i)[1]
      gradients.append(gradient)
    gradients       = tf.convert_to_tensor([gradients],dtype=tf.float32)
    return gradients

  @tf.function
  def tfq_gradient(self):
    with tf.GradientTape() as g:
      g.watch(self.gammas_tf) 
      H_expectation = self.get_H_tfq_expectation()
    gradients = g.gradient(H_expectation, self.gammas_tf)
    return gradients

  def get_gamma_gradients(self):
    if self.use_qsim:
      gradients = self.fd_gradient()
    else:
      gradients = self.tfq_gradient()

    return gradients

  def train(self,gamma_lr=-1):
    # Calculate gamma gradients 
    gradients    = self.get_gamma_gradients()
    # Update ansatz gamma
    self.optimizer.apply_gradients(zip([gradients], [self.gammas_tf]))
    self.gammas = np.squeeze(self.gammas_tf.numpy().copy().astype(np.float64)) # infinitely important to use .astype(np.float64)
    # Save metrics
    self.energy.append(self.get_H_qsim_expectation() if self.use_qsim else self.get_H_tfq_expectation().numpy()[0][0])
    # Save best model so far
    if self.energy[-1] < self.best_energy:
      self.best_energy        = self.energy[-1]
      self.best_gammas        = self.gammas.copy()

class ALBO(object):
  def __init__(self, 
              N_QUBITS = 3, 
              Jb={"J":(0,2),"b":(0,2)}, 
              THETA={"L":{"type":"int","constraints":(1,8),"prior":"exp","prior_params":1}}, 
              two_D_ex=True,
              H_J_terms=["ZZ"],
              H_b_terms=["Z"],
              ansatz="qaoa-r", 
              eps_prog=0.01,
              n_a_test=10000,
              n_b_test=3,
              n_a_epochs=10,
              gp_length_scale=0.5,
              n_b_epochs=5,
              n_c_epochs=20,
              c_restarts=3,
              grid_space=10,
              search="AL+BO", # any combination of "RS","GS","AL","AL"
              earlier_experiments={}, 
              seed=0,
              verbøgse=False,
              test_run=False,
              plot_gp=False,
              plot_Jb=False,
              savepth="drive/MyDrive/PhD/ALBO-VQE/results/"):
    """
    Actively Learned and Bayesian Optimized (ALBO) Quantum Circuits

    Parameters
    ----------
    N_SYSTEM  : number of qubits (int)

    Jb        : Hamiltonian coefficients with constraints (dict)
      Areas of coefficients to be explored with active learning
    
    THETA     : Hyperparameters with constraints (dict of dicts)
      Areas of coefficients to be explored with bayesian optimization
    
    two_D_ex  : Explore only two dimensional problems (boolean)
      Use only one interaction (J) and one bias (b) term to be explored.
      Allows for 2D image plot for visualization.

    eps_prog  : early stopping threshold
      Incrementing one unit in hyperparameter, only eps_prog is allowed for
      the energy expectation to decrease before bayesian optimization is
      terminated
    
    seed      : Random number generator seed (int)

    """
    self.seed           = seed
    np.random.seed(self.seed)
    self.N_QUBITS       = N_QUBITS
    self.Jb             = Jb
    self.THETA          = THETA
    self.DIMS_THETA     = len(self.THETA)
    self.two_D_ex       = two_D_ex
    self.plot_gp        = plot_gp
    self.plot_Jb        = plot_Jb
    self.gp_length_scale= gp_length_scale
    self.c_restarts     = c_restarts

    if self.two_D_ex:
      self.H_J_terms    = H_J_terms
      self.H_b_terms    = H_b_terms
      self.x_1_term     = H_J_terms[0]
      self.x_2_term     = H_b_terms[0]
      self.DIMS_J       = len(H_J_terms)
      self.DIMS_b       = len(H_b_terms)
      self.DIMS_X       = 2
    else:
      self.H_J_terms    = H_J_terms
      self.H_b_terms    = H_b_terms
      self.DIMS_J       = len(self.H_J_terms)*self.N_QUBITS
      self.DIMS_b       = len(self.H_b_terms)*self.N_QUBITS
      self.DIMS_X       = self.DIMS_J + self.DIMS_b

    self.make_results_column()
    self.DIMS           = self.DIMS_X + self.DIMS_THETA
    self.savepth        = savepth
    self.p_d            = 100 # discretize priors if continouous
    self.y_tol          = 1e-2
    self.n_a_test       = n_a_test
    self.grid_space     = grid_space
    self.verbøgse       = verbøgse
    self.ansatz         = ansatz
    self.Jb_THETA       = np.full((0,self.DIMS),np.nan)  # hyperparameters + hamiltonian ops
    self.psi_H_psi      = np.full((0,1),np.nan)  # hyperparameters + hamiltonian ops
    self.E_0            = []  # min energy
    self.E_1            = []  # first exicited energy
    self.construct_THETA_priors()
    self.Kernel         = 1 * RBF(length_scale=self.gp_length_scale)
    self.gp             = GP(kernel=self.Kernel,noise=0.01,random_state=self.seed)
    self.bo_xi          = 0
    self.al_xi          = 0
    self.epsilon        = 1e-6
    self.results        = {}
    self.train_results  = pd.DataFrame(columns=self.res_columns)
    self.test_results   = pd.DataFrame(columns=self.res_columns)
    self.continue_exp   = isinstance(earlier_experiments, pd.DataFrame) and not earlier_experiments.empty
    self.search         = search
    self.Jb_search,self.THETA_search = search.split("+")
    self.n_a_epochs     = n_a_epochs
    self.n_b_epochs     = n_b_epochs
    self.n_c_epochs     = n_c_epochs
    self.settings       = self.search+"--N-"+str(self.N_QUBITS)+"--Na-"+str(self.n_a_epochs)+"--Nb-"+str(self.n_b_epochs)+"--Nc-"+str(self.n_c_epochs)+"--2D-"+str(self.two_D_ex)+"--H-J-"+'-'.join(H_J_terms)+"--H-b-"+'-'.join(H_b_terms)+"--pT-"+self.pT+"--seed-"+str(seed)
    self.make_test_set(n=10)
    
    print("SETTINGS:",self.settings)
    if self.verbøgse:
      print(self.res_columns)

    if self.continue_exp:
      data            = earlier_experiments
      columns         = data.columns
      self.results    = data
      self.E_0        = data["E0"].values.tolist()
      self.E_1        = data["E1"].values.tolist()
      self.psi_H_psi  = data["H"].values
      if self.two_D_ex:
        J_terms       = np.unique([c for c in columns if "-" in c and len(c.split("-")[0]) == 2])
        b_terms       = np.unique([c for c in columns if "-" in c and len(c.split("-")[0]) == 1])
        T_terms       = np.unique([c for c in columns if "L" in c])
        self.Jb_THETA = self.results[[J_terms[0],b_terms[0],T_terms[0]]].values
      else:
        self.Jb_THETA = self.results.loc[:, 0:(np.where(columns=="H")-1)].values
      self.gp.fit(self.Jb_THETA,self.psi_H_psi)

      if self.verbøgse:
        print("Loaded data:")
        print(self.results)

    if test_run:
      self.run(n_a_epochs=1,n_b_epochs=1,n_c_epochs=10,save_res=False)

  def make_results_column(self):
    self.res_columns    = []
    for h_J in self.H_J_terms:
      for q in range(1,self.N_QUBITS):
        self.res_columns.append(h_J+"-"+str(q)+"-"+str(q+1))
      if self.N_QUBITS > 2:
        self.res_columns.append(h_J+"-"+str(self.N_QUBITS)+"-1")
    
    for h_b in self.H_b_terms:
      for q in range(1,self.N_QUBITS):
        self.res_columns.append(h_b+"-"+str(q)+"-"+str(q+1))
      if self.N_QUBITS > 2:
        self.res_columns.append(h_b+"-"+str(self.N_QUBITS)+"-1")
    
    self.X_terms = [self.x_1_term,self.x_2_term] if self.two_D_ex else self.res_columns.copy()
    self.res_columns.extend(self.THETA.keys())
    self.res_columns.extend(['H','E0','E1'])

  def get_Hamiltonian_symbols(self,N_QUBITS=0):
    if N_QUBITS == 0:
      N_QUBITS    = self.N_QUBITS
    qubits        = cirq.GridQubit.rect(N_QUBITS, 1)
    H_terms       = ["XX","YY","ZZ","X","Y","Z"]
    cirq_objs     = {H_term: [] for H_term in H_terms}
    for H_term in H_terms:
      gate        = getattr(cirq, H_term)
      pauli       = getattr(cirq, H_term[0])
      if len(H_term) == 2:
        for n in range(N_QUBITS-1):
          cirq_objs[H_term].append(pauli(qubits[n])*pauli(qubits[n+1]))
        if N_QUBITS > 2:
          cirq_objs[H_term].append(pauli(qubits[-1])*pauli(qubits[0]))
      if len(H_term) == 1:
        for n in range(N_QUBITS):
          cirq_objs[H_term].append(gate(qubits[n]))

    return cirq_objs

  def get_H_obj(self,X_Jb,N_QUBITS=0):
    if N_QUBITS == 0:
      N_QUBITS= self.N_QUBITS
    cirq_objs = self.get_Hamiltonian_symbols(N_QUBITS=N_QUBITS)
    H_obj     = cirq.PauliSum()
    for i,H_term in enumerate(X_Jb):
      for n in range(len(X_Jb[H_term])):
        H_obj -= float(X_Jb[H_term][n])*cirq_objs[H_term][n]
    return H_obj

  def get_Hamiltonian(self,X_Jb):
    H_obj   = self.get_H_obj(X_Jb)
    if self.N_QUBITS <= 11:
      H_mat = H_obj.matrix()
      eigs  = np.sort(np.linalg.eigvals(H_mat))
      E_0   = eigs[0].real
      E_1   = eigs[1].real
    elif self.two_D_ex:
      X_Jb_         = {k: X_Jb[k][:3] for k in X_Jb.keys()}
      H_obj_        = self.get_H_obj(X_Jb_)
      H_mat_3       = H_obj_.matrix()
      eigs_3        = np.sort(np.linalg.eigvals(H_mat_3))
      E_0_3         = eigs_3[0].real
      E_1_3         = eigs_3[1].real

      X_Jb_         = {k: X_Jb[k][:4] for k in X_Jb.keys()}
      H_obj_        = self.get_H_obj(X_Jb_)
      H_mat_4       = H_obj_.matrix()
      eigs_4        = np.sort(np.linalg.eigvals(H_mat_4))
      E_0_4         = eigs_4[0].real
      E_1_4         = eigs_4[1].real

      # Linear extrapolation
      E_0           = E_0_3 + (E_0_4-E_0_3)*(self.N_QUBITS-3)
      E_1           = E_1_3 + (E_1_4-E_1_3)*(self.N_QUBITS-3)
    else:
      E_0           = 0
      E_1           = 0

    return H_obj,E_0,E_1

  def bo_ei(self,y_mean,y_std,xi):
    f_acq       = np.zeros_like(y_mean)
    mask        = y_std > 0
    improve     = np.min(y_mean) - xi - y_mean[mask]
    scaled      = improve / y_std[mask]
    cdf         = norm.cdf(scaled)
    pdf         = norm.pdf(scaled)
    exploit     = improve * cdf
    explore     = y_std[mask] * pdf
    f_acq[mask] = exploit + explore
    return f_acq

  def bo_pi(self,y_mean,y_std,xi):
    f_acq       = np.zeros_like(mu)
    mask        = std > 0
    improve     = y_opt - xi - mu[mask]
    scaled      = improve / std[mask]
    f_acq[mask] = norm.cdf(scaled)
    return f_acq

  def bo_acquisition(self,y_mean,y_std,use_prior=True,use_sample=True,return_acq=False):
    f_acq       = self.bo_ei(y_mean,y_std,self.bo_xi)
    if use_prior:
      p_acq     = f_acq*self.p_THETA/np.sum(f_acq*self.p_THETA)
      if use_sample:
        THETA_new = np.random.choice(self.THETA_i,p=p_acq)
      else:
        THETA_new = self.THETA_i[np.argmax(p_acq)]
    elif use_sample:
      p_acq       = f_acq/np.sum(f_acq)
      THETA_new   = np.random.choice(self.THETA_i,p=p_acq)
    else:
      THETA_new   = self.THETA_i[np.argmax(f_acq)]
    if return_acq:
      return f_acq,p_acq

    return THETA_new

  def al_mu(self,X,y_std):
    """Maximum uncertainty: the next point x is found where the uncertainty is the largest
    Returns: index (int) of position in y_std."""
    idx_new = np.argmax(y_std)
    return idx_new

  def al_uc(self,X,y_std):
    """Uncertainty sampling: each point x in X has the probability of p(x) = sigma(x)/sum_x(sigma(x)).
    Returns: index (int) of position in y_std."""
    p_std   = y_std/np.sum(y_std)
    idxs    = list(range(X.shape[0]))
    idx_new = np.random.choice(idxs,p=p_std)
    return idx_new

  def al_muc(self,X,y_std):
    """Maximal uncertainty change
    INPUTS
    X       : query candidates
    Returns: index (int) of position in y_std."""
    accum_unc   = np.sum(y_std)
    unc_diff    = []
    for i,x in enumerate(X):
      gp_       = clone(self.gp)
      y         = gp_.predict(x)
      gp_.fit(x,y)
      _,y_std_  = gp_.predict(X,return_std=True)
      unc_diff.append(np.abs(accum_unc-np.sum(y_std_)))

    idx_new = np.argmax(unc_diff)
    return idx_new

  def al_acquisition(self,X,y_std,method="uc"):
    method    = getattr(self,"al_"+method)
    idx_new   = method(X,y_std)
    X_new     = X[idx_new]
    return X_new

  def gp_plot(self,x,y_mean,y_std,x_samples=np.array([]),y_samples=np.array([]),save_name="",plot_aqc=True):
    if True: # Plotting two y-axes

      f, ax1 = plt.subplots()

      ax1.set_xlabel(r"$\Theta$")
      ax1.set_ylabel(r"$\langle  H | \rangle_{\infty}$", color='blue')
      ax1.plot(x, y_mean,"--s", color='blue', zorder=9,label=r"\mathcal{GP}($\mu$)")
      ax1.fill_between(x, y_mean - y_std, y_mean + y_std, 
                        alpha=0.2, color='blue')#,label=r"\sigma$")
      if x_samples.size == y_samples.size > 0:
        ax1.scatter(x_samples,y_samples,label="Samples",s=200, facecolors='none', edgecolors='b');
      # ax1.tick_params(axis='y', labelcolor=color)

      ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

      ax2.set_ylabel('Acquisition', color="red")  # we already handled the x-label with ax1
      f_aqc,p_aqc = self.bo_acquisition(y_mean,y_std,use_prior=True,use_sample=True,return_acq=True)
      ax2.plot(x, p_aqc,"--*", color="red")
      # ax2.tick_params(axis='y', labelcolor=color)

      # plt.legend()
      f.tight_layout()  # otherwise the right y-label is slightly clipped

    else:
      f = plt.figure()
      plt.plot(x, y_mean,"--*", color='blue', zorder=9,label=r"\mathcal{GP}($\mu$)")
      plt.fill_between(x, y_mean - y_std, y_mean + y_std, 
                        alpha=0.2, color='blue')#,label=r"\sigma$")
      if x_samples.size == y_samples.size > 0:
        plt.scatter(x_samples,y_samples,label="Samples",s=200, facecolors='none', edgecolors='b');
      plt.ylabel(r"$\langle \psi | H | \psi \rangle$")
      plt.legend(bbox_to_anchor=(1.4, 1))#

      plt.xlabel(r"$\Theta$")
    
    if save_name !="" :
      f.savefig(self.savepth+"gp-plot---"+save_name+".pdf")

    plt.show()

  def construct_THETA_priors(self):
    for theta in self.THETA:
      t_min,t_max         = self.THETA[theta]["constraints"]
      if self.THETA[theta]["type"] == "int":
        self.THETA_i      = np.linspace(t_min,t_max,1+t_max-t_min,dtype=int)
      else:
        self.THETA_i      = np.linspace(t_min,t_max,self.p_d,dtype=float)
      self.n_THETAS       = len(self.THETA_i)
      
      if self.THETA[theta]["prior"] == "exp":
        decl              = self.THETA["L"]["prior_params"]
        self.pT           = ("%.2E" % +decl).replace(".","-")
        self.p_THETA      = np.exp(-decl*self.THETA_i)/np.sum(np.exp(-decl*self.THETA_i))
      elif self.THETA[theta]["prior"] == "uni":
        l                 = self.THETA_i.shape[0]
        self.p_THETA      = np.tile(1/l,(l,))

  def X_Jb_2_Jb(self,X_Jb):
    return np.append(list(X_Jb.values())[0][0],list(X_Jb.values())[1][0]) if self.two_D_ex else np.concatenate(list(X_Jb.values()))

  def get_X_Jb(self,x=[],random=False):
    J_min,J_max = self.Jb["J"][0],self.Jb["J"][1]
    b_min,b_max = self.Jb["b"][0],self.Jb["b"][1]
    if random:
      if self.two_D_ex:
        x_1         = J_min + np.random.rand(1)*(J_max-J_min)
        x_2         = b_min + np.random.rand(1)*(b_max-b_min)
        X_Jb        = {self.x_1_term: np.tile(x_1,(self.N_QUBITS,)), self.x_2_term: np.tile(x_2,(self.N_QUBITS,))}
      else:
        X_Jb        = {H_J: [] for H_J in self.H_J_terms}
        X_Jb.update({H_b: [] for H_b in self.H_b_terms})  
        for H_J in self.H_J_terms:
          x_1         = J_min + np.random.rand(self.N_QUBITS,)*(J_max-J_min)
          X_Jb[H_J]   = x_1
        for H_b in self.H_b_terms:
          x_2         = b_min + np.random.rand(self.N_QUBITS,)*(b_max-b_min)
          X_Jb[H_b]   = x_2
    else:
      if self.two_D_ex:
        x_1         = x[0]
        x_2         = x[1]
        X_Jb        = {self.x_1_term: np.tile(x_1,(self.N_QUBITS,)), self.x_2_term: np.tile(x_2,(self.N_QUBITS,))}
      else:
        X_Jb        = {H_J: [] for H_J in self.H_J_terms}
        i_t         = 0
        for H_J in self.H_J_terms:
          X_Jb[H_J]   = np.append(X_Jb[H_J],x[i_t])
          i_t         += 1

        X_Jb.update({H_b: [] for H_b in self.H_b_terms})  
        for H_b in self.H_b_terms:
          X_Jb[H_b]   = np.append(X_Jb[H_b],x[i_t])
          i_t         += 1

    return X_Jb

  def sample_Jb_point(self,random=True,n=0):
    n_points      = n if n > 0 else self.n_a_test
    J_grid        = np.linspace(self.Jb["J"][0],self.Jb["J"][1],n_points)
    b_grid        = np.linspace(self.Jb["b"][0],self.Jb["b"][1],n_points)

    sample_idx    = np.random.randint(low=0,high=n_points-1,size=(n_points,self.DIMS_X))
    J_samples     = J_grid[sample_idx[:,:self.DIMS_J]]
    b_samples     = b_grid[sample_idx[:,self.DIMS_J:]]

    X_samples     = np.append(J_samples,b_samples,axis=1)

    return X_samples

  def get_Jb_by_AL(self):
    THETA_test      = np.random.choice(self.THETA_i,size=(self.n_a_test,1),p=self.p_THETA)
    Jb_test         = self.sample_Jb_point()  
    X_test          = np.append(Jb_test,THETA_test,axis=1)
    _,H_std         = self.gp.predict(X_test,return_std=True)
    X_Jb_new        = self.al_acquisition(X_test,H_std)[:self.DIMS_X]

    if self.plot_Jb:
      plt.tricontourf(Jb_test[:,0], Jb_test[:,1], H_std, 15)
      plt.grid(False)
      plt.scatter(self.Jbs.iloc[:,0],self.Jbs.iloc[:,1],color="white")
      plt.xlim([0,2])
      plt.ylim([0,2])
      plt.colorbar()
      plt.show()

    return X_Jb_new

  def sample_Jb_by_random(self,GS=True):
    n_points      = self.grid_space if GS else self.n_a_test
    J_grid        = np.linspace(self.Jb["J"][0],self.Jb["J"][1],n_points)
    b_grid        = np.linspace(self.Jb["b"][0],self.Jb["b"][1],n_points)
    i = 0
    while i < 1000:
      J_sample_idx  = np.random.randint(low=0,high=n_points-1,size=(1,self.DIMS_J))
      b_sample_idx  = np.random.randint(low=0,high=n_points-1,size=(1,self.DIMS_b))
      X_Jb_new      = np.append(J_grid[J_sample_idx],b_grid[b_sample_idx],axis=1)
      contained     = (self.Jbs == X_Jb_new).all(1).any()
      if not contained:
        return np.squeeze(X_Jb_new)
      i += 1
    return []

  def sample_THETA(self,H_mean,H_std):
    # Query acqusition function
    if self.THETA_search == "BO":
      THETA_new       = self.bo_acquisition(H_mean,H_std,use_prior=True,use_sample=False)
    elif self.THETA_search == "RS":
      THETA_new       = np.random.choice(self.THETA_i,p=[1/len(self.THETA_i)]*len(self.THETA_i))
    elif self.THETA_search == "GS":
      for t in range(1,8+1):
        if t not in self.cur_THETA:
          THETA_new       = t
          break
    return THETA_new

  def make_test_set(self,n=0):
    n_points        = n if n > 0 else self.n_a_test
    THETA_test      = np.random.choice(self.THETA_i,size=(n_points,1),p=self.p_THETA)
    Jb_test         = self.sample_Jb_point(n=n_points)  
    self.X_test     = np.append(Jb_test,THETA_test,axis=1)
    y_test          = []
    for test_point in Jb_test:
      X_Jb          = self.get_X_Jb(x=test_point)
      _,E_0,_       = self.get_Hamiltonian(X_Jb)
      y_test.append(E_0)
    self.y_test     = np.array(y_test)

  def test_nllh(self):
    y_p,y_sigma     = self.gp.predict(self.X_test,return_std=True)
    nllh            = -np.mean(-(1/2)*np.log(2*np.pi*y_sigma) - (1/(2*y_sigma))*np.sum((y_p-self.y_test)**2))
    return nllh

  def run_test(self,n_t_epochs):
    """ TEST """
    self.results.update({"test_loss": []})
    i_r                 = 0
    for t in range(n_t_epochs):
      Jb                = self.sample_Jb_by_random(GS=False)
      Jb_THETAs         = np.append(np.tile(Jb[:,np.newaxis].T,reps=(self.n_THETAS,1)),self.THETA_i[:,np.newaxis],axis=1)
      H_mean            = self.gp.predict(Jb_THETAs)
      THETA             = self.THETA_i[np.argmin(H_mean)]
      X_Jb              = self.get_X_Jb(x=Jb) 
      H,E_0,E_1         = self.get_Hamiltonian(X_Jb)

      self.qc           = QC(H=H, 
                          L=THETA, 
                          N_system=self.N_QUBITS,
                          ansatz=self.ansatz,
                          seed=self.seed+t) 
      for c in range(self.n_c_epochs):
        self.qc.train()
      psi_H_psi         = np.min(self.qc.energy)
      if self.two_D_ex:
        Jb_           = [Jb[0]]*self.N_QUBITS
        Jb_.extend([Jb[1]]*self.N_QUBITS)
      else:
        Jb_           = list(Jb)
      Jb_.extend([THETA,psi_H_psi,E_0,E_1])
      self.test_results.loc[i_r] = Jb_
      i_r += 1
      self.test_results.to_csv(self.savepth+"test-result---"+self.settings+".csv",index=False)
      self.results["test_loss"].append((psi_H_psi-E_0)/np.abs(E_0))

  def run(self,n_a_epochs=0,n_b_epochs=0,n_c_epochs=0,n_t_epochs=0,save_res=True):
    """--------------------Active Learning--------------------""" 
    if self.continue_exp:
      X_Jb_new        = self.get_Jb_by_AL()
      X_Jb            = self.get_X_Jb(x=X_Jb_new[:-1]) 
      i_r             = self.train_results.shape[0]
      i_a             = 10
    else:
      X_Jb            = self.get_X_Jb(random=True) 
      i_r             = 0
      i_a             = 0
    
    self.all_al_loss      = []
    self.all_bo_loss      = []
    self.all_qc_energies  = []
    self.best_psi_H_psi, self.best_THETA, self.results["test_nllh"] = [],[],[]
    self.Jbs          = pd.DataFrame(columns=self.X_terms) if self.two_D_ex else self.train_results[:,self.DIMS_X]
    """ TRAINING """
    for a in tqdm(range(self.n_a_epochs),leave=False):
      self.cur_THETA,self.cur_psi_H_psi = [],[]
      Jb              = self.X_Jb_2_Jb(X_Jb)
      self.Jbs.loc[i_a]   = Jb
      i_a             += 1
      H,E_0,E_1       = self.get_Hamiltonian(X_Jb)
      self.E_0.append(E_0)
      self.E_1.append(E_1)
      THETA_test      = np.append(np.tile(Jb[:,np.newaxis].T,reps=(self.n_THETAS,1)),self.THETA_i[:,np.newaxis],axis=1)
      # Sample hyperparameters
      H_mean,H_std    = self.gp.predict(THETA_test,return_std=True)
      THETA           = self.sample_THETA(H_mean,H_std)
      self.gp         = GP(kernel=self.Kernel,noise=0.00001,random_state=self.seed)
      qc_energies     = []

      """--------------------Bayesian Optimization--------------------""" 
      for b in range(self.n_b_epochs):
        psi_H_psi     = []
        cur_qc_energies     = []
        """--------------------Circuit Learning--------------------""" 
        for c_r in range(self.c_restarts):
          self.qc      = QC(H=H, 
                      L=THETA, 
                      N_system=self.N_QUBITS,
                      ansatz=self.ansatz,
                      seed=self.seed*b + c_r) 
          
          for c in range(self.n_c_epochs):
            self.qc.train()
          psi_H_psi.append(np.min(self.qc.energy))
          cur_qc_energies.append(self.qc.energy)
        qc_energies.append(cur_qc_energies[np.argmin(psi_H_psi)])
        psi_H_psi       = np.min(psi_H_psi)

        self.Jb_THETA   = np.append(self.Jb_THETA,np.append(Jb,THETA)[:,np.newaxis].T,axis=0)
        self.psi_H_psi  = np.append(self.psi_H_psi,psi_H_psi)
        
        # Fit surrogate
        self.gp.fit(self.Jb_THETA,self.psi_H_psi)
        # Predict on test set
        H_mean,H_std    = self.gp.predict(THETA_test,return_std=True)
        # Get new hyperparameters
        THETA_new       = self.sample_THETA(H_mean,H_std)
        
        # Save results
        if self.two_D_ex:
          Jb_           = [Jb[0]]*self.N_QUBITS
          Jb_.extend([Jb[1]]*self.N_QUBITS)
        else:
          Jb_           = list(Jb)
        Jb_.extend([THETA,psi_H_psi,E_0,E_1])
        self.train_results.loc[i_r] = Jb_
        self.train_results.to_csv(self.savepth+"train-result---"+self.settings+".csv",index=False)
        i_r += 1
        # Print progress
        self.cur_THETA.append(THETA)
        self.cur_psi_H_psi.append(psi_H_psi)
        if self.plot_gp:
          self.gp_plot(self.THETA_i,H_mean,H_std,save_name=self.settings+"--iter-"+str(b+1),x_samples=np.array(self.cur_THETA),y_samples=np.array(self.cur_psi_H_psi))

        # Stop Criteria
        i_sort  = np.argsort(self.cur_psi_H_psi)
        H_sort  = np.array(self.cur_psi_H_psi)[i_sort]
        if (b > 2 and (np.abs(H_sort[0]-H_sort[1])<self.y_tol)):
          break

        # THETA Update
        THETA = THETA_new

      self.best_psi_H_psi.append(np.min(self.cur_psi_H_psi))
      self.best_THETA.append(self.cur_THETA[np.argmin(self.cur_psi_H_psi)])
      self.all_qc_energies.append(qc_energies)

      if self.Jb_search == "AL":
        X_Jb_new        = self.get_Jb_by_AL()
      if self.Jb_search == "GS":
        X_Jb_new        = self.sample_Jb_by_random(GS=True)
      if self.Jb_search == "RS":
        X_Jb_new        = self.sample_Jb_by_random(GS=False)
      if len(X_Jb_new)  == 0:
        break

      self.results["test_nllh"].append(self.test_nllh())
      X_Jb              = self.get_X_Jb(x=X_Jb_new) 
    
    """ Append """
    self.results.update({"Jb":self.Jbs})
    self.results.update({"Jb_THETA":self.Jb_THETA})
    self.results.update({"psi_H_psi":self.psi_H_psi})
    self.results.update({"qc_learning_curves":self.all_qc_energies})

    self.run_test(n_t_epochs)
    with open(self.savepth+"result---"+self.settings+".pickle", 'wb') as handle:
        pickle.dump(self.results, handle, protocol=pickle.HIGHEST_PROTOCOL)
