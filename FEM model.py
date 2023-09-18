import pandas as pd
import mstdef
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

L = 25  # [m]
E = 2.87 * 10 ** 9  # e-modulus [N/m^2]
I = 2.9
A = 1  # cross section [m^2]
m = 2303  # self weight of the beam
dam_var = 0.6  # how much the EI will decrease due to damage
num_el = 50  # number of chose elements
L_elm = L / num_el  # length of the elements
place_of_damage = np.linspace(25,34, 10) # which element is damaged
num_nod = num_el + 1  # number of nodes in the model
damage_off_on = True  # setting the damage off and on
print(place_of_damage)
#making stiffness matrix
lK = mstdef.local_stiffness_matrix(E, I, L_elm)
K_dam = mstdef.global_stiffness_matrix(True, num_el, place_of_damage, lK, dam_var, num_nod)
K = mstdef.global_stiffness_matrix(False, num_el, place_of_damage, lK, dam_var, num_nod)

#making mass matrix
lM = mstdef.local_mass_matrix(A,m,L_elm)
M = mstdef.global_mass_matrix(lM, num_nod, num_el)

#setting initial conditions
K = np.delete(K, [0,-2], 0)
K = np.delete(K, [0,-2], 1)
M = np.delete(M, [0,-2], 0)
M = np.delete(M, [0,-2], 1)
K_dam = np.delete(K_dam, [0,-2], 0)
K_dam = np.delete(K_dam, [0,-2], 1)

#making modes
eig_val, eig_vec = sc.linalg.eigh(K,M)
eig_val_dam, eig_vec_dam = sc.linalg.eigh(K_dam,M)
mode1 = eig_vec[:,2].reshape(num_nod - 1 ,2)[:,1]
mode1 = mode1[:-1]
u1 = np.concatenate([[0], mode1, [0]])
mode1_dam = eig_vec_dam[:,2].reshape(num_nod - 1 ,2)[:,1]
mode1_dam = mode1_dam[:-1]
u1_dam = np.concatenate([[0], mode1_dam, [0]])
#making plots
x = np.linspace(0,L,num_nod)
plt.plot(x, u1, label = 'undamaged')
plt.plot(x, u1_dam, label = 'damaged')
plt.title(f'third mode of the bridge')
plt.axhline(0, color = 'k')
plt.grid()
plt.legend()
plt.savefig('undamaged and damaged mode 3')
print(np.sqrt(eig_val_dam[0]),np.sqrt(eig_val_dam[1]),np.sqrt(eig_val_dam[2]))