import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sp
import scipy as sc

# defining al the needed variables
L = 25  # [m]
E = 2.87*10**9  # e-modulus [N/m^2]
I = 2.9  # moment of inertia [m^4]
A = 1  # cross section [m^2]
w = 2303 # self weight of the beam [N/m^3]
dam_var = 0.8  # how much the EI will decrease due to damage
num_el = 600 # number of chose elements
L_elm = L/num_el  # length of the elements
place_of_damage = [0,1]  # which element is damaged
num_nod = num_el + 1 # number of nodes in the model
damage_off_on = False  # setting the damage off and on
F = 1000 # force on beam
s0 = 0.2
s1 = 0.2
betha, gamma = 0.25 , 0.5 # newarks betha method

# defining the stiffness matrix
# local stiffness matrix made out of beam theory
def local_stiffness_matrix(E,I,L):
    lk = ((E*I)/(L**3)) * \
         np.array([[12, 6*L, -12, 6*L],
                   [6*L, 4*L**2, -6*L, 2*L**2],
                   [-12, -6*L, 12, -6*L],
                   [6*L, 2*L**2, -6*L, 4*L**2]])
    return lk

lk = local_stiffness_matrix(E, I, L_elm)

#global stiffness matrix
gk = np.zeros((2 * num_nod, 2 * num_nod))

if damage_off_on:
    for i in range(num_el):
        if i in place_of_damage:
            print('damage')
            gk[2 * i:2 * i + 4, 2 * i:2 * i + 4] += lk*dam_var
        else:
            gk[2 * i:2 * i + 4, 2 * i:2 * i + 4] += lk
else:
    for i in range(num_el):
        gk[2 * i:2 * i + 4, 2 * i:2 * i + 4] += lk

# defining the mass matrix
# local mass matrix
def local_mass_matrix(A,w,L):
    lm = ((A*w*L)/420) * \
         np.array([[156, 22*L, 54, -13*L],
                   [22*L, 4*L**2, 13*L, -3*L**2],
                   [54, 13*L, 156, -22*L],
                   [-13*L, -3*L**2, -22*L, 4*L**2]])
    return lm
lm = local_mass_matrix(A,w,L_elm)
# global mass matrix
gm = np.zeros((2 * num_nod, 2 * num_nod))
for i in range(num_el):
    gm[2 * i:2 * i + 4, 2 * i:2 * i + 4] += lm

#making damping matrix

C = np.zeros((2 * num_nod, 2 * num_nod))
C = gm*s0 + gk*s1

# applying boundary conditions wich are u1 = 0 and u(n-2) = 0
gm = np.delete(gm, [0,-2], 1)
gm = np.delete(gm, [0,-2], 0)
gk = np.delete(gk, [0,-2], 1)
gk = np.delete(gk, [0,-2], 0)
C = np.delete(C, [0,-2], 1)
C = np.delete(C, [0,-2], 0)
# making dataframes to look at the matrices
def dataframes(C,gk,gm):
    Cdata = pd.DataFrame(C)
    kdata = pd.DataFrame(gk)
    mdata = pd.DataFrame(gm)
    print(Cdata)
    print(kdata)
    print(mdata)

#dataframes(C,gk,gm)

# calculating eigenvalues and eigenvectors
eig_val, eig_vec = sc.linalg.eigh(gk,gm)
print(eig_vec)
print(np.sqrt(eig_val))
# making plots of the beam with different modes according to the eigen vectors
def makeplots_ofbeam(modes, eig_vec,num_nod):
    x = np.linspace(0,L,num_nod)
    for n in range(modes):
        mode1 = eig_vec[:,n].reshape(num_nod - 1 ,2)[:,1]
        mode1 = mode1[:-1]
        u1 = np.concatenate([[0],mode1,[0]])
        plt.plot(x, u1, label = f'mode {n+1}')
        plt.axhline(0, label = 'beam', color = 'k')
        plt.title(f'mode {n+1}')
        plt.legend()
        plt.savefig(f'mode {n}')
        plt.show()

makeplots_ofbeam(3,eig_vec,num_nod)

# making the shape functions for the force
t = np.linspace(0,10,21)
dt = t[1] - t[0]
v_f = 2.5
f_pos = v_f*t
nl = -L_elm/2
element_number = np.trunc(f_pos/L_elm)
x_pos = (f_pos - element_number * L_elm + nl)/(L_elm/2)
a0 = 1/(betha*dt**2)
a1 = gamma/(betha*dt)
a2 = 1/(betha*dt)
a3 = (1/(2*betha)) - 1
a4 = (gamma/betha) - 1
a5 = (dt/2)*((gamma/betha)-2)
a6 = dt*(1-gamma)
a7 = gamma*dt
for n in range(len(f_pos)):
    f_vector = np.zeros(num_nod * 2)
    N1 = (1/(2*L_elm))*(-3+3*(x_pos[n]**2))
    N2 = (1/4)*(-1-2*x_pos[n]+3*x_pos[n]**2)
    N3 = (1/(2*L_elm))*(3-3*(x_pos[n]**2))
    N4 = (1/4)*(-1+2*x_pos[n]+3*x_pos[n]**2)
    f_vector[int(element_number[n]):int(element_number[n])+4] = np.array((N1,N2,N3,N4))*F



