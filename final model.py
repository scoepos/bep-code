import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sp
import scipy as sc
from numpy import ndarray
import mstdef

# defining al the needed variables
L = 30  # [m]
E = 29.43*10**9 # e-modulus [N/m^2]
I = 8.72  # moment of inertia [m^4]
A = 7.94  # cross section [m^2]
m = 36056 # self weight of the beam
dam_var = 0.8  # how much the EI will decrease due to damage
num_el = 4 # number of chose elements
L_elm = L/num_el  # length of the elements
place_of_damage = [0,1]  # which element is damaged
num_nod = num_el + 1 # number of nodes in the model
damage_off_on = False  # setting the damage off and on
betha, gamma = 0.25 , 0.5 # newarks betha method
t = np.linspace(0,10,201) # making the time vector
dt = t[1] - t[0]  # making dt
v_f = 2.5  # speed of the force
f_pos = v_f*t  # position vector of the force
nl = -L_elm/2  # left coordinate of the local coordinate system
element_number = np.trunc(f_pos/L_elm)  # wich vector the force is on
x_pos = (f_pos - element_number * L_elm + nl)/(L_elm/2)  # position of the force in local coordinate sytem
mw = 5000
mv = 24000
p = (mw + mv)*9.81
kv = 1500
cv = 85
kb = 40*10**3
rc = 0
damping_ratio = 0.025
ub = np.zeros(num_nod*2 - 2)
ub_dot = np.zeros(num_nod*2 - 2)
ub_dotdot = np.zeros(num_nod*2 - 2)
pb = np.zeros(num_nod*2 - 2)
z1 = 0
z1_dot = 0
z1_dotdot = 0
z2 = 0
z2_dot = 0
z2_dotdot = 0

#  coefficients needed to solve this system
a0 = 1/(betha*dt**2)
a1 = gamma/(betha*dt)
a2 = 1/(betha*dt)
a3 = (1/(2*betha)) - 1
a4 = (gamma/betha) - 1
a5 = (dt/2)*((gamma/betha)-2)
a6 = dt*(1-gamma)
a7 = gamma*dt
# mass matrix
lm = mstdef.local_mass_matrix(A, m, L_elm)
gm = mstdef.global_mass_matrix(lm, num_nod, num_el)
# stiffness matrix
lk = mstdef.local_stiffness_matrix(E, I, L_elm)
gk1 = mstdef.global_stiffness_matrix(damage_off_on, num_el, place_of_damage, lk, dam_var, num_nod)
# getting damping ratios for the structure
s1, s2, w1, w2 = mstdef.getting_damping_coeffients(gk1,gm,damping_ratio)
# making vehicle equations
Dm = np.array([[kv+kb+a0*mw+a1*cv, -kv-a1*cv],
               [-kv-a1*cv, kv+a0*mv+a1*cv]])
D = np.linalg.det(Dm)

#setting initial conditions for mass matrix
gm = np.delete(gm, [0, -2], 1)
gm = np.delete(gm, [0, -2], 0)
# making a matrix for the shape vectors
row1N = (1/(2*L_elm))*(-3+3*(x_pos**2))
row2N = (1/4)*(-1-2*x_pos+3*x_pos**2)
row3N = (1/(2*L_elm))*(3-3*(x_pos**2))
row4N = (1/4)*(-1+2*x_pos+3*x_pos**2)
N = np.array([row1N,row2N,row3N,row4N])
# doing the iterations
for n in range(len(f_pos)):
    el_num = int(element_number[n])
    NC = N[::,n]
    NCcolumn = NC
    NCrow = NC.transpose()
    # making the k_hat matrix
    k_hat = np.zeros((4,4))
    k_hat = lk + kb*(a0/D)*((mv + mw)*(kv+a1*cv)+a0*mv*mw)*NCcolumn@NCrow
    # making global stiffness matrix
    gk = np.zeros((2 * num_nod, 2 * num_nod))
    gk = gk1
    if damage_off_on and el_num in place_of_damage:
        print('damage')
        gk[el_num:el_num+4,el_num:el_num+4] = k_hat*dam_var
    else:
        gk[el_num:el_num + 4, el_num:el_num + 4] = k_hat
    gk = np.delete(gk, [0, -2], 1)
    gk = np.delete(gk, [0, -2], 0)
    # making the damping matrix
    gc = np.zeros((2 * num_nod - 2, 2 * num_nod - 2))
    gc = s1*gm + s2*gk
    k_eff = a0*gm + gc*a1 + gk
    #making the forces
    Fb = np.zeros(num_nod*2 - 2)
    qs1 = (kv + kb) * z1 - kv * z2
    qs2 = -kv * z1 + kv * z2
    qe1 = -mw * (a2 * z1_dot + a3 * z1_dotdot) - cv * (a4 * (z1_dot - z2_dot) + a5 * (z1_dotdot - z2_dotdot))
    qe2 = -mv * (a2 * z2_dot + a3 * z2_dotdot) - cv * (a4 * (z2_dot - z1_dot) + a5 * (z2_dotdot - z1_dotdot))
    qs = qs1 + qs2
    qe = qe1 + qe2
    if el_num == 0:
        NCcolumn = np.delete(NCcolumn, 0)
        NCrow = np.delete(NCrow, 0)
        k_hat = np.delete(k_hat, 0, 0)
        k_hat = np.delete(k_hat, 0, 1)
        pb[el_num:el_num+3] = -kb * (rc - (p+kb*rc)*(1/D)*(kv + a0*mv +a1*cv))*NCcolumn
        fs = kb * ((1/D)*(qs1+qe1)*a0*mv + (qs + qe)*(kv + a1 * cv) - z1)*NCcolumn
        Fb[num_el:num_el+3] = fs + k_hat@ub[num_el:num_el+3]
    elif el_num == num_el:
        NCcolumn = np.delete(NCcolumn, 0)
        NCrow = np.delete(NCrow, -2)
        k_hat = np.delete(k_hat, -2, 0)
        k_hat = np.delete(k_hat, -2, 1)
        pb[el_num:el_num + 3] = -kb * (rc - (p - kb * rc) * (1 / D) * (kv + a0 * mv + a1 * cv)) * NCcolumn
        fs = kb * ((1 / D) * (qs1 + qe1) * a0 * mv + (qs + qe) * (kv + a1 * cv) - z1) * NCcolumn
        Fb[num_el:num_el + 3] = fs + k_hat @ ub[num_el:num_el + 3]
    else:
        pb[el_num:el_num + 4] = -kb * (rc - (p - kb * rc) * (1 / D) * (kv + a0 * mv + a1 * cv)) * NCcolumn
        fs = kb * ((1 / D) * (qs1 + qe1) * a0 * mv + (qs + qe) * (kv + a1 * cv) - z1) * NCcolumn
        Fb[num_el:num_el + 4] = fs + k_hat @ ub[num_el:num_el + 4]
    for i in range(20):
        #fc = kb*(NCrow@ub[el_num:el_num+4] + rc - z1)
        Fb += gm@(a2*ub_dot + a3*ub_dotdot) + gc@(a4*ub_dot+a5*ub_dotdot)
        qs1 = (kv+kb)*z1 - kv*z2
        qs2 = -kv*z1 + kv*z2
        qe1 = -mw*(a2*z1_dot + a3*z1_dotdot) - cv*(a4*(z1_dot - z2_dot)+a5*(z1_dotdot-z2_dotdot))
        qe2 = -mv * (a2 * z2_dot + a3 * z2_dotdot) - cv * (a4 * (z2_dot - z1_dot) + a5*(z2_dotdot - z1_dotdot))
        qs = qs1 + qs2
        qe = qe1 + qe2
        deltaub = (pb - Fb)@np.linalg.inv(k_eff)
        deltaz1 = -1/D*((qs1 + qe1)*a0*mv + (qs+qe)*(kv+a1*cv))
        deltaz2 = -1/D*((qs1 + qe1)*(a0*mv +kb) + (qs+qe)*(kv+a1*cv))
        z1 += deltaz1
        z2 += deltaz2
        z1_dotdot_t = z1_dotdot
        z2_dotdot_t = z2_dotdot
        z1_dotdot = a0*deltaz1 - a2*z1_dot - a3*z1_dotdot
        z2_dotdot = a0 * deltaz2 - a2 * z2_dot - a3 * z2_dotdot
        z1_dot += a6*z1_dotdot + a7*z1_dotdot_t
        z2_dot += a6 * z2_dotdot + a7 * z2_dotdot_t
        ub += deltaub
        ub_dotdot_t = ub_dotdot
        ub_dotdot = a0*deltaub - a2*ub_dot - a3*ub_dotdot_t
        ub_dot += a6*ub_dotdot_t + a7*ub_dotdot
        if i == 0:
            print(pb - Fb)
        elif i == 19:
            print(pb - Fb)