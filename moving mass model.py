import matplotlib.pyplot as plt
import numpy as np
import mstdef

L = 30  # [m]
E = 29.43*10**9  # e-modulus [N/m^2]
I = 8.72 # moment of inertia
A = 7.94  # cross section [m^2]
m = 36056  # self weight of the beam
dam_var = 0.6  # how much the EI will decrease due to damage
num_el = 60  # number of chose elements
L_elm = L / num_el  # length of the elements
place_of_damage = np.linspace(25,34, 10) # which element is damaged
num_nod = num_el + 1  # number of nodes in the model
damage_off_on = False  # setting the damage off and on
mw = 5000 # weight of the wheel
mv = 24000 # weight of the train
mt = mw
p = ((mw) * 9.81)  # force of the train
damping_ratio = 0.025  # damping ratio for this beam
beta, gamma = 0.25, 0.5  # Newmark's beta method
kv = 1500*10**3  # spring inbetween wheel and train
kb = 4*10**7  # bedding constant
cv = 85*10**3
Mt = np.array([[mv,0],[0,mw]])
Kt = np.array([[-kv,kv],[kv,-kv]])
Ct = np.array([[-cv, cv], [cv, -cv]])

# making stiffness matrix
lk = mstdef.local_stiffness_matrix(E, I, L_elm)
gk_dam = mstdef.global_stiffness_matrix(True, num_el, place_of_damage, lk, dam_var, num_nod)
gk_dam = np.delete(gk_dam, [0, -2], 1)
gk_dam = np.delete(gk_dam, [0, -2], 0)
gk = mstdef.global_stiffness_matrix(False, num_el, place_of_damage, lk, dam_var, num_nod)
gk = np.delete(gk, [0, -2], 1)
gk = np.delete(gk, [0, -2], 0)

# making mass matrix
lm = mstdef.local_mass_matrix(A, m, L_elm)
gm = mstdef.global_mass_matrix(lm, num_nod, num_el)
gm = np.delete(gm, [0, -2], 1)
gm = np.delete(gm, [0, -2], 0)

#making damping matrix
s1,s2,w1,w2 = mstdef.getting_damping_coeffients(gk,gm,damping_ratio)
gc = s1*gm + s2*gk
s1_dam,s2_dam,w1_dam,w2_dam = mstdef.getting_damping_coeffients(gk_dam,gm,damping_ratio)
gc_dam = s1*gm + s2*gk_dam

# setting initial conditions
u = np.zeros(num_nod * 2 - 2)
u_dot = np.zeros(num_nod*2 - 2)
u_dot_dot = np.zeros(num_nod*2 - 2)
u_dam = np.zeros(num_nod * 2 - 2)
u_dot_dam = np.zeros(num_nod*2 - 2)
u_dot_dot_dam = np.zeros(num_nod*2 - 2)
z = np.zeros(2)
z_dot = np.zeros(2)
z_dot_dot = np.zeros(2)

#making the place vector
t1 = 1
t = np.linspace(0,t1, 2001)
v = 30
f_pos = t*v + 0.3
dt = t[1] - t[0]

#making shape matrix
element_number = np.trunc(f_pos / L_elm)
element_number[-1] = element_number[-1] - 1
s = (f_pos - element_number * L_elm) / L_elm
N1 = 1 - 3 * s ** 2 + 2 * s ** 3
N2 = L_elm * (s - 2 * s ** 2 + s ** 3)
N3 = 3 * s ** 2 - 2 * s ** 3
N4 = L_elm * (- s ** 2 + s ** 3)
N_vec = np.array([N1, N2, N3, N4])
N_matrix = N_vec.transpose()
f = 0
f_n = np.inf
midu = []
g = 9.81
flist = []
#starting time integration
for n in range(len(t)):
    if n == 1:
        break
    while np.abs(f - f_n) > 1:
        f_n = f
        nf = int(element_number[n])
        N_vec = np.zeros(num_nod * 2)
        N_vec[nf * 2:nf * 2 + 4] = N_matrix[n]
        N_vec = np.delete(N_vec, [0, -2])
        z[1], z_dot[1], z_dot_dot[1] = u@N_vec, u_dot@N_vec, u_dot_dot@N_vec
        f = mw*(g-z_dot_dot[1]) + kv*(z[0]-z[1]) + cv*(z_dot[0]-z_dot[1]) + mv*g
        f1 = np.array([0,f])
        F_vec = -f*N_vec
        u, u_dot, u_dot_dot = mstdef.newmark(gm, gc, gk, F_vec, gamma, beta, dt, u, u_dot, u_dot_dot)
        z, z_dot, z_dot_dot = mstdef.newmark(Mt, Ct, Kt, f1, gamma, beta, dt, z, z_dot, z_dot_dot)
        flist.append(f - f_n)

plt.plot(range(len(flist[5::])), flist[5::])
plt.show()
