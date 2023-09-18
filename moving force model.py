import matplotlib.pyplot as plt
import numpy as np
import mstdef

L = 25  # [m]
E = 2.87 * 10 ** 9  # e-modulus [N/m^2]
I = 2.9
A = 1  # cross section [m^2]
m = 2303  # self weight of the beam
dam_var = 0.4  # how much the EI will decrease due to damage
num_el = 50  # number of chose elements
L_elm = L / num_el  # length of the elements
#place_of_damage = np.linspace(5,14, 10)
place_of_damage = np.linspace(25,34, 10) # which element is damaged
num_nod = num_el + 1  # number of nodes in the model
damage_off_on = False  # setting the damage off and on
mw = 5000 # weight of the wheel
mv = 24000  # weight of the train
p = mw  * 9.81  # force of the train
damping_ratio = 0.025  # damping ratio for this beam
beta, gamma = 0.25, 0.5  # Newmark's beta method

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
gc_dam = s1_dam*gm + s2_dam*gk_dam

# setting initial conditions
u = np.zeros(num_nod * 2 - 2)
u_dot = np.zeros(num_nod*2 - 2)
u_dot_dot = np.zeros(num_nod*2 - 2)
u_dam = np.zeros(num_nod * 2 - 2)
u_dot_dam = np.zeros(num_nod*2 - 2)
u_dot_dot_dam = np.zeros(num_nod*2 - 2)

#making the place vector
t1 = 1
t = np.linspace(0,t1, 2001)
v = 30
f_pos = t*v
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

#making integration constants
a0 = 1/(beta*dt**2)
a1 = gamma/(beta*dt)
a2 = 1/(beta*dt)
a3 = (1/(2*beta)) - 1
a4 = 1 - (gamma/beta)
a5 = dt*(1-(gamma/(2*beta)))
a6 = dt*(1-gamma)
a7 = gamma*dt

# making effective matrix
A = a0 * gm + a1*gc + gk

midu = []
midu_dam = []
t_new = []
#starting time integration
for n in range(len(t)):
    if n == 2001:
        break
    if f_pos[n] >= L:
        break
    nf = int(element_number[n])
    F_vec = np.zeros(num_nod*2)
    F_vec[nf*2:nf*2+4] = N_matrix[n]*-p
    F_vec = np.delete(F_vec,[0,-2])
    u, u_dot, u_dot_dot = mstdef.newmark(gm, gc, gk, F_vec, gamma, beta, dt, u, u_dot, u_dot_dot)
    u_dam, u_dot_dam, u_dot_dot_dam = mstdef.newmark(gm, gc_dam, gk_dam, F_vec, gamma, beta, dt, u_dam, u_dot_dam, u_dot_dot_dam)
    midu.append(u[9])
    midu_dam.append(u_dam[9])
    t_new.append(t[n])
    print(n)

plt.figure(figsize=(10,7))
plt.plot(t_new, midu, label='without damage')
plt.plot(t_new, midu_dam, label='with damage = 0.4')
plt.axhline(0, color='k')
plt.title('displacement of the fourth node when damage is placed at the middle')
plt.legend()
plt.grid()
plt.xlabel('time')
plt.ylabel('displacement')
plt.savefig('moving force model mid with 0,6 mid fourth node')
print('done')