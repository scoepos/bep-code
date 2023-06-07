import matplotlib.pyplot as plt
import numpy as np
import mstdef

L = 30  # [m]
E = 29.43 * 10 ** 9  # e-modulus [N/m^2]
I = 2.9
A = 1  # cross section [m^2]
m = 36056  # self weight of the beam
dam_var = 0.8  # how much the EI will decrease due to damage
num_el = 60  # number of chose elements
L_elm = L / num_el  # length of the elements
place_of_damage = [0, 1]  # which element is damaged
num_nod = num_el + 1  # number of nodes in the model
damage_off_on = False  # setting the damage off and on
mw = 5000 # weight of the wheel
mv = 24000  # weight of the train
p = ((mw + mv) * 9.81)  # force of the train
damping_ratio = 0.025  # damping ratio for this beam
beta, gamma = 0.25, 0.5  # Newmark's beta method
kv = 1500*10**3  # spring inbetween wheel and train
kb = 4*10**7  # bedding constant
cv = 85*10**3
Mt = np.array([[mw,0],[0,mv]])
Kt = np.array([[kv,-kv],[-kv,kv]])
Ct = np.array([[cv, -cv], [-cv, cv]])
f_Mm = mv*9.81

# making stiffness matrix
lk = mstdef.local_stiffness_matrix(E, I, L_elm)
gk = mstdef.global_stiffness_matrix(damage_off_on, num_el, place_of_damage, lk, dam_var, num_nod)
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


#making the place vector
f_pos_1 = 0.3
t1 = 20
t = np.linspace(0,t1, 201)
v = L/t1
f_pos = t*v + f_pos_1
dt = t[1] - t[0]

#making shape matrix
element_number = np.trunc(f_pos / L_elm)
s = (f_pos - element_number * L_elm) / L_elm
N1 = 1 - 3 * s ** 2 + 2 * s ** 3
N2 = L_elm * (s - 2 * s ** 2 + s ** 3)
N3 = 3 * s ** 2 - 2 * s ** 3
N4 = L_elm * (- s ** 2 + s ** 3)
N_vec = np.array([N1, N2, N3, N4])
N_matrix = N_vec.transpose()
N_0 = np.zeros(num_nod*2)
N_0[0:4] = N_matrix[0]
N_0 = np.delete(N_0, [0,-2])

# setting initial conditions
u = np.linalg.inv(gk)@(-N_0*p)
u_dot = np.zeros(num_nod*2 - 2)
u_dot_dot = np.zeros(num_nod*2 - 2)
z1 = N_0.transpose()@u - p/kb
#z2 = z1 - f_Mm/kv
z = z1
z_dot = 0
z_dot_dot = 0
plt.plot(u)
plt.show()
#making integration constants
a0 = 1/(beta*dt**2)
a1 = gamma/(beta*dt)
a2 = 1/(beta*dt)
a3 = (1/(2*beta)) - 1
a4 = 1 - (gamma/beta)
a5 = dt*(1-(gamma/(2*beta)))
a6 = dt*(1-gamma)
a7 = gamma*dt
l = []
fc = p
# making effective matrix
for n in range(len(f_pos)):
    if f_pos[n+1] > 25:
        break
    else:
        nf = int(element_number[n+1])
        N_vec = np.zeros(num_nod * 2)
        N_vec[nf * 2:nf * 2 + 4] = N_matrix[n+1]
        N_vec = np.delete(N_vec, [0, -2])
        F_vec = N_vec * -fc
        for n in range(1):
            fc_n = fc
            z_dot_dot_n = z_dot_dot
            z_dot_n = z_dot
            z_dot_dot = (-p+fc)/(mw+mv)
            z_dot = z_dot_n + (dt/2)*(z_dot_dot_n + z_dot_dot)
            z += (dt/2)*(z_dot_n+ z_dot)
            #z, z_dot, z_dot_dot = mstdef.newmark(Mt, Ct, Kt, F, gamma, beta, dt, z, z_dot, z_dot_dot)
            u, u_dot, u_dot_dot = mstdef.newmark(gm, gc, gk, F_vec, gamma, beta, dt, u, u_dot, u_dot_dot)
            fc = kb*(N_vec.transpose()@u - z)
            F_vec = -N_vec * fc
            l.append(N_vec.transpose()@u)

print(l)
plt.plot(range(len(l)) , l)
plt.show()