import matplotlib.pyplot as plt
import numpy as np
import mstdef

L = 25  # [m]
E = 2.87 * 10 ** 9  # e-modulus [N/m^2]
I = 2.9
A = 1  # cross section [m^2]
m = 2303  # self weight of the beam
dam_var = 0.6  # how much the EI will decrease due to damage
num_el = 50  # number of chose elements
L_elm = L / num_el  # length of the elements
#place_of_damage = np.linspace(5,14, 10)
place_of_damage = np.linspace(25,34, 10)  # which element is damaged
#place_of_damage = np.linspace(40,49,10)
num_nod = num_el + 1  # number of nodes in the model
damage_off_on = False  # setting the damage off and on
mw = 5750 # weight of the wheel
mv = 24000  # weight of the train
p = mw * 9.81  # force of the train
damping_ratio = 0.025  # damping ratio for this beam
beta, gamma = 0.25, 0.5  # Newmark's beta method
kv = 1.1*10**9  # spring inbetween wheel and train
kb = 4*10**9  # bedding constant
cv = 85*10**3
Mt = np.array([[mw,0],[0,mv]])
Kt = np.array([[kv,-kv],[-kv,kv]])
Ct = np.array([[cv, -cv], [-cv, cv]])
f_Mm = mv*9.81
tol = 1

# making stiffness matrix
lk = mstdef.local_stiffness_matrix(E, I, L_elm)
gk = mstdef.global_stiffness_matrix(False, num_el, place_of_damage, lk, dam_var, num_nod)
gk = np.delete(gk, [0, -2], 1)
gk = np.delete(gk, [0, -2], 0)
gk_dam = mstdef.global_stiffness_matrix(True, num_el, place_of_damage, lk, dam_var, num_nod)
gk_dam = np.delete(gk_dam, [0, -2], 1)
gk_dam = np.delete(gk_dam, [0, -2], 0)
# making mass matrix
lm = mstdef.local_mass_matrix(A, m, L_elm)
gm = mstdef.global_mass_matrix(lm, num_nod, num_el)
gm = np.delete(gm, [0, -2], 1)
gm = np.delete(gm, [0, -2], 0)

#making damping matrix
s1,s2,w1,w2 = mstdef.getting_damping_coeffients(gk,gm,damping_ratio)
s1_dam,s2_dam,w1_dam,w2_dam = mstdef.getting_damping_coeffients(gk_dam,gm,damping_ratio)
gc = s1*gm + s2*gk
gc_dam = s1_dam*gm + s2_dam*gk_dam

#making the place vector
f_pos_1 = 0
t = np.linspace(0,1, 20001)
v = 100/3.6
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

nf_0 = int(element_number[0])
N_0 = np.zeros(num_nod*2)
N_0[nf_0 * 2:nf_0 * 2 + 4] = N_matrix[0]
N_0 = np.delete(N_0, [0,-2])

# setting initial conditions
u = np.linalg.solve(gk,-N_0*p)
u_dot = np.zeros(num_nod*2 - 2)
u_dot_dot = np.zeros(num_nod*2 - 2)
u_dam = np.linalg.solve(gk_dam,-N_0*p)
u_dot_dam = np.zeros(num_nod*2 - 2)
u_dot_dot_dam = np.zeros(num_nod*2 - 2)
# calculate the initial displacement of wheel
z = N_0@u - p/kv
z_int = N_0@u - p/kv
z_dot = 0
z_dot_dot = 0
z_dam = N_0@u_dam - p/kv
z_dot_dam = 0
z_dot_dot_dam = 0

#making lists
ulist = np.zeros([num_nod*2 - 2,len(t)])
udotlist = np.zeros([num_nod*2 - 2,len(t)])
udotdotlist = np.zeros([num_nod*2 - 2,len(t)])
udamlist = np.zeros([num_nod*2 - 2,len(t)])
udamdotlist = np.zeros([num_nod*2 - 2,len(t)])
udamdotdotlist = np.zeros([num_nod*2 - 2,len(t)])
# initial conditions
ulist[:,0] = u
udotlist[:,0] = u_dot
udotdotlist[:,0] = u_dot_dot
udamlist[:,0] = u_dam
udamdotlist[:,0] = u_dot_dam
udamdotdotlist[:,0] = u_dot_dot_dam
zlist = np.zeros([3,len(t)])
zlist[0,0] = z
zlist[1,0] = z_dot
zlist[2,0] = z_dot_dot
zdamlist = np.zeros([3,len(t)])
zdamlist[0,0] = z_dam
zdamlist[1,0] = z_dot_dam
zdamlist[2,0] = z_dot_dot_dam
midu = []
midu_dam = []
t_new = []
#starting time integration
for time_step in range(len(f_pos)):
    if f_pos[time_step] >= L:
        break
    else:
        nf = int(element_number[time_step + 1])
        if nf == num_el:
            nf -= 1
        N_vec = np.zeros(num_nod * 2)
        N_vec[nf * 2:nf * 2 + 4] = N_matrix[time_step + 1]
        N_vec = np.delete(N_vec, [0, -2])
        # initial penetration
        fc = kv*abs(z - N_vec@u)
        fc_n = np.inf
        while abs(fc-fc_n) > tol:
            fc_n = fc
            F_vec = -1 * N_vec * fc_n
            u, u_dot, u_dot_dot = mstdef.newmark(gm, gc, gk, F_vec, gamma, beta, dt, ulist[:,time_step], udotlist[:,time_step], udotdotlist[:,time_step])# do not update u, u_dot, u_dot_dot each time iteration
            z_dot_dot = ((-mw*9.81+fc_n)/(mw))
            z_dot = zlist[1,time_step] + dt/2 * (zlist[2,time_step] + z_dot_dot)
            z = zlist[0,time_step] + dt/2 * (zlist[1,time_step] + z_dot)
            fc = kv * abs(z - N_vec@u)
    ulist[:,time_step+1] = u
    udotlist[:,time_step+1] = u_dot
    udotdotlist[:,time_step+1] = u_dot_dot
    zlist[0,time_step+1] = z
    zlist[1,time_step+1] = z_dot
    zlist[2,time_step+1] = z_dot_dot
    midu.append(z_dot_dot)
    t_new.append(t[time_step])
    print(time_step)

for time_step in range(len(f_pos)):
    if f_pos[time_step] >= L:
        break
    else:
        nf = int(element_number[time_step + 1])
        if nf == num_el:
            nf -= 1
        N_vec = np.zeros(num_nod * 2)
        N_vec[nf * 2:nf * 2 + 4] = N_matrix[time_step + 1]
        N_vec = np.delete(N_vec, [0, -2])
        # initial penetration
        fc_dam = kv*abs(z_dam - N_vec@u_dam)
        fc_n_dam = np.inf
        while abs(fc_dam-fc_n_dam) > tol:
            fc_n_dam = fc_dam
            F_vec_dam = -1 * N_vec * fc_n_dam
            u_dam, u_dot_dam, u_dot_dot_dam = mstdef.newmark(gm, gc_dam, gk_dam, F_vec_dam, gamma, beta, dt, udamlist[:,time_step], udamdotlist[:,time_step], udamdotdotlist[:,time_step])# do not update u, u_dot, u_dot_dot each time iteration
            z_dot_dot_dam = ((-mw*9.81+fc_n_dam)/(mw))
            z_dot_dam = zdamlist[1,time_step] + dt/2 * (zdamlist[2,time_step] + z_dot_dot_dam)
            z_dam = zdamlist[0,time_step] + dt/2 * (zdamlist[1,time_step] + z_dot_dam)
            fc_dam = kv * abs(z_dam - N_vec@u_dam)
    udamlist[:,time_step+1] = u_dam
    udamdotlist[:,time_step+1] = u_dot_dam
    udamdotdotlist[:,time_step+1] = u_dot_dot_dam
    zdamlist[0,time_step+1] = z_dam
    zdamlist[1,time_step+1] = z_dot_dam
    zdamlist[2,time_step+1] = z_dot_dot_dam
    midu_dam.append(z_dot_dot_dam)
    print(time_step)


plt.figure()
plt.title("mass spring model accelerations")
plt.plot(t_new, midu, label = 'wheel mass')
plt.plot(t_new, midu_dam, label = 'damage = 0.6')
plt.axhline(0, color = 'k')
plt.legend()
plt.grid()
plt.xlabel('time [s]')
plt.ylabel('acceleration [m/s^2)')
plt.savefig('mass spring model accelaration damage = 0,6 middle bigger spring constant ')



