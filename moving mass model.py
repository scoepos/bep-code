import matplotlib.pyplot as plt
import numpy as np
import mstdef
from timeit import default_timer as timer
start = timer()
L = 25  # [m]
E = 2.87 * 10 ** 9  # e-modulus [N/m^2]
I = 2.9
A = 1  # cross section [m^2]
m = 2303 # self weight of the beam
dam_var = 0.6  # how much the EI will decrease due to damage
num_el = 60  # number of chose elements
L_elm = L / num_el  # length of the elements
#place_of_damage = np.linspace(5,14, 10)
place_of_damage = np.linspace(195,204, 10) # which element is damaged
print(place_of_damage)
num_nod = num_el + 1  # number of nodes in the model
mw = 2500 # weight of the wheel
mv = 3250 # weight of the train
mt = mw + mv
p = ((mt) * 9.81)  # force of the train
damping_ratio = 0.025  # damping ratio for this beam
beta, gamma = 0.25, 0.5  # Newmark's beta method
kv = 1595*10**3  # spring inbetween wheel and train
cv = 85*10**3
kb = 4*10**9
Mt = np.array([[mv,0],[0,mw]])
Kt = np.array([[-kv,kv],[kv,-kv]])
Ct = np.array([[-cv, cv], [cv, -cv]])

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
gc = s1*gm + s2*gk
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
z_dam = np.zeros(2)
z_dot_dam = np.zeros(2)
z_dot_dot_dam = np.zeros(2)


#making the place vector
t1 = L/30
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
f = 0
f_n = np.inf
midu = []
g = 9.81
flist = []
nf = int(element_number[0])
N_vec = np.zeros(num_nod * 2)
N_vec[nf * 2:nf * 2 + 4] = N_matrix[0]
N_vec = np.delete(N_vec, [0, -2])
u = np.linalg.inv(gk)@(-N_vec*p)
ulist = np.zeros([num_nod*2 - 2,len(t)])
udotlist = np.zeros([num_nod*2 - 2,len(t)])
udotdotlist = np.zeros([num_nod*2 - 2,len(t)])
ulist[:,0] = u
zlist = np.zeros([2,len(t)])
zdotlist = np.zeros([2,len(t)])
zdotdotlist = np.zeros([2,len(t)])
#starting time integration
for n in range(len(t)):
    f_n = np.inf
    nf = int(element_number[n])
    z[1], z_dot[1], z_dot_dot[1] = u @ N_vec, u_dot @ N_vec, u_dot_dot @ N_vec
    f = p
    if nf == num_el:
        nf -= 1
    while np.abs(f - f_n) > 100:
        f_n = f
        N_vec = np.zeros(num_nod * 2)
        N_vec[nf * 2:nf * 2 + 4] = N_matrix[n]
        N_vec = np.delete(N_vec, [0, -2])
        f1 = np.array([0,f])
        F_vec = -f*N_vec
        u, u_dot, u_dot_dot = mstdef.newmark(gm, gc, gk, F_vec, gamma, beta, dt, ulist[:,n], udotlist[:,n], udotdotlist[:,n])
        z, z_dot, z_dot_dot = mstdef.newmark(Mt, Ct, Kt, f1, gamma, beta, dt, zlist[:,n], zdotlist[:,n], zdotdotlist[:,n])
        f = kb*abs(z[0] - N_vec@u)
        print(f)
    else:
        ulist[:, n + 1] = u
        udotlist[:, n + 1] = u_dot
        udotdotlist[:, n + 1] = u_dot_dot
        zlist[:, n + 1] = z
        zdotlist[:, n + 1] = z_dot
        zdotdotlist[:, n + 1] = z_dot_dot
        midu.append(u[61])
        print(n)


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
    midu_dam.append(u_dam[51])
    print(time_step)

plt.figure(figsize=(10,7))
plt.plot(t, ulist, label='without damage')
plt.axhline(0, color='k')
plt.title('displacement of the middle node of longer bridge')
plt.legend()
plt.grid()
plt.xlabel('time')
plt.ylabel('displacement')
plt.savefig('mass damping model with 200m bridge')

end = timer()
print(end - start)
