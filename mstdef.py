import numpy as np
import scipy as sc

def local_mass_matrix(A,m,L):
    lm = ((A*m*L)/420) * \
         np.array([[156, 22*L, 54, -13*L],
                   [22*L, 4*L**2, 13*L, -3*L**2],
                   [54, 13*L, 156, -22*L],
                   [-13*L, -3*L**2, -22*L, 4*L**2]])
    return lm

def global_mass_matrix(lm, num_nod, num_el):
    gm = np.zeros((2 * num_nod, 2 * num_nod))
    for i in range(num_el):
        gm[2 * i:2 * i + 4, 2 * i:2 * i + 4] += lm
    return gm

def local_stiffness_matrix(E,I,L):
    lk = ((E*I)/(L**3)) * \
         np.array([[12, 6*L, -12, 6*L],
                   [6*L, 4*L**2, -6*L, 2*L**2],
                   [-12, -6*L, 12, -6*L],
                   [6*L, 2*L**2, -6*L, 4*L**2]])
    return lk

def global_stiffness_matrix(damage_off_on, num_el, place_of_damage, lk, dam_var, num_nod):
    gk = np.zeros((2 * num_nod, 2 * num_nod))
    if damage_off_on:
        print('damage')
        for i in range(num_el):
            if i in place_of_damage:
                gk[2 * i:2 * i + 4, 2 * i:2 * i + 4] += lk * dam_var
            else:
                gk[2 * i:2 * i + 4, 2 * i:2 * i + 4] += lk
    else:
        for i in range(num_el):
            gk[2 * i:2 * i + 4, 2 * i:2 * i + 4] += lk
    return gk

def getting_damping_coeffients(gk,gm,daming_ratio):
    gm = np.delete(gm, [0, -2], 1)
    gm = np.delete(gm, [0, -2], 0)
    gk = np.delete(gk, [0, -2], 1)
    gk = np.delete(gk, [0, -2], 0)
    eig_val, eig_vec = sc.linalg.eigh(gk,gm)
    w1 = np.sqrt(eig_val[0])
    w2 = np.sqrt(eig_val[1])
    s1 = ((2*daming_ratio)/(w1 + w2))*w1*w2
    s2 = ((2 * daming_ratio) / (w1 + w2))
    return s1, s2, w1, w2

def newmark(M, C, K, F, gamma, beta, dt, u , u_dot, u_dot_dot):
    a0 = 1 / (beta * dt ** 2)
    a1 = gamma / (beta * dt)
    a2 = 1 / (beta * dt)
    a3 = (1 / (2 * beta)) - 1
    a4 = (gamma / beta) - 1
    a5 = (dt / 2) * ((gamma / beta) - 2)
    a6 = dt * (1 - gamma)
    a7 = gamma * dt
    K_eff = a0*M + a1*C + K
    F_eff = F + M@(a0*u + a2*u_dot + a3*u_dot_dot) + C@(a1*u + a4*u_dot + a5*u_dot_dot)
    u_dt = np.linalg.inv(K_eff)@F_eff
    u_dt_dot_dot = a0*(u_dt-u) - u_dot*a2 - a3*u_dot_dot
    u_dt_dot = a1*(u_dt-u) + u_dot*a4 + a5*u_dot_dot
    return u_dt, u_dt_dot, u_dt_dot_dot