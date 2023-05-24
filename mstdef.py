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