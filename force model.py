import matplotlib.pyplot as plt
import numpy as np
import mstdef

L = 25  # [m]
E = 2.87 * 10 ** 9  # e-modulus [N/m^2]
I = 2.9
A = 1  # cross section [m^2]
m = 2303  # self weight of the beam
dam_var = 0.8  # how much the EI will decrease due to damage
num_el = 50  # number of chose elements
L_elm = L / num_el  # length of the elements
place_of_damage = [0, 1]  # which element is damaged
num_nod = num_el + 1  # number of nodes in the model
damage_off_on = False  # setting the damage off and on
f_pos = 2 # position of the force in local coordinate system
mw = 50 # weight of the wheel
mv = 5750  # weight of the train
p = ((mw + mv) * 9.81)

# making force vector
f_vec = np.zeros(num_nod * 2)

# element number where the force is on
element_number = int(np.trunc(f_pos / L_elm))
s = (f_pos - element_number*L_elm)/L_elm
# shape functions
N1 = 1 -3 * s**2 + 2*s**3
N2 = L_elm*(s - 2*s**2 + s**3)
N3 = 3 * s**2 - 2*s**3
N4 = L_elm*(- s**2 + s**3)
N_vec = np.array([N1, N2, N3, N4])

# using shape functions
f_vec[element_number * 2:element_number * 2 + 4] = N_vec * p
f_vec = np.delete(f_vec, [0, -2])

# making stiffness matrix
lk = mstdef.local_stiffness_matrix(E, I, L_elm)
gk = mstdef.global_stiffness_matrix(damage_off_on, num_el, place_of_damage, lk, dam_var, num_nod)
gk = np.delete(gk, [0, -2], 1)
gk = np.delete(gk, [0, -2], 0)


# solving of ku = f
u = f_vec@np.linalg.inv(gk)

u1 = u.reshape(num_nod - 1, 2)[:-1, 1]
u1 = np.concatenate([[0], u1, [0]])
# plotting
x = np.linspace(0, L, num_nod)
plt.plot(x, u1)
plt.show()

