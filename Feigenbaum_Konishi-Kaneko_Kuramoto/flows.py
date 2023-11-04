import numpy as np
from scipy.integrate import odeint


######Hamiltonian systems#######

def Toda_U(x,y):
    U = (1 / 24) * (np.exp(2 * y + 2 * np.sqrt(3) * x) + np.exp(2 * y - 2 * np.sqrt(3) * x) + np.exp(-4 * y)) - 1 / 8
    return U

def HH_U(x,y):
    U = (x ** 2 + y ** 2) / 2 + (x ** 2) * y - (y ** 3) / 3
    return U

def HH_H(x,y,p_x,p_y):
    H = (p_x ** 2 + p_y ** 2) / 2 + U(x,y)
    return H

def HH_canonic_eq(Y, t):
    x, y, p_x, p_y = Y
    dydt = [p_x, p_y, -x - 2 * x * y, -y - x**2 + y**2]
    return dydt

def Init_conds(x_0,y_0,p_x_0,E):
    x_0, y_0,p_x_0 = np.meshgrid(x_0,y_0,p_x_0)
    init_group = np.array((x_0.ravel(), y_0.ravel(), p_x_0.ravel())).T
    init_group = init_group.reshape(len(x_0),-1,3)
    
    p_y_0 = np.sqrt(2 * (E - HH_U(init_group[:,:,0],init_group[:,:,1])) - init_group[:,:,2] ** 2)
    p_y_0 = p_y_0[:,:,None]    
    init_group = np.concatenate((init_group,p_y_0),axis=-1)
    init_group = init_group[ ~np.isnan(init_group).any(axis=-1)]
    
    return init_group

def Sol(init_group):
    t = np.linspace(0, 1000, 10000)
    Sol = []
    for i in init_group:
        sol = odeint(HH_canonic_eq, i, t)
        Sol.append(sol)
    return np.asarray(Sol)

def Poincare_section_y_py(Solution):
    p_y_cross = []
    y_cross = []
    for j in range(Solution.shape[0]):
        p_y_cross.append([])
        y_cross.append([])
        for i in range(Solution.shape[1]-1):
            if (Solution[j,i,0] <= 0 and Solution[j,i+1,0] >= 0) or (Solution[j,i,0] >= 0 and Solution[j,i+1,0] <= 0):
                c = -Solution[j,i,0] / (Solution[j,i+1,0] - Solution[j,i,0])
                rx, ry = (1 - c) * Solution[j,i,1] + c * Solution[j,i+1,1], (1 - c) * Solution[j,i,-1] + c * Solution[j,i+1,-1]
                y_cross[j].append(rx); 
                p_y_cross[j].append(ry);
    return y_cross, p_y_cross


######dissipative systems#######

def Lorenz(X, t, s, b, r):
    x,y,z = X
    x_d = s*(y - x)
    y_d = r*x - y - x*z
    z_d = x*y - b*z
    return x_d, y_d, z_d
