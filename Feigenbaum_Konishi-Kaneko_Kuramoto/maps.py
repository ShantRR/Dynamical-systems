import numpy as np
import networkx as nx



def KK_map(k,p_0,x_0,connection_matrix):
    x_i = x_0.reshape(-1,1)
    x_matrix = np.tile(x_0,(len(x_0),1))
    N = len(x_i)
    coeff = k#/(2*np.pi*np.sqrt(N-1))
    p = (p_0 + coeff*(connection_matrix*np.sin(2*np.pi*(x_matrix-x_i))).sum(axis=-1))# % 1
    x = (x_0 + p) % 1
    return p,x
    
def KK_map_seq(N,n,k,p_0=None,x_0=None):
    if not(p_0.any()):
        p_0 = 0.01*np.ones(N)
    if not(x_0.any()):
        x_0 = np.random.rand(N)
    
    G = nx.complete_graph(N)
    connection_matrix = nx.to_numpy_array(G)
    p = p_0[None,:]
    x = x_0[None,:]
    for i in range(n-1):
        next_val = KK_map(k,p[i],x[i],connection_matrix)
        p = np.concatenate((p,next_val[0][None,:]))
        x = np.concatenate((x,next_val[1][None,:]))
    return p,x


###################################################################################


def Standard_map(p_0,x_0,k):
    p = (p_0 + (k/(2*np.pi)) * np.sin(2*np.pi*x_0))%1  # p = (p_0 + k * np.sin(x_0))%(2*np.pi)
    x = (x_0 + p)%1                       # x = (x_0 + p)%(2*np.pi)
    return p, x

def Standard_map_seq(n,p_0,x_0,k):
    p_s = [p_0]
    x_s = [x_0]
    for i in range(n-1):
        sol = Standard_map(p_s[i],x_s[i],k)
        p_s.append(sol[0])
        x_s.append(sol[1])
        
    return p_s, x_s

def Standard_map_phase(n,p_0,x_0,k):
    p_0,x_0 = np.meshgrid(p_0,x_0)
    init_group = np.array((p_0.ravel(), x_0.ravel())).T
    init_group = init_group.reshape(count,-1,2)
                   
    p = []
    x = []

    for i, i_group in enumerate(init_group):
        p.append([])
        x.append([])

        for init in i_group:
            sol = Standard_map_seq(n,init[0],init[1],k)
            p[i].append(sol[0])
            x[i].append(sol[1])
            
    return p,x

###################################################################################



def Henon_map(x_0,y_0,a,b):
    x = y_0 + 1 - a * x_0**2
    y = b * x_0
    return x, y

def Henon_map_seq(n,x_0,y_0,a,b=0.3):
    x_s = [x_0]
    y_s = [y_0]
    for i in range(n-1):
        sol = Henon_map(x_s[i],y_s[i],a,b)
        x_s.append(sol[0])
        y_s.append(sol[1])
        
    return x_s, y_s


####################################################################################


def Logistic_map(x_0,λ):
    x = λ * x_0 * (1 - x_0)
    return x

def Logistic_map_seq(n,x_0,λ):
    x_s = [x_0]
    for i in range(n-1):
        sol = Logistic_map(x_s[i],λ)
        x_s.append(sol)
    return x_s


def Logistic_map_i(x_0,λ,i=None):
    f = Logistic_map(x_0,λ)
    if i:
        for _ in range(i-1):
            f = Logistic_map(f,λ)
    return f

def exp_map(x_0,λ):
    x = x_0 * np.exp(-λ * (1 - x_0))
    return x

def sin_map(x_0,λ):
    x = λ * np.sin(np.pi * x_0)
    return x

def cos_map(x_0,λ):
    x = λ * np.cos(x_0)
    return x

####################################################################################


def Exampe_Tangential_Map(x_0,λ):
    x = λ + x_0 - x_0 ** 2
    return x

def Exampe_Pitchfork_Map(x_0,λ):
    x = λ * x_0 - x_0 ** 3
    return x

def Exampe_Pitchfork_reverse_Map(x_0,λ):
    x = λ * x_0 + x_0 ** 3
    return x
    