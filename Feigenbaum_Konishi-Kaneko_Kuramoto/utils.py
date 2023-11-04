import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore, Style
from maps import *



def map_plot(x_0,param,Map=Logistic_map_i,t_int=[0,1],n=None,f_i=None,ax=None,cobweb=True, points=True):
    
    t = np.linspace(t_int[0], t_int[1], 200)
    f = Map(t, param)
    ax.plot(t, f, color='C0', lw=2, label='$f(x)$')
    if f_i:
        if 1 in f_i:
            f_i.remove(1)
        if f_i:
            for i in f_i:
                F = Map(t,param,i)
                ax.plot(t, F, color=f'C{i}', lw=2, label=f'$f^{{{i}}}(x)$', linestyle='dotted')
        else:
            F = [0]
    ax.plot([t_int[0], t_int[1]], [t_int[0], t_int[1]], 'k', lw=2,linestyle='dashed')
    
    x = x_0
    if cobweb or points:
        for i in range(n):
            y = Map(x, param)
            if cobweb:
                ax.plot([x, x], [x, y], 'k', lw=1)
                ax.plot([x, y], [y, y], 'k', lw=1)
            if points:
                ax.plot([x], [y], 'ok', ms=10, alpha=(i + 1) / n)
            x = y

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=13)
    ax.set_title(f"$\lambda={param:.3f}$")
    
#     return f, F
    
def Logistic_map_bif_diagram(n,ax1,ax2):
    
    λ_c = np.array([3.,1+np.sqrt(6),3.54409035,3.569945672])
    λ = np.linspace(2.5,4,n)
    x = 0.01 * np.ones(n)
    
    σ = np.zeros(n)

    j = 1000
    k = 10
    
    for i in range(j):
        x = Logistic_map(x, λ)
        σ += np.log(abs(λ-2*x*λ))
        if i >= (j - k):
            ax1.plot(λ, x, ',k')
        
    ax1.vlines(λ_c,0,1,ls='--')
    ax1.axhline(0.5, ls='--', color='black')
    ax1.set_ylim([0, 1])
    ax1.set_ylabel('$x_n$',fontsize=15)
    ax1.set_xlabel('$\lambda$',fontsize=15)
    
    
    ax2.axhline(0, ls='--',alpha=0.5,color='black')
    ax2.plot(λ, σ / j,'.',ms=.5 )
    ax2.set_ylabel('$\sigma$',fontsize=15)
    ax2.set_xlabel('$\lambda$',fontsize=15)
    
    
def one_dim_map_bif_diagram(Map,n,param_range,init,ax):
    λ = np.linspace(param_range[0],param_range[1],n)
    x = init * np.ones(n)
    
    j = 1000
    k = 10
    
    for i in range(j):
        x = Map(x, λ)
        if i >= (j - k):
            ax.plot(λ, x, ',k')
    
def bruteforce_finding_superstable(λ_start=1.5, λ_end=3.6, x_0=0.01, n=500, 
                                   j=1, j_max = 512, step=100, λ_add=0.001, acc=0.0001):
    periods = np.array([])
    λ_superstable = np.array([])
    points_superstable = [] 

    while j != 2*j_max and λ_start < λ_end:
        print(f'{λ_start:.12f}', j, end='\r')
        if j >= 16 and j < 128:
            step = 2
            λ_add = 0.0000001
            acc = 0.00001
            n = 5000
            λ_s = np.linspace(λ_start,λ_start + λ_add, step)[-1]

        if j >= 128:
            step = 2
            λ_add = 0.00000001
            acc = 0.00001
            n = 5000
            λ_s = np.linspace(λ_start,λ_start + λ_add, step)[-1]

        λ_s = np.linspace(λ_start,λ_start + λ_add, step)
        index_tmp = np.array([])

        for λ in λ_s:
            x = x_0
            for i in range(n):
                x = Logistic_map(x, λ)

                if i > 8 * n / 10:
                    if np.abs(x - 0.5) < acc:
                        index_tmp = np.hstack((index_tmp,np.array([i])))
                    if len(index_tmp) == 2 and np.diff(index_tmp) == j:
                        λ_superstable = np.hstack((λ_superstable, λ))
                        periods = np.hstack((periods, j))
                        print(Fore.GREEN + '\033[1m Parameter \033[0m' + Style.RESET_ALL + f'\033[1m λ={λ:.6f} \033[0m' +  
                              Fore.GREEN + '\033[1m Period \033[0m' + Style.RESET_ALL + f'\033[1m {j} \033[0m')
                        j = 2*j

                        break
                    else:
                        continue
        λ_start = λ
        
    return periods, λ_superstable

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def d_i(x,period,n,acc=0.0001):
    if period >= 16:
        acc = 0.00001
    x = np.asanyarray(x)
    I = np.array([],dtype=np.int64)
    for i in range(n):
        if np.abs(x[i]-0.5) < acc:
            I = np.hstack((I,np.array([i])),dtype=np.int64)
    nearest = find_nearest(x[I[-2]+1:I[-1]],x[I[-2]])
    d = np.abs(x[I[-2]] - nearest)
    return d, nearest


def nearest_points(a,b,i):
    a = np.asarray(a)
    b = np.asarray(b)
    Δ = np.abs(a - b)
    Ind = np.argsort(Δ)
    
    return Ind[:,:i], a[np.arange(a.shape[0])[:,None],Ind[:,:i]]

def superstable_similar_interval(i=10,n=30000):
    distance = np.array([])
    nearest = np.array([])
    for λ, period in zip(superstable_λ_s[1:i],superstable_periods[1:i]):
    
        x = Logistic_map_seq(n,0.1,λ)
        d, x_near = d_i(x,period,n)
        distance = np.hstack((distance,d))
        nearest = np.hstack((nearest,x_near))
    
    X = np.linspace(0.5*np.ones_like(nearest),nearest,n).T
    f_i = np.array([])
    for k, (p, λ) in enumerate(zip(superstable_periods[1:i],superstable_λ_s[1:i])):
        f = Logistic_map_i(X[k],λ,p)
        f_i = np.concatenate((f_i,f),axis=-1)

    f_i = f_i.reshape(X.shape[0],-1)
    index, val = nearest_points(f_i,X,3)
    x_for_g = np.sort(val)[:,1]
    
    Interval = np.sort(np.concatenate((x_for_g[:,None],(1-x_for_g)[:,None]),axis=-1))
    
    return Interval

def f_i_transform(i=10,n=30000,α=-2.5029):
    Interval = superstable_similar_interval(i,n)
    Transformed_f_i = []
    
    for m, (p,λ,interval) in enumerate(zip(superstable_periods[1:i],superstable_λ_s[1:i],Interval)):
        t = np.linspace(interval[0], interval[1],5000)
        x = np.linspace(0,1,5000)
        transformed_f_i = (α ** (m + 1)) * (Logistic_map_i(t,λ,p)-0.5)
        Transformed_f_i.append(transformed_f_i)
        
        
    return Transformed_f_i



def Bif_diagram_2param(Map,a,b,P,iteration_count=300,Δ=1e-3):
    part = int(iteration_count * 8 / 10)
    x_0,y_0 = 0.001,0.001
    
    
    params_cartesian = np.transpose([np.tile(a, len(a)), np.repeat(b, len(b))])
    Params_dict ={f'{key}':[] for key in P}
    
    for params in params_cartesian:
        a, b = params
        x, y = Map(iteration_count,x_0,y_0,a,b)
        x, y = x[part:],y[part:]

        if np.any(np.isinf(x)) or np.any(np.isnan(x)):

            Params_dict['0'].append([a,b])
        else:
            for p in P[1:]:
                if np.abs(x[0] - x[p]) < eps:
                    Params_dict[f'{p}'].append([a,b])
                    break
    return Params_dict



# def intersec(a,b,tol=1e-5):
#     ind_b = np.where((np.abs(a[:,None] - b) < tol).any(0))
#     ind_a = np.where((np.abs(b[:,None] - a) < tol).any(0))
    
#     return ind_b,ind_a

# def Logistic_map_sym(x=sympy.Symbol('x'),λ=sympy.Symbol('lambda') ):
#     return λ * x * (1 - x)


# def Logistic_map_sym_n(n):
#     f = Logistic_map_sym()
    
#     f_i = [f]
    
#     for i in range(n-1):
#         f_i.append(f_i[i].subs(x,Logistic()))
    
#     return f_i

# def Polynomial_coeffs(n):
#     F = Logistic_map_sym_n(n)
#     Poly_coeffs_F = [sympy.Poly(F_i,x).all_coeffs() for F_i in F]       
    
#     return Poly_coeffs_F



superstable_periods = np.array([1,2,4,8,16,32,64,128,256])

superstable_λ_s = np.array([1.9996060606060055,
                            3.2357878787876877,
                            3.49840404040382,
                            3.5545555555553294,
                            3.5666626999806867,
                            3.569241099976467,
                            3.569793999975562,
                            3.5699127499748404,
                            3.5699383999746845,
                            3.5699440299746503])


