import networkx as nx
import matplotlib
import matplotlib.pyplot as plt 
import numpy as np
from scipy.integrate import odeint
import matplotlib.animation as animation
plt.rcParams["animation.html"] = "jshtml"
matplotlib.rcParams['animation.embed_limit'] = 2**128





class Kuramoto():
    def __init__(self, N, K, omega={'Normal': [3,1]}):
        
        self.N = N
        self.K = K
        
        self.G = nx.complete_graph(self.N)
        self.G_matrix = nx.to_numpy_array(self.G)
#         self.omega = np.random.uniform(0.9, 1.1, self.N) #np.random.uniform(0.9, 1.1, self.N) #np.random.normal(mean=0, std=1, self.N)
        if list(omega.keys())[0] == 'Normal':
            mean = list(omega.values())[0][0]
            std = list(omega.values())[0][1]
            self.omega = np.random.normal(mean, std, self.N)
            pde_0 = np.exp(-((mean/std**2)**2)/2)/((std**2)*np.sqrt(2*np.pi))
            K_crit = 2/(np.pi*pde_0)
        self.theta_0 = 2*np.pi*np.random.rand(self.N)
        
        fig = plt.figure(figsize=(15,5))
        ax1 = plt.subplot(122)
#         nx.draw(self.G, with_labels=True, font_weight='bold')
        ax2 = plt.subplot(121)
        ax2.hist(self.omega,bins=25)
        ax2.set_ylabel('$g(\omega)$', fontsize=15)
        ax2.set_xlabel('$\omega$', fontsize=15)
        
        print(f'Critical value of strgth ---> {K_crit}')
        
    def motion(self, theta, t, *args):
        
        omega = args[0]
        K     = args[1]
        Conect_matrix = args[2]
        
        theta_matrix = np.tile(theta,(len(theta),1))
        theta_i = theta.reshape(-1,1)

        d_theta = omega + (self.K/self.N)*(Conect_matrix*np.sin(theta_matrix-theta_i)).sum(axis=-1)
        
        return d_theta
        
    def phase_coherence(self,theta):
        suma = sum([(np.e ** (1j * i)) for i in theta])
        return abs(suma / len(theta))
    def psi(self,theta):
        suma = sum([(np.e ** (1j * i)) for i in theta])
        return np.angle(suma/len(theta))


        
    def integration(self, T, dt):
        
        t = np.arange(0, T, dt)
        
        args = (self.omega, self.K, self.G_matrix)
        
        sol_no_modulo = odeint(self.motion,self.theta_0,t,args).T
#         plt.clf()

        
        
        sol = sol_no_modulo % 2*np.pi
        sol = np.asarray(sol)
        psi = self.psi(sol)
        r_module = self.phase_coherence(sol)
        rx = r_module*np.cos(psi)
        ry = r_module*np.sin(psi) 
        r_vec = np.vstack((rx, ry)).T
        
        fig2 = plt.figure(figsize=(15,5))
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)
        for i in range(self.N):
            ax1.plot(t[1:], np.diff(sol_no_modulo[i,:]))
            ax2.plot(t,sol[i,:])
        ax1.set_xlabel('$t$', fontsize=15)
        ax1.set_ylabel('$\\dot{\\theta}_i$', fontsize=15)
        ax2.set_xlabel('$t$', fontsize=15)
        ax2.set_ylabel('$\\theta_i$', fontsize=15)
        fig2.tight_layout()
        plt.show()
        return sol, t, r_vec
    
    def motion_animation(self,sol,r,save=True,save_path='motion_videos/'):
        
        f,ax=plt.subplots(figsize=(6,6),dpi=80)
        ax.set_xlim([-1.1,1.1])
        ax.set_ylim([-1.1,1.1])

        Ox = np.cos(sol).T
        Oy = np.sin(sol).T
        
        r_module = np.sqrt(r[:,0]**2 + r[:,1]**2)

        line,=ax.plot(Ox[0][:], Oy[0][:],'o', alpha=0.2, label='$\\theta_{i}$')
        tail = np.zeros((2,self.N))
        quiver = ax.quiver(*tail, r[0,0], r[0,1], scale=1, scale_units='xy', angles='xy', label=f'$r={r_module[0]:.3f}$')
        L = ax.legend(loc=1,fontsize=12)
        def animate(i, quiver, *tail):
            lab = f'$r={r_module[i]:.3f}$'
            line.set_xdata(Ox[i][:])
            line.set_ydata(Oy[i][:])
            
            rx, ry = r[i,0], r[i,1]
            quiver.set_UVC(rx,ry)
            L.get_texts()[1].set_text(lab)
            return line, quiver

        ani = animation.FuncAnimation(f, animate, frames=np.arange(0,len(r)), fargs=(quiver, *tail), interval=30)
        phi=np.arange(0,2.05*(np.pi),0.1)
        plt.plot(np.cos(phi),np.sin(phi),'k', linestyle='dashed', alpha=0.1)
        plt.show()
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, bitrate=600)
        ani
        if save:
            print('Start saving the video')
            ani.save(save_path+f'count={self.N}_K={self.K}.avi', writer=writer)
            print('Video saved    ' + save_path+f'count={self.N}_K={self.K}.avi' )
            return ani
        else:
            return ani

