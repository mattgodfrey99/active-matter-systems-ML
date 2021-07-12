import matplotlib.pyplot as plt 
from matplotlib import animation
import numpy as np
import random
import time

np.seterr(divide='ignore', invalid='ignore')
# ignore division by 0 error, the value is not used so there is no issue

def force(pos, dimensions):
    
    force_total_shift = np.zeros([len(pos),2])
    # array for full force on every particle
    
    for i in range(len(pos)):
        
        delta_r_0 = pos[:,0:2] - pos[i,0:2]
        # difference in coordinates
        
        delta_r = np.abs(delta_r_0)
        # absolute difference in coordiantes
    
        delta_r = np.where(delta_r > 0.5 * dimensions, np.sign(delta_r_0)*(delta_r - dimensions), delta_r_0)
        # smallest difference of coordinates
    
        r_abs = np.sqrt((delta_r ** 2).sum(axis=-1))
        # absolute value of distance
    
        ind = [k for k,j in enumerate(r_abs) if j <= (2**(1/6)) and j > 0]
        # indicies of those to find force for
    
        direction = -np.sign(delta_r)[ind]
        # tells which way to push particle
    
        theta = (np.arctan(abs(np.divide(delta_r[:,1][ind],delta_r[:,0][ind]))))
        # all theta values (arctan(y/x)) relative to particles
    
        F = (abs(24*(2-(r_abs[ind]**6))/(r_abs[ind]**(13))))
        # strength of force for our particle on every other particle
      
        x_shift = np.sum((F*np.cos(theta))*direction[:,0])
        # x direction shift of particle
    
        y_shift = np.sum((F*np.sin(theta))*direction[:,1])
        # y direction shift of particle
        
        force_total_shift[i,:] = [x_shift,y_shift]

    return force_total_shift

density = 0.4
# density of particles in box

L = 5
# length of box

dimensions = np.array([L, L])
# size of box

n = int(density * (L**2))
# no. particles 

dt = 0.005                                             
# time steps

t = 30000
# placeholder for loop numbers

v_0 = 1
# bare self-propulsion speed

sigma = 1
#size of particles

l_p = 60
#persistence length

D_r = sigma/l_p
# rotational diffusivity

D = (D_r*sigma**2)/3                                                
# translational diffusivity

mu = D/(D_r*sigma**2)
# particle mobility

w_a = np.zeros(t)
# active work array

pos = (pos_tot[3000,:,:]).copy()
# set inital position to some positions where flocking/clustering is occuring

pos_tot = np.zeros([t,n,3])
# array for all positions, with theta in third column

time_start = time.time()

for i in range(t):
    
    print(i)
    
    noise_theta = np.zeros((n,1))
    
    for j in range(n):      
        
        placeholder = pos.copy()
        
        delta_r_0 = placeholder[:,0:2] - placeholder[j:j+1,0:2]
        # relative x,y,theta coordinates
        delta_r = np.abs(delta_r_0)
        # absolute distance between points
        placeholder_new = np.where(delta_r > 0.5 * dimensions, np.sign(delta_r_0)*(delta_r - dimensions), delta_r_0)
        # periodic boundary conditions check
        placeholder_new = np.append(placeholder_new, placeholder[:,2:3] - placeholder[j:j+1,2:3],1)
        # add back change in theta 
        placeholder_new = np.delete(placeholder_new,j,0)
        
        
        relatives = np.sqrt(np.sum((placeholder_new[:,0:2])**2,1))
        relatives = np.expand_dims(relatives,1)
        placeholder_new = np.append(placeholder_new,relatives,1)  
        # distances from each particle to target particle
        
        #signs = np.sign(placeholder_new[:,1:2])
        #placeholder_new = np.append(placeholder_new,signs,1)
        
        
        placeholder_new[:,2][placeholder_new[:,2]>(np.pi)] = placeholder_new[:,2][placeholder_new[:,2]>(np.pi)] - (2*np.pi)
        placeholder_new[:,2][placeholder_new[:,2]<(-np.pi)] = placeholder_new[:,2][placeholder_new[:,2]<(-np.pi)] + (2*np.pi)
        # fixing value of theta
        
        #close = (placeholder_new[:,3:4] <= 2**(1/6))*1
        #placeholder_new = placeholder_new*close
        #setting everything else to 0
        
        placeholder_new[:,0:2] = placeholder_new[:,0:2]/(L/2)
        placeholder_new[:,2:3] = placeholder_new[:,2:3]/(np.pi)
        placeholder_new[:,3:4] = (placeholder_new[:,3:4]/((L/2)*np.sqrt(2))*2)-1
        placeholder_new = placeholder_new.reshape((-1, 4*(n-1)))
        # normalisation and reshaping
        
        close = (placeholder_new[:,3:4] <= -0.3650395792)*1
        placeholder_new = placeholder_new*close
        placeholder_new[placeholder_new == 0] = 0   
        
        noise_predict = model.predict(placeholder_new)
        #print(noise_predict)
        noise_predict = bins[int(abs(np.round(noise_predict*(bin_no)))-1)]
        
        #if abs(noise_predict)<0.00005:
        #    noise_predict = 0
            
        noise_theta[j] = noise_predict
        
       
    
    
    LJ_force = force(pos, dimensions)
    # force form LJ pot
    
    noise_trans = np.sqrt(2*D)*np.random.normal(0,1,[n,2])
    # random translational noise
    
    velocity_term = v_0*(np.stack((np.cos(pos[:,2]),np.sin(pos[:,2])),1))
    # shift due to velocity of particle

    pos[:,0:2] = pos[:,0:2] + (mu*LJ_force*dt) + (noise_trans*np.sqrt(dt)) + (velocity_term*dt)
    # each generation having repulsive force + random motion
    
    u_i = np.transpose(np.array([np.cos(pos[:,2]),np.sin(pos[:,2])]))
    # orientation vector
    
    w_a[i] = np.sum((mu*LJ_force*u_i + noise_trans*u_i + velocity_term*u_i))*(dt*v_0/mu)
    # active work
    
    pos[:,0:2] = pos[:,0:2] % dimensions
    # periodic boundary conditions
    
    pos[:,2:3] = pos[:,2:3] + 3*noise_theta 
    # change of theta
    
    pos[:,2:3] = pos[:,2:3] % (2*np.pi)
    # periodic boundary conditions
    
    pos_tot[i,:,0:2] = pos[:,0:2]
    # append positions to final array
    
    pos_tot[i,:,2:3] = pos[:,2:3]
    # append theta values to final array
    
    
time_end = time.time()
time_total = time_end-time_start

## Animation #################################################################

fig = plt.figure(figsize=(7,7))
ax = plt.axes(xlim=(0,L),ylim=(0,L))
scatter=ax.scatter(pos_tot[0,:,0], pos_tot[0,:,1], s=sigma*(85**2), marker='o', edgecolors='k', alpha=0.8)
quiv = ax.quiver(pos_tot[0,:,0], pos_tot[0,:,1], np.cos(pos_tot[0,:,2]), np.sin(pos_tot[0,:,2]))
ttl = ax.text(0.25, 1.05, 'Brownian Motion', transform = ax.transAxes)
ax.set_xticks(np.arange(0, L, 1))
ax.set_yticks(np.arange(0, L, 1))
plt.grid()

def connect(i):
    
    step_size = 20
    
    quiv.set_offsets(pos_tot[step_size*i,:,0:2])
    quiv.set_UVC(np.cos(pos_tot[step_size*i,:,2]), np.sin(pos_tot[step_size*i,:,2]))
    #update arrow positions and direction respectively
    
    scatter.set_offsets(pos_tot[step_size*i,:,0:2])
    #update particle positions
    
    ttl.set_text(('Brownian Motion - "Time": ', round(step_size*i*dt, 3)))
    # update text
    
    return scatter,   
    return quiv,

ani = animation.FuncAnimation(fig, connect, np.arange(0, t-1), interval=1)
plt.show()    

Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=800)
ani.save('normal_flocking.mp4', writer=writer)

## Active Work ###############################################################

w_a_cum = np.cumsum(w_a)
w = (w_a_cum*mu)/(n*dt*np.array([range(1,t+1)])*v_0**2)
# normalised active work

# normalised active work plot

plt.figure(0)
plt.plot(np.linspace(0,t*dt,t), np.transpose(w))
plt.title('Normalised Active Work')
plt.ylabel('Normalised Active Work')
plt.xlabel('"Time"')
#plt.legend(['', ''], loc='best')
#plt.axhline(y=0.5, color='k', linestyle='--')



