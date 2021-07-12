import numpy as np
import random
import matplotlib.pyplot as plt 
from matplotlib import animation
import time

time_start = time.time()

np.seterr(divide='ignore', invalid='ignore')
# ignore division by 0 error, the value is not used so there is no issue

def initial_pos(L,n):
    
    x = np.repeat(np.arange(0, L), L)
    y = np.tile(np.arange(0,L), L)
    xy = np.transpose(np.array([x,y]))
    # all possible coordinates
    
    ind = random.sample(range(0,L**2),n)
    coords = np.squeeze(xy[ind])
    # chosen coordinates
    
    pos = np.zeros([n,3])
    pos[:,0:2] = coords
    # initial positions of particles
    
    pos[:,2] = np.random.uniform(0,1,[n])*2*np.pi
    #intial theta values of particles
    
    return pos

def force(pos, dimensions):
    
    force_total_shift = np.zeros([len(pos),2])
    # array for full force on every particle
    
    B_i_total = np.zeros([len(pos),1])
    # array for change in angle
    
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
        
        e_i = (-delta_r[ind])/(np.expand_dims(r_abs[ind],1))
        # unit vectors between particle and all others

        B_i = np.sum(e_i * [-np.sin(pos[i,2]),np.cos(pos[i,2])])
        # term to shift particle path to flock
        
        force_total_shift[i,:] = [x_shift,y_shift]
        
        B_i_total[i] = B_i

    return force_total_shift, B_i_total
   
            
density = 0.4
# density of particles in box (change so density and n are picked, not L)
            
n = 10
# no. particles

L = int(np.sqrt(n/density))
# length of 'box' 
# INTEGER ATM AS I WILL NEED TO REDO HOW INTIAL PARTICLES POS. MADE
            
dimensions = np.array([L, L])
# size of box
            
dt = 0.005                                             
# time steps
            
t = 5000
# placeholder for loop numbers
            
buffer = 2000
# initial time for initiating system
            
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
            
g = 1
# + for flocking, - for clustering, 0 for normal

train_images_o = np.zeros([1,n,3])
train_labels_o = np.zeros([1,n,1])
# empty arrays to get one large dataset

starting = 0
# when data has 'settled'

loops = 50


for j in range(loops):
    
    print(j)
    
    pos = initial_pos(L,n)
    # use above function for intital coordinates
                                
    pos_tot = np.zeros([t,n,3])
    # array for all positions, with theta in third column
    
    delta_theta = np.zeros([t,n,1])
    
    w_a = np.zeros(t)
    # active work array
    
    for i in range(buffer):
        
        LJ_force, B_i_tot = force(pos, dimensions)
        # force form LJ pot. and change in theta
        
        noise_trans = np.sqrt(2*D)*np.random.normal(0,1,[n,2])
        # random translational noise
        
        noise_theta = np.sqrt(2*D_r)*np.random.normal(0,1,[n,1])
        # random rotational noise
        
        velocity_term = v_0*(np.stack((np.cos(pos[:,2]),np.sin(pos[:,2])),1))
        # shift due to velocity of particle
    
        pos[:,0:2] = pos[:,0:2] + (mu*LJ_force*dt) + (noise_trans*np.sqrt(dt)) + (velocity_term*dt)
        # each generation having repulsive force + random motion
        
        pos[:,0:2] = pos[:,0:2] % dimensions
        # periodic boundary conditions
        
        pos[:,2:3] = pos[:,2:3] + (noise_theta*np.sqrt(dt))
        # change of theta
        
        pos[:,2:3] = pos[:,2:3] % (2*np.pi)
        # periodic boundary conditions
    
    for i in range(t):
        
        LJ_force, B_i_tot = force(pos, dimensions)
        # force form LJ pot. and change in theta
        
        noise_trans = np.sqrt(2*D)*np.random.normal(0,1,[n,2])
        # random translational noise
        
        noise_theta = np.sqrt(2*D_r)*np.random.normal(0,1,[n,1])
        # random rotational noise
        
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
        
        theta_term = (0*noise_theta*np.sqrt(dt)) + (g*B_i_tot*dt)
        
        delta_theta[i] = theta_term
        
        pos[:,2:3] = pos[:,2:3] + theta_term
        # change of theta
        
        pos[:,2:3] = pos[:,2:3] % (2*np.pi)
        # periodic boundary conditions
        
        pos_tot[i,:,0:2] = pos[:,0:2]
        # append positions to final array
        
        pos_tot[i,:,2:3] = pos[:,2:3]
        # append theta values to final array
        
    
    train_images_o = np.concatenate((train_images_o,pos_tot[starting:-1,:,:]), 0)
    train_labels_o = np.concatenate((train_labels_o,delta_theta[starting+1:,:,0:1]), 0)
    # concatenate all different trajectories
                                   
    
train_images_o = np.delete(train_images_o,0,0)
train_labels_o = np.delete(train_labels_o,0,0)
# remove inital 0 layer

skip_size = 100
train_images_o = train_images_o[1::skip_size]
train_labels_o = train_labels_o[1::skip_size]
# remove skip_size elements inbetween array, so less of same data

train_images = np.zeros([1,n-1,3])
train_labels = []
# empty arrays for final data

for i in range(n):
    # 'i' is the target particle
    
    delta_r_0 = train_images_o[:,:,0:2] - train_images_o[:,i:i+1,0:2]
    # relative x,y,theta coordinates
    delta_r = np.abs(delta_r_0)
    # absolute distance between points
    train_images_p = np.where(delta_r > 0.5 * dimensions, np.sign(delta_r_0)*(delta_r - dimensions), delta_r_0)
    # periodic boundary conditions check
    train_images_p = np.append(train_images_p,train_images_o[:,:,2:3] - train_images_o[:,i:i+1,2:3],2)
    # add back change in theta 
    train_labels_p = train_labels_o[:,i]
    # single label of theta 
    train_images_p = np.delete(train_images_p,i,1)
    # remove redundant 0,0,0
    
    train_images = np.concatenate((train_images,train_images_p), 0)
    train_labels = np.append(train_labels, train_labels_p)
    # concatenate trajectories for all target particles

train_images = np.delete(train_images,0,0)
# remove inital 0 layer

relatives = np.sqrt(np.sum((train_images[:,:,0:2])**2,2))
# distance between chosen particle and all others
relatives = relatives.reshape((len(train_images),len(train_images[0,:,:]),1))
# reshaped to add to original array
train_images = np.append(train_images,relatives,2)
# add to array

train_images[:,:,2][train_images[:,:,2]>(np.pi)] = train_images[:,:,2][train_images[:,:,2]>(np.pi)] - (2*np.pi)
train_images[:,:,2][train_images[:,:,2]<(-np.pi)] = train_images[:,:,2][train_images[:,:,2]<(-np.pi)] + (2*np.pi)
# make value of delta_theta smallest


time_end = time.time()
time_total = time_end - time_start

# plt.figure(6)
# plt.plot(train_images[:,0,2])
# just to see distribution of delta_thetas

#np.save('train_images', train_images)
#np.save('train_labels', train_labels)


## ANIMATION TEST ############################################################

fig = plt.figure(figsize=(7,7))
ax = plt.axes(xlim=(-L/2,L/2),ylim=(-L/2,L/2))
scatter=ax.scatter(train_images[0,:,0], train_images[0,:,1], s=sigma*(80**2), marker='o', edgecolors='k', alpha=0.8)
quiv = ax.quiver(train_images[0,:,0], train_images[0,:,1], np.cos(train_images[0,:,2]), np.sin(train_images[0,:,2]))
ttl = ax.text(0.25, 1.05, 'Brownian Motion', transform = ax.transAxes)
ax.set_xticks(np.arange(-L/2, L/2, 1))
ax.set_yticks(np.arange(-L/2, L/2, 1))
plt.plot(0,0,'ko')
plt.grid()

def connect(i):
    
    step_size = 10
    
    quiv.set_offsets(train_images[step_size*i,:,0:2])
    quiv.set_UVC(np.cos(train_images[step_size*i,:,2]), np.sin(train_images[step_size*i,:,2]))
    #update arrow positions and direction respectively
    
    scatter.set_offsets(train_images[step_size*i,:,0:2])
    #update particle positions
    
    ttl.set_text(('Brownian Motion - "Time": ', round(step_size*i*dt, 3)))
    # update text
    
    return scatter,   
    return quiv,

ani = animation.FuncAnimation(fig, connect, np.arange(0, t-1), interval=1)
plt.show() 

# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=900)
# ani.save('relative_flocking.mp4', writer=writer)   

