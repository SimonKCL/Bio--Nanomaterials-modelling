#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import numpy as np
import random
import matplotlib.pyplot as plt

lx = 30
T = 1 #temp
p = .1 #density
alpha = 1 ## attraction between the disks
remapped = False #extra credit, for remapping in gro file
N = 14000 #timesteps
delt_t = 0.001 #delta t

GDP_distance = 2    #EXTRA CREDIT# given distance probability, distance criterian
GDP = np.zeros(N)   #EXTRA CREDIT# vector of probabilities for a particle to have a particle within given distance


# In[ ]:


#assign velocity to each particle based on random prob distr

num_particles = 0
array = np.zeros((lx,lx))

particles = []

fill = 0

for i in range(0, lx):
    for j in range(0, lx):
        rando = random.uniform(0,1)
        if rando > p:
            array[i,j] = 0  # 0 represents no disk
        else:
            array[i,j] = 1  # 1 represents a disk
            particles.append([i,j])   
            num_particles += 1
            fill = num_particles/(lx*lx)
         
pos = np.asarray(particles, dtype=np.float64)                       
array_init = np.copy(array)   #saving intial array

vxy = np.random.uniform(-100,100, size=(num_particles,2))  #assign uniform initial velocities
vr = np.sqrt((np.square(vxy[:,0]) + np.square(vxy[:,1])))  #overall velocities

vavg = np.zeros(3)
for j in [0,1]:                                              #remove initial drift
    vavg[j] = np.sum(vxy[:,j])/num_particles                 #
vavg[2] = np.sum(vr)/num_particles                           #
fs =  np.sqrt(2*T/vavg[2]**2)                                #          
for i in range(num_particles):                               #
    for j in [0,1]:                                          #
        vxy[i,j] = (vxy[i,j] - vavg[j])*fs                   #
vri = np.sqrt((np.square(vxy[:,0]) + np.square(vxy[:,1])))    #

plt.hist(vri, bins =  50)
plt.show()


# In[ ]:


rcut  = 2.5 ## cutoff distance for the potential energy

epsil = 1 ## characteristic energy
sig = 1 ## size of disks
m     = 1 ## mass

logname = "log.txt"
flog = 10 #rate of logging energies

trajname = "traj.gro"
ftraj = 10 #rate of logging positions

velname = "vel.txt"
fvel = 10 #rate of logging velocities

va = np.zeros(N)                #array for average velocity
vr = np.zeros((N,num_particles)) #overall velocity array, keeping track for each timestep

prev_step = np.copy(pos)  #vector to keep track of positions at t - delt_t
non_mapped = np.copy(pos) #vector of positions that will not be mapped back into the lattice
prev_non = np.copy(pos)  #prev_step for non_mapped positions
KE = np.zeros(N) #vector of Kinetic Energies
U = np.zeros(N)  #vector of Potential Energies
TotalE = np.zeros(N) #vector ot total energies
MSD = np.zeros(N/2)  #vector of mean square displacements
tref = N/2           #reference time to calculate MSD


f= open(trajname,"w+")           # GROMAC file for positons
f.write("MD Sim\n")
f.write("     {}\n".format(num_particles))
for i in range(num_particles):
    f.write("{:5d}   LJ    C{:5d}{:8.3f}{:8.3f}   0.000\n".format((i+1),(i+1),pos[i,0],pos[i,1]))
f.write("   {:6.4f}  {:6.4f}     5.0000\n".format(lx,lx))
f.close() 

f= open(logname,"w+")            #file for energies
f.write("Energies\n")
f.close() 

f= open(velname,"w+")           #file for velocities
f.write("0\n")
f.write("     {}\n".format(num_particles))
for i in range(num_particles):
    f.write("{:8f} {:8f} {:8f}\n".format((i+1),vxy[i,0],vxy[i,1]))
f.close() 
        
for n in range(N):
    Fx = np.zeros(num_particles)
    Fy = np.zeros(num_particles)
    Particler = np.zeros(num_particles)
    GDP_check = np.zeros(num_particles)
    for i in range(num_particles):
        #print(Fx[i])
        if n > tref:  #only start calculating MSD after tref
            rx = non_mapped[i,0] - ptref[i,0]
            ry = non_mapped[i,1] - ptref[i,1]
            rdif2 = rx**2+ry**2     #Distance from tref positions
            MSD[n-1-tref] += rdif2  #Adding differences squared for MSD calc
            
        for j in range(i+1, num_particles):

            dx = pos[i,0] - pos[j,0]
            dy = pos[i,1] - pos[j,1]
       
            if abs(dx) > (lx/2.0): #boundary conditions for shortest distance dx
                if dx < 0:
                    dx = dx + lx
                if dx > 0:
                    dx = dx - lx
            if abs(dy) > (lx/2.0): #boundary conditions for shortest distance dy
                if dy < 0:
                    dy = dy + lx
                if dy > 0:
                    dy = dy - lx
                
            r2 = dx**2+dy**2
            r = np.sqrt(r2)
            
            if r <= GDP_distance: #create vector which will be summed for particles w/ another particle in given dist. 
                GDP_check[i] = 1
                GDP_check[j] = 1
                
            if r < sig:           #set to ensure particles are not overlapping and forces do not get excessively large
                #print("Weird", r, r2, dx, dy, n, i, pos[i,:], Fx[i], Fy[i])
                r = sig
                r2 = r**2
    
            if (r2 < rcut**2):     #if within rcut, calculate forces and potential energy
                
                U[n] = U[n] + (4*epsil*(((sig/r2)**6) - alpha*((sig/r2)**3)))*2

                Fx_change = -24*dx/(r2)*epsil*(alpha*(sig**6)*((1/r2)**3) - 2*(sig**12)*(1/r2)**6) 
                Fy_change = -24*dy/(r2)*epsil*(alpha*(sig**6)*((1/r2)**3) - 2*(sig**12)*(1/r2)**6) 

                Fx[i] = Fx[i] + Fx_change
                Fy[i] = Fy[i] + Fy_change
                Fx[j] = Fx[j] - Fx_change
                Fy[j] = Fy[j] - Fy_change 
                
                if abs(Fx[i]) > 10000000:   #checking foor large forces
                    print(dx , r , i, j, Fx[i])
                    
    GDP[n] = np.sum(GDP_check)/num_particles
    if n > tref:
        MSD[n-tref-1] = MSD[n-tref-1]/(n-tref)/num_particles   #calculate MSD by dividing over times used and num_particles

    print(n)
    holder = np.copy(pos[:,0])          #set holders to maintain previous steps, t-delt_t
    holder2 = np.copy(pos[:,1])         #
    holder3 = np.copy(non_mapped[:,0])  #
    holder4 = np.copy(non_mapped[:,1])  #
    
    if n == 0:                          #determine previous step from initialized speed
        prev_step[:,0] = pos[:,0] - vxy[:,0]*delt_t
        prev_step[:,1] = pos[:,1] - vxy[:,1]*delt_t
        prev_non[:,0] = pos[:,0] - vxy[:,0]*delt_t
        prev_non[:,1] = pos[:,1] - vxy[:,1]*delt_t

    pos[:,0] = 2*pos[:,0] - prev_step[:,0] + (Fx[:]/m)*delt_t**2   #position movement
    pos[:,1] = 2*pos[:,1] - prev_step[:,1] + (Fy[:]/m)*delt_t**2   #
        
    non_mapped[:,0] = 2*non_mapped[:,0] - prev_non[:,0] + (Fx[:]/m)*delt_t**2  #position movement
    non_mapped[:,1] = 2*non_mapped[:,1] - prev_non[:,1] + (Fy[:]/m)*delt_t**2  #position movement
        
    pos[:,0] = pos[:,0] % lx    #boundary conditions
    pos[:,1] = pos[:,1] % lx    #

    vxy[:,0] = (non_mapped[:,0] - prev_non[:,0]) / (2*delt_t) #velocity calc
    vxy[:,1] = (non_mapped[:,1] - prev_non[:,1]) / (2*delt_t) #

    prev_step[:,0] = holder      #assign previous steps from holder values, t-delt_t
    prev_step[:,1] = holder2     #
    prev_non[:,0] = holder3      #
    prev_non[:,1] = holder4      #

    if n == N/2:
        tref = n
        ptref = np.copy(non_mapped)
    
    vr[n,:] = np.sqrt((np.square(vxy[:,0]) + np.square(vxy[:,1]))) #overall velocity
    va[n] = np.mean(vr[n,:])                                      #average velocity
    KE[n] = 0.5*m*num_particles*va[n]**2                    #Kintetic energy at n
    TotalE[n] = KE[n] + U[n]                                #Total energy at n
    Inst_temp = KE[n]/1                           #calculate from kT = 1/2 m v**2 , k = 1
    Timestep = n
    
    if (n-1) % flog == 0:
        f= open(logname,"a")
        
        f.write("{:8f} {:8f} {:8f} {:8f} {:8f}\n".format(Timestep,Inst_temp,KE[n-1],U[n-1],TotalE[n-1]))
        
        f.close() 
        
    if (n % ftraj == 0) and remapped == False:
        f= open(trajname,"a")
        f.write("MD Sim\n")
        f.write("     {}\n".format(num_particles))
        for i in range(num_particles):
            f.write("{:5d}   LJ    C{:5d}{:8.3f}{:8.3f}   0.000\n".format((i+1),(i+1),non_mapped[i,0],non_mapped[i,1]))
        f.write("   {:6.4f}  {:6.4f}     5.0000\n".format(lx,lx))
        f.close()
        
    if (n % ftraj == 0) and remapped == True:
        f= open(trajname,"a")
        f.write("MD Sim\n")
        f.write("     {}\n".format(num_particles))
        for i in range(num_particles):
            f.write("{:5d}   LJ    C{:5d}{:8.3f}{:8.3f}   0.000\n".format((i+1),(i+1),pos[i,0],pos[i,1]))
        f.write("   {:6.4f}  {:6.4f}     5.0000\n".format(lx,lx))
        f.close() 

    if n % fvel == 0:
        f= open(velname,"a")
        f.write("{}\n".format(n))
        f.write("     {}\n".format(num_particles))
        for i in range(num_particles):
            f.write("{:8f} {:8f} {:8f}\n".format((i+1),vxy[i,0],vxy[i,1]))
        f.close() 
        


# In[ ]:


V_boltz = np.reshape(vr[(N/2):,:],(N-N/2)*num_particles)     
plt.hist(V_boltz, bins =  20)
plt.show()      


# In[ ]:


plt.scatter(non_mapped[:,0],non_mapped[:,1])
plt.show()


# In[ ]:


plt.plot(GDP)
plt.show()


# In[ ]:


plt.plot(va)
plt.axis([0, N, 0, 4])
plt.show()


# In[ ]:


plt.plot(MSD)
plt.show()
print(np.polyfit(range(N/2),MSD,1))


# In[ ]:


plt.plot(KE)
plt.axis([0, N, 0, 400])
plt.show()


# In[ ]:


plt.plot(TotalE)
plt.axis([0, N, -20, 400])
plt.show()


# In[ ]:


plt.plot(U)
plt.axis([0, N, -70,20])
plt.show()


# In[ ]:




