#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import numpy as np
import random
import matplotlib.pyplot as plt


# In[25]:


nx = 50 #matrix dimmensions
p = .9 #density of nodes
fa = 0.5 #fraction of a disks
p_neighborshift = 0 #extra credit, probability of shifting to neighbor, set to 0 for normal
Eaa = -2 #energy kcal/mol of aa interaction
Eab = 2  #energy kcal/mol of ab interaction
Ebb = -2 #energy kcal/mol of bb interaction
nmoves = 1000000 #number of moves
T = 300  #K
k = 0.0019872041 #kcal/(mol*K)
array = np.zeros((nx,nx))


# In[26]:


for i in range(0, nx):
    for j in range(0,nx):
        rando = random.random()
        if rando > p:
            array[i,j] = 0  # 0 represents no disk
        elif rando > p*fa:
            array[i,j] = 1  # 1 represents b
        else:
            array[i,j] = 2  # 2 represents a
                       
array_init = np.copy(array) #saving intial array
        


# In[27]:


def EnergyCalc(x,y,disk):
    E = 0
    
    #accounting for periodic boundary conditions
    if x+1 == nx:
        Xu = array[0,y] * disk #disk is  0,1,or 2 for no disk, b, and a
    else:
        Xu = array[x+1,y] * disk
    if x-1 == -1:
        Xd = array[nx-1,y] * disk
    else:
        Xd = array[x-1,y] * disk
    if y+1 == nx:
        Yu = array[x,0] * disk
    else:
        Yu = array[x,y+1] * disk
    if y-1 == -1:
        Yd = array[x,nx-1] * disk
    else:
        Yd = array[x,y-1] * disk
        
    #summing energy interactions with neighboring nodes    
    for c in [Xu,Xd,Yu,Yd]:
        if c == 4:
            E = E + Eaa
        elif c == 2:
            E = E + Eab
        elif c == 1:
            E = E + Ebb
            
    return E

def UTotal(array):
    Enew = 0
    
    for i in range(0, nx):
        for j in range(0,nx):
            a = 0
            a = EnergyCalc(i,j,array[i,j])
            Enew = Enew + a
    return Enew


# In[28]:


print(UTotal(array))


# In[29]:



total_accepted = 0
EnergyAtMove = range(nmoves)
EnergyAtMove[0] = UTotal(array)
RatioAccepted = range(nmoves)
Acceptance = range(nmoves)
for i in range(nmoves):
    
    x = random.randint(0,nx-1)
    y = random.randint(0,nx-1)
    
    
    while array[x,y]==0:
        x = random.randint(0,nx-1)
        y = random.randint(0,nx-1)
        
    
    ######
    ######extra credit, choose whether to swap with random position or adjacent
    ######
    choose = random.random() 
    if choose > p_neighborshift:    #for random position swap
        new_x = random.randint(0,nx-1)
        new_y = random.randint(0,nx-1)
        
    else:                           #for adjacent swap
        
        #
        #choosing direction of shift
        #
        choose_direction = random.random() #choosing direction of adjacent shift
        if choose_direction < .25:
            new_x = x+1
            new_y = y
        elif choose_direction < .5:
            new_x = x-1
            new_y = y
        elif choose_direction < .75:
            new_x = x
            new_y = y+1
        else:
            new_x = x
            new_y = y-1
        #
        #
        
        
        #
        #accounting for periodic boundary conditions
        #
        if new_x == nx:
            new_x = 0
        if new_x == -1:
            new_x = nx-1
        if new_y == nx:
            new_y = 0
        if new_y == -1:
            new_y = nx-1
        #
        #
        #


    ######
    ######
    ######
    
    
    f = array[x,y]
    d = array[new_x,new_y]
    
    
    #
    #Calculating Un-Um with multiplying by 2 at each point shift, given neighboring node energies also change
    #
    Un_Um= 0
    Un_Um = 2*(EnergyCalc(new_x,new_y,f)+EnergyCalc(x,y,d)) - 2*(EnergyCalc(new_x,new_y,d)+EnergyCalc(x,y,f))
        
        
    #
    #
    #
    
    
    if (Un_Um) < 0:              #accept if new system has lower energy
        array[x,y] = d
        array[new_x,new_y] = f
        total_accepted += 1
        Acceptance[i]=True
        
    else:
        W = 0 
        W = np.exp(-(Un_Um)/(k*T))   #metropolis criterion
        if W > random.random():
            array[x,y] = d
            array[new_x,new_y] = f
            total_accepted += 1
            Acceptance[i]=True
            
        else:
            Acceptance[i]= False
    
    if Acceptance[i] == True:                       #save energy at move i
        if i == 0:
            EnergyAtMove[i] = EnergyAtMove[i] + Un_Um  
        else:
            EnergyAtMove[i] = EnergyAtMove[i-1] + Un_Um
    if Acceptance[i]==False:
        if i == 0:
            EnergyAtMove[i] = EnergyAtMove[i]
        else:
            EnergyAtMove[i] = EnergyAtMove[i-1]
    
    RatioAccepted[i]= float(total_accepted)/(i+1)
    #print(EnergyAtMove[i])
            
            


# In[ ]:



fig, (ax1,ax2) = plt.subplots(2)



ax1.imshow(array, cmap=plt.cm.Blues, interpolation='nearest')
ax1.set_title('Final Array')
# Move left and bottom spines outward by 10 points
ax1.spines['left'].set_position(('outward', 10))
ax1.spines['bottom'].set_position(('outward', 10))
# Hide the right and top spines
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax1.yaxis.set_ticks_position('left')
ax1.xaxis.set_ticks_position('bottom')

ax2.imshow(array_init, cmap=plt.cm.Blues, interpolation='nearest')
ax2.set_title('Initial Array')
# Move left and bottom spines outward by 10 points
ax2.spines['left'].set_position(('outward', 10))
ax2.spines['bottom'].set_position(('outward', 10))
# Hide the right and top spines
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax2.yaxis.set_ticks_position('left')
ax2.xaxis.set_ticks_position('bottom')

plt.show()




# In[71]:


plt.plot(EnergyAtMove)
plt.ylabel('Total Energy')
plt.show()


# In[72]:


plt.plot(RatioAccepted)
plt.ylabel('Ratio Accepted')
plt.show()


# In[ ]:





# In[30]:


l = UTotal(array)
print(l)


# In[ ]:




