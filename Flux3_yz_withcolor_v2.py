# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 12:29:38 2019
Local flux
@author: gadgr
"""



import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns

from itertools import chain

import numpy as np
import csv

def wrap_at_bounds(x):
    if x[2]<0:
        x[2]=378.842 + x[2]
    
    x[2]-=189.421
            
    return(x)

steps=np.arange(0000000,5000000,10000)
#steps[0]=357


cond_pos=np.array([-120,-110,-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90,100,110])
cond_midpos=np.array([-125,-115,-105,-95,-85,-75,-65,-55,-45,-35,-25,-15,-5,5,15,25,35,45,55,65,75,85,95,105,115])
h_pos=np.zeros((15,len(cond_pos)))
h_pos[0,:]=np.array([-32.5,-32.5,-22.5,-14.5,-10.0,-5.5,-1.5,-1.0,-1.0,-0.5,0.5,0.5,3.5,3.5,3.5,3.0,2.0,-0.5,-4.0,-9.0,-16.5,-22,-32.5,-32.5])
#h_pos[0,:]=h_pos[0,:]+1.5
for i in range(14):
    h_pos[i+1,:]=h_pos[i,:]-2.5



flux_net=np.zeros((15,len(cond_pos),len(steps)),dtype=float)

flux_net_z=np.zeros((15,len(cond_pos),len(steps)),dtype=float)


    
initialpos=[] 
afterpos=[]
########################################################################
#initial positions
########################################################################
with open('corners_1350_%i.xyz'%(steps[0]), 'r') as f:
    
    after_3 = f.readlines()
    
    #50000000s
    first_few=after_3[9:]
    #reader = csv.reader(last_few,delimiter=' ')
    #for row in reader:
    #last_few_split=last_few.split("")
    #planes.append(list(map(float, last_few_split)))
    for i in range(len(first_few)):
        first_few_split=str.split(first_few[i])
        first_few_splitmap=list(map(float, first_few_split))
        #first_few_splitmap[0]=0
        initialpos.append(first_few_splitmap) 
    
f.close()

pos1=np.asarray(initialpos)
orderedpos1=np.zeros((pos1.shape[0],pos1.shape[1]-1),dtype=float)
for i in range(pos1.shape[0]):
    
    index=int(pos1[i,0]-1)
    orderedpos1[index,:]=wrap_at_bounds(pos1[i,1:])
print('initial')
print('corners_1350_%i.xyz'%(steps[0]))    



orderedpos2=np.zeros((pos1.shape[0],8),dtype=float)

#limits=([390,148.3,-143])

r_check=np.zeros(len(cond_pos),dtype=int)

for l in  range(0,int(len(steps))-1):
    finalpos=[]
    with open('corners_1350_%i.xyz'%(steps[l]), 'r') as f:
        after_3 = f.readlines()
        
        #50000000s
        first_few=after_3[9:]
        #reader = csv.reader(last_few,delimiter=' ')
        #for row in reader:
        #last_few_split=last_few.split("")
        #planes.append(list(map(float, last_few_split)))
        for i in range(len(first_few)):
            first_few_split=str.split(first_few[i])
            first_few_splitmap=list(map(float, first_few_split))
            #first_few_splitmap[0]=0
            finalpos.append(first_few_splitmap) 
        
    f.close()
    
    pos2=np.asarray(finalpos)
    orderedpos2_old=np.copy(orderedpos2)
    orderedpos2=np.zeros((pos2.shape[0],8),dtype=float)
    for i in range(len(pos2)):
        #print(i)
        index=int(pos2[i,0]-1)
        orderedpos2[index,:]=wrap_at_bounds(pos2[i,1:])
    
    
    for i in range(len(orderedpos1)):
        if l>5:
            for h in range(15):
                #if orderedpos2_old[i,2]<cond_midpos[0]:
                #    r=0
                #elif  orderedpos2_old[i,2]>cond_pos[23]:
                #    r=24
                r=0
                for ri in range(len(cond_pos)):
                    if orderedpos2_old[i,2]>cond_midpos[ri] and orderedpos2_old[i,2]<cond_midpos[ri+1]:
                        r=ri
                
                #print(6)
                #if h>0 and h<11:
                condition1=orderedpos2[i,3]>h_pos[h,r]+1.25 and orderedpos2_old[i,3]<h_pos[h,r]+1.25 and orderedpos2_old[i,0]==1
                condition2=orderedpos2[i,3]<h_pos[h,r]+1.25 and orderedpos2_old[i,3]>h_pos[h,r]+1.25 and orderedpos2_old[i,0]==1
                if  condition1: 
                    
                    #flux_plus_z[h,r,l]+=1
                    flux_net_z[h,r,l]+=1
                    
                elif  condition2:
                    
                    #flux_minus_z[h,r,l]+=1
                    flux_net_z[h,r,l]-=1
                        
# =============================================================================
#                 elif r==0:
#                     if np.abs(orderedpos2[i,2]-cond_pos[r])<50 and np.abs(orderedpos2_old[i,2]-cond_pos[r])>100:
#                         flux_plus[h,r,l]+=1
#                         flux_net[h,r,l]+=1
#                     elif np.abs(orderedpos2[i,2]-cond_pos[r])>100 and np.abs(orderedpos2_old[i,2]-cond_pos[r])<50:
#                         flux_minus[h,r,l]+=1
#                         flux_net[h,r,l]-=1
# =============================================================================
    #for i in range(len(orderedpos1)):
    #    if l>5:
            for r in range(len(cond_pos)):
                if orderedpos2_old[i,3]<h_pos[14,r]:
                    h=14
                elif  orderedpos2_old[i,3]>h_pos[0,r]:
                    h=0
                for hi in range(13):
                    if orderedpos2_old[i,3]>h_pos[hi+1,r] and orderedpos2_old[i,3]<h_pos[hi,r]:
                        h=hi+1
                
                #print(6)
                if r>=0:
                    condition1=orderedpos2[i,2]>cond_pos[r] and orderedpos2_old[i,2]<cond_pos[r] and orderedpos2_old[i,0]==1 and np.abs(orderedpos2[i,2] - orderedpos2_old[i,2])<100
                    condition2=orderedpos2[i,2]<cond_pos[r] and orderedpos2_old[i,2]>cond_pos[r] and orderedpos2_old[i,0]==1 and np.abs(orderedpos2[i,2] - orderedpos2_old[i,2])<100
                    if  condition1:
                        
                        #flux_plus[h,r,l]+=1
                        flux_net[h,r,l]+=1
                        
                    elif  condition2:
                        
                        #flux_minus[h,r,l]+=1
                        flux_net[h,r,l]-=1
                        
# =============================================================================
#                 elif r==0:
#                     if np.abs(orderedpos2[i,2]-cond_pos[r])<50 and np.abs(orderedpos2_old[i,2]-cond_pos[r])>100:
#                         #flux_plus[h,r,l]+=1
#                         flux_net[h,r,l]+=1
#                     elif np.abs(orderedpos2[i,2]-cond_pos[r])>100 and np.abs(orderedpos2_old[i,2]-cond_pos[r])<50:
#                         #flux_minus[h,r,l]+=1
#                         flux_net[h,r,l]-=1
# =============================================================================

                        
                        
    if l%10==0:
        print('corners_1350_%i.xyz'%(steps[l])) 
    
    



#cond_pos=np.array([-187,-180,-160,-140,-130,-120,-110,-100,-90,-80,-70,-50,0,50,80,90,100,110,120,130,140,150,160,180])
cond_pos2=np.copy(cond_pos)
#cond_pos2[0:12]=cond_pos[12:]
cond_pos3=np.copy(cond_pos2)
#for i in range(len(cond_pos2[12:])):   
#    cond_pos2[12+i]=190 +(190+cond_pos[i])
#cond_pos3[0:12]=cond_pos2[12:]
#cond_pos3[12:]=cond_pos2[0:12]


flux_sum=np.copy(flux_net)
flux_sum_z=np.copy(flux_net_z)


for h in range(12):
    for r in range(len(cond_pos)):
        for w in range(1,int(len(steps))):
                    flux_sum[h,r,w]+=flux_sum[h,r,w-1]
                    flux_sum_z[h,r,w]+=flux_sum_z[h,r,w-1]

##############################################################################
#Smoothening
##############################################################################
                
flux_sum2=np.zeros((12,len(cond_pos),len(steps)),dtype=float)
flux_sum2_z=np.zeros((12,len(cond_pos),len(steps)),dtype=float)

#Normalizing with area
for h in range(12):
    for r in range(len(cond_pos)):
        flux_sum2[h,r]=flux_sum[h,r]/(2.5*33.5)
        flux_sum2_z[h,r]=flux_sum_z[h,r]/((cond_midpos[r+1]-cond_midpos[r])*33.5)

               
flux_smooth=np.zeros((flux_sum.shape),dtype=float)
flux_smooth_z=np.zeros((flux_sum.shape),dtype=float)



for h in range(12):
    for i in range(12,(flux_sum2.shape[2])-13):
        print(i)
        for j in range(25):
            flux_smooth[h,:,i]+=flux_sum2[h,:,i-12+j]
            flux_smooth_z[h,:,i]+=flux_sum2_z[h,:,i-12+j]
            #print(flux_smooth[:,i])

        flux_smooth[h,:,i]=flux_smooth[h,:,i]/25
        flux_smooth_z[h,:,i]=flux_smooth_z[h,:,i]/25

flux_smooth[:,:,-25:]=flux_sum2[:,:,-25:]
flux_smooth_z[:,:,-25:]=flux_sum2_z[:,:,-25:]
# =============================================================================
# 
# flux_smooth=np.copy(flux_sum2)
# flux_smooth_z=np.copy(flux_sum2_z)
# =============================================================================

plt.rc('font', size=60)          # controls default text sizes
plt.rc('axes', titlesize=65)     # fontsize of the axes title
plt.rc('axes', labelsize=65)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=65)    # fontsize of the tick labels
plt.rc('ytick', labelsize=65)    # fontsize of the tick labels
plt.rc('legend', fontsize=60)    # legend fontsize
plt.rc('figure', titlesize=60)  # fontsize of the figure title

colors=np.zeros((21*len(cond_pos)),dtype=float)
for i in range(len(colors)):
    if i%15==0:
        colors[i]=0.9
    elif i%15==1:
        colors[i]=0.5
    elif i%15==2:
        colors[i]=0.25
# =============================================================================
# colors[:len(cond_pos)]=0.9
# colors[len(cond_pos):2*len(cond_pos)]=0.5
# colors[2*len(cond_pos):3*len(cond_pos)]=0.25
# 
# =============================================================================
colormap=cm.nipy_spectral
    
colors2=[cm.nipy_spectral(0.9),cm.nipy_spectral(0.5),cm.nipy_spectral(0.25)]                
for r in range(12,len(cond_pos)):    
    plt.figure(figsize=(15,15))
    #plt.plot(val_perA[:,0],(val_perA[:,5]))
    #plt.ylim(1.8e9,2.0e9)
    plt.plot(steps[100:-25]*1e-6,flux_sum2[0,r,100:-25],label=" flux sum at %.1f$\AA$ surface "%(cond_pos[r]),linewidth=2,marker='o',markersize=5,color=colors2[0])
    plt.plot(steps[100:-25]*1e-6,flux_smooth[0,r,100:-25],linewidth=2,marker='o',markersize=5,color=colors2[0])
    plt.plot(steps[100:-25]*1e-6,flux_sum2[1,r,100:-25],label=" flux sum at %.1f$\AA$ layer1 "%(cond_pos[r]),linewidth=2,marker='o',markersize=5,color=colors2[1])
    plt.plot(steps[100:-25]*1e-6,flux_smooth[1,r,100:-25],linewidth=2,marker='o',markersize=5,color=colors2[1])
    plt.plot(steps[100:-25]*1e-6,flux_sum2[2,r,100:-25],label=" flux sum at %.1f$\AA$ layer2 "%(cond_pos[r]),linewidth=2,marker='o',markersize=5,color=colors2[2])
    plt.plot(steps[100:-25]*1e-6,flux_smooth[2,r,100:-25],linewidth=2,marker='o',markersize=5,color=colors2[2])
    #plt.plot(steps[1:]*1e-6,flux_sum_5[r,1:],label=" flux sum at %.1f$\AA$ -12.5$\AA$ "%(cond_pos3[r]),linewidth=2,marker='o',markersize=5)
    #plt.plot(steps[1:]*1e-6,flux_sum[r,1:],label=" flux sum at %.1f$\AA$ deeper"%(cond_pos3[r]),linewidth=2,marker='o',markersize=5)
    #plt.plot(steps[3:]*2e-6,flux_net[r,3:],label="    net flux %.1f "%(cond_pos[r]),linewidth=4,marker='o',markersize=10)
    #plt.plot(steps[3:]*2e-6,flux_minus[r,3:],label="    flux mean %.1f "%(cond_pos[r]),linewidth=4,marker='o',markersize=10)
    
    #plt.legend(bbox_to_anchor=(0.01, 0.99), loc=2, borderaxespad=0.)
    #plt.legend()
    plt.ylim(-6,11)
    plt.ylabel('Number of atoms ($atoms/\AA^2$)')
    plt.xlabel('Time (ns)')
    plt.show()   
    
    
plt.rc('font', size=60)          # controls default text sizes
plt.rc('axes', titlesize=40)     # fontsize of the axes title
plt.rc('axes', labelsize=40)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=40)    # fontsize of the tick labels
plt.rc('ytick', labelsize=40)    # fontsize of the tick labels
plt.rc('legend', fontsize=60)    # legend fontsize
plt.rc('figure', titlesize=40)  # fontsize of the figure title

for i in range(4,19):    
    slopes2=np.zeros((len(cond_pos),15), dtype=float)  
    #slopes=np.zeros((len(cond_pos)*15), dtype=float)
    slopes_y=np.zeros((len(cond_pos)*15), dtype=float)
    slopes_z=np.zeros((len(cond_pos)*15), dtype=float)
    
    theta_pos=np.zeros(len(cond_pos),dtype=float)
    theta_pos[4:10] =60.0*np.pi/180
    theta_pos[14:20] =-60.0*np.pi/180
    
    
    for r in range(len(cond_pos3)):
        for h in range(15):
# =============================================================================
#             slopes2[r,h]=np.sqrt(((flux_sum2[h,r,20+(10*i)]-flux_sum2[h,r,10+(10*i)])/((steps[20+(10*i)]-steps[10+(10*i)])*1e-6))**2 + ((flux_sum2_z[h,r,20+(10*i)]-flux_sum2_z[h,r,10+(10*i)])/((steps[20+(10*i)]-steps[10+(10*i)])*1e-6))**2)
#             #slopes[r*15+h]=np.sqrt(((flux_sum2[h,r,20+(10*i)]-flux_sum2[h,r,10+(10*i)])/((steps[20+(10*i)]-steps[10+(10*i)])*1e-6))**2 + ((flux_sum2_z[h,r,20+(10*i)]-flux_sum2_z[h,r,10+(10*i)])/((steps[20+(10*i)]-steps[10+(10*i)])*1e-6))**2)
#             
#             slopes_y[r*15+h]=(flux_sum2[h,r,20+(10*i)]-flux_sum2[h,r,10+(10*i)])/((steps[20+(10*i)]-steps[10+(10*i)])*1e-6)
#             slopes_z[r*15+h]=(flux_sum2_z[h,r,20+(10*i)]-flux_sum2_z[h,r,10+(10*i)])/((steps[20+(10*i)]-steps[10+(10*i)])*1e-6)
#             
# =============================================================================
            slopes2[r,h]=np.sqrt(((flux_smooth[h,r,49+(25*i)]-flux_smooth[h,r,25+(25*i)])/((steps[49+(25*i)]-steps[25+(25*i)])*1e-6))**2 + ((flux_smooth_z[h,r,49+(25*i)]-flux_smooth_z[h,r,25+(25*i)])/((steps[49+(25*i)]-steps[25+(25*i)])*1e-6))**2)
            #slopes[r*15+h]=np.sqrt(((flux_smooth[h,r,49+(25*i)]-flux_smooth[h,r,25+(25*i)])/((steps[49+(25*i)]-steps[25+(25*i)])*1e-6))**2 + ((flux_smooth_z[h,r,49+(25*i)]-flux_smooth_z[h,r,25+(25*i)])/((steps[49+(25*i)]-steps[25+(25*i)])*1e-6))**2)
            
            slopes_y[r*15+h]=(flux_smooth[h,r,49+(25*i)]-flux_smooth[h,r,25+(25*i)])/((steps[49+(25*i)]-steps[25+(25*i)])*1e-6)
            slopes_z[r*15+h]=(flux_smooth_z[h,r,49+(25*i)]-flux_smooth_z[h,r,25+(25*i)])/((steps[49+(25*i)]-steps[25+(25*i)])*1e-6)
            
            #slopes_y[r*15+h]=slopes[r*15+h]*0#*np.cos(theta_pos[r])
            
            #slopes_z[r*15+h]=slopes[r*15+h]*1#*np.sin(theta_pos[r])
            
        
    x_pos=np.zeros((15*len(cond_pos)),dtype=float)
    y_pos=np.zeros((15*len(cond_pos)),dtype=float)
    
    for r in range(len(cond_pos)):
        for e in range(15):
            x_pos[r*15+e]=cond_pos[r]
        
            if r<30:
                y_pos[r*15+e]=h_pos[e,r]+1.25
               
            else:
                y_pos[r*15+e]=h_pos[e,r-15]+1.25
            

    def symlog(x):
        """ Returns the symmetric log10 value """
        return np.sign(x) * np.log10(np.abs(x))
            
    plt.figure(figsize=(50,20))
    #plt.plot(val_perA[2200:26000,0],(val[2200:13000,5]))
    #plt.plot(val[2200:26000,0],(val[2200:52000,4]+85000))
    #plt.plot(val[2200:52000,0],(val[2200:52000,3]))
    #plt.plot(val_perA[:,0],(val_perA[:,5]))
    #plt.colorbar()
    ax=plt.quiver(x_pos,y_pos, (slopes_y), (slopes_z),color=colormap(colors),width=0.002,pivot='mid',label='%.2f-%.2fns'%(steps[25+(25*i)]*1e-6,steps[49+(25*i)]*1e-6),scale=50)#, norm_vel)
    #plt.legend()
    #plt.colorbar(ax)
    plt.ylabel('Z(A)')
    plt.xlabel('Y(A)')
    plt.show()


#flux_sum2[:,0:12,:]=flux_sum[:,12:,:]


#for i in range(len(cond_pos2[12:])):   
    #cond_pos2[12+i]=190 +(-1*(cond_pos[12-i]))
#    flux_sum2[:,12+i,:]=flux_sum[:,i,:]
    
  
# =============================================================================
# plt.figure(figsize=(50,15))
# for w in range(10,int(len(steps)/3),40) :
#     plt.plot(cond_pos,-1*flux_sum[:,w],label=" Net Flux %.1f "%(steps[w]*1e-6),linewidth=2,marker='o',markersize=10)
# #plt.plot(steps[3:]*2e-6,flux_meansum[r,s,3:],label=" %s   flux mean %.1f "%(surface[s],cond_pos[r]),linewidth=4,marker='o',markersize=10)
# plt.legend()
# plt.ylabel('Number of atoms')
# plt.xlabel('Position(A)')
# plt.show()
# =============================================================================
for h in range(5):
    plt.figure(figsize=(50,15))
    for w in range(40,int(len(steps))-25,10) :
        plt.plot(cond_pos[:],flux_sum2[h,:,w],label=" at %.1f ns"%(steps[w]*1e-6),linewidth=2,marker='o',markersize=10)
    #plt.plot(steps[3:]*2e-6,flux_meansum[r,s,3:],label=" %s   flux mean %.1f "%(surface[s],cond_pos[r]),linewidth=4,marker='o',markersize=10)
    plt.legend()
    plt.ylabel('Number of atoms')
    plt.xlabel('Position(A)')
    plt.show()


flux_sum2_sum=np.copy(flux_sum2[0,:,:])
for i in range(14):
    flux_sum2_sum[:,:]+=flux_sum2[i+1,:,:]


plt.figure(figsize=(50,15))
for w in range(40,int(len(steps))-25,10) :
    plt.plot(cond_pos[:],(flux_sum2_sum[:,w]),label=" at %.1f ns "%(steps[w]*1e-6),linewidth=2,marker='o',markersize=10)
#plt.plot(steps[3:]*2e-6,flux_meansum[r,s,3:],label=" %s   flux mean %.1f "%(surface[s],cond_pos[r]),linewidth=4,marker='o',markersize=10)
plt.legend()
plt.ylabel('Number of atoms')
plt.xlabel('Position(A)')
plt.show()

for h in range(5):
    plt.figure(figsize=(50,15))
    for w in range(40,int(len(steps))-25,10) :
        plt.plot(cond_pos,flux_sum2_z[h,:,w],label="vf at %.1f ns"%(steps[w]*1e-6),linewidth=2,marker='o',markersize=10)
    #plt.plot(steps[3:]*2e-6,flux_meansum[r,s,3:],label=" %s   flux mean %.1f "%(surface[s],cond_pos[r]),linewidth=4,marker='o',markersize=10)
    plt.legend()
    plt.ylabel('Number of atoms')
    plt.xlabel('Position(A)')
    plt.show()


flux_sum2_z_sum=np.copy(flux_sum2_z[0,:,:])
for i in range(14):
    flux_sum2_z_sum[:,:]+=flux_sum2_z[i+1,:,:]


plt.figure(figsize=(50,15))
for w in range(40,int(len(steps))-25,10) :
    plt.plot(cond_pos,(flux_sum2_z_sum[:,w]),label="vf at %.1f ns "%(steps[w]*1e-6),linewidth=2,marker='o',markersize=10)
#plt.plot(steps[3:]*2e-6,flux_meansum[r,s,3:],label=" %s   flux mean %.1f "%(surface[s],cond_pos[r]),linewidth=4,marker='o',markersize=10)
plt.legend()
plt.ylabel('Number of atoms')
plt.xlabel('Position(A)')
plt.show()