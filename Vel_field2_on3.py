#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 10:57:41 2021

Vel_field2_on2_but with every 5000 steps
@author: ganesha2
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
from itertools import chain

import numpy as np
import csv


def wrap_at_bounds(x):
    for i in range(x.shape[0]):
        if x[i,3]<0:
            x[i,3]=378.842 + x[i,3]
    
    x[:,3]-=189.421
            
    return(x)

#steps=np.arange(1000000,5000000,50000)
steps=np.arange(1000000,4500000,10000)
dt=1.0*100000*1e-6
#steps[0]=357


# =============================================================================
# cond_pos=np.array([-120,-110,-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90,100,110])
# cond_midpos=np.array([-125,-115,-105,-95,-85,-75,-65,-55,-45,-35,-25,-15,-5,5,15,25,35,45,55,65,75,85,95,105,115])
# h_pos=np.zeros((15,len(cond_pos)))
# h_pos[0,:]=np.array([-32.5,-32.5,-22.5,-14.5,-10.0,-5.5,-1.5,-1.0,-1.0,-0.5,0.5,0.5,3.5,3.5,3.5,3.0,2.0,-0.5,-4.0,-9.0,-16.5,-22,-32.5,-32.5])
# #h_pos[0,:]=h_pos[0,:]+1.5
# for i in range(14):
#     h_pos[i+1,:]=h_pos[i,:]-2.5
# =============================================================================

cond_pos=np.array([-30,-20,-10,0,10,20,30,40,50,60,70,80,90,100,110])
cond_midpos=np.array([-45,-25,-15,-5,5,15,25,35,45,55,65,75,85,95,105,125])
h_pos=np.zeros((16,len(cond_pos)))
h_pos[0,:]=np.array([-0.5,0.5,0.5,3.5,3.5,3.5,3.0,2.0,-0.5,-4.0,-9.0,-16.5,-22,-32.5,-32.5])
#h_pos[0,:]=h_pos[0,:]+1.5
for i in range(15):
    h_pos[i+1,:]=h_pos[i,:]-3



velo_x=np.zeros(((h_pos.shape[0]+1)*len(cond_pos),len(steps)),dtype=float)
velo_y=np.zeros(((h_pos.shape[0]+1)*len(cond_pos),len(steps)),dtype=float)
#noo_xy=np.zeros(((h_pos.shape[0]+1)*len(cond_pos),len(steps)),dtype=float)
velo_x2=np.zeros(((h_pos.shape[0]+1),len(cond_pos),len(steps)),dtype=float)
velo_y2=np.zeros(((h_pos.shape[0]+1),len(cond_pos),len(steps)),dtype=float)
#noo_xy2=np.zeros(((h_pos.shape[0]+1),len(cond_pos),len(steps)),dtype=float)

vel_x=np.zeros(((h_pos.shape[0]+1)*len(cond_pos),len(steps)),dtype=float)
vel_y=np.zeros(((h_pos.shape[0]+1)*len(cond_pos),len(steps)),dtype=float)
no_xy=np.zeros(((h_pos.shape[0]+1)*len(cond_pos),len(steps)),dtype=float)
vel_x2=np.zeros(((h_pos.shape[0]+1),len(cond_pos),len(steps)),dtype=float)
vel_y2=np.zeros(((h_pos.shape[0]+1),len(cond_pos),len(steps)),dtype=float)
no_xy2=np.zeros(((h_pos.shape[0]+1),len(cond_pos),len(steps)),dtype=float)
no_posx=np.zeros(((h_pos.shape[0]+1)*len(cond_pos)),dtype=float)
no_posy=np.zeros(((h_pos.shape[0]+1)*len(cond_pos)),dtype=float)
no_posxy=np.zeros(((h_pos.shape[0]+1),len(cond_pos),2),dtype=float)
########################################################################
#initial positions
########################################################################

initialpos=[] 
afterpos=[]


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
orderedpos1=wrap_at_bounds(pos1[pos1[:,0].argsort()])
orderedpos1=orderedpos1[orderedpos1[:,1]==1]
# =============================================================================
#     for i in range(len(pos2)):
#         #print(i)
#         index=int(pos2[i,0]-1)
#         orderedpos2[index,:]=(pos2[i,1:])#wrap_at_bounds(pos2[i,1:])
#         
#     print('corners_1350_%i.xyz'%(steps[l])) 
# =============================================================================
orderedpos2=np.copy(orderedpos1)  
    
     

#limits=([390,148.3,-143])

#r_check=np.zeros(len(cond_pos),dtype=int)

for l in  range(1,int(len(steps))-1):
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
    orderedpos2_old=wrap_at_bounds(pos2[pos2[:,0].argsort()])
    orderedpos2_old=orderedpos2_old[orderedpos2_old[:,1]==1]
    #orderedpos2=pos2[pos2[:,0].argsort()]
    #orderedpos2=orderedpos2[orderedpos2[:,1]==1]
    
    # steps after 50000 steps
    finalpos2=[]
    with open('corners_1350_%i.xyz'%(steps[l]+ 100000), 'r') as f:
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
            finalpos2.append(first_few_splitmap) 
        
    f.close()
    
    pos3=np.asarray(finalpos2)
    #orderedpos2_old=np.copy(orderedpos2)
    orderedpos2=wrap_at_bounds(pos3[pos3[:,0].argsort()])
    orderedpos2=orderedpos2[orderedpos2[:,1]==1]
    
    #write file
# =============================================================================
#     xlim=[0,34]
#     ylim=[0,380]
#     zlim=[-50,50]
#     print("writing")
#     f=open('wrap_test_%i'%(l),'w+')
#     print("writing")
#     f.write('(written by Ganesh A)\n\n')
#     f.write('%d\tatoms\n'%(len(orderedpos2)))
#     f.write('2\tatom types\n')
#     f.write('%.9g\t%.9g\txlo xhi\n'%(xlim[0],xlim[1]))
#     f.write('%.9g\t%.9g\tylo yhi\n'%(ylim[0],ylim[1]))
#     f.write('%.9g\t%.9g\tzlo zhi\n'%(zlim[0],zlim[1]))
#     
#     f.write('\n\n')
#     f.write('Masses\n\n')
#     
#     f.write('1\t63.456\n')
#     f.write('2\t12.00\n\n')
#     #f.write('3\t1.00\n\n')
#     f.write('Atoms\n\n')
#     i=0
#     for i in range(len(orderedpos2)):
#         #print(1)
#         f.write('%d %d %d %d %1.3f %1.3f %1.3f %d %d %d\n'%(i+1,orderedpos2[i,1],orderedpos2[i,1],0,orderedpos2[i,2],orderedpos2[i,3],orderedpos2[i,4],0,0,0))
#     
#     #f.write('\nVelocities\n\n')
#     #i=0
#     #for i in range(len(val)):
#         #print(2)
#     #    f.write('%d %1.4f %1.4f %1.4f\n'%(i+1,val[i,5],val[i,6],val[i,7]))
#     
#     
#     f.close()
# =============================================================================

    
# =============================================================================
#     for i in range(len(pos2)):
#         #print(i)
#         index=int(pos2[i,0]-1)
#         orderedpos2[index,:]=(pos2[i,1:])#wrap_at_bounds(pos2[i,1:])
#         
#     print('corners_1350_%i.xyz'%(steps[l])) 
# =============================================================================
    ########################################################################
    #Select region
    ########################################################################
        
        
        
    for i in range(len(orderedpos2)):   
        if l>0 and orderedpos2[i,3]>-30 and orderedpos2[i,3]<115:
        
            
            for ri in range(len(cond_pos)):
                #if (orderedpos2[i,3]>cond_midpos[ri] and orderedpos2[i,3]<cond_midpos[ri+1]) :
                if (orderedpos2[i,3]>cond_midpos[ri] and orderedpos2[i,3]<cond_midpos[ri+1]) :
                    r=ri
            
            for ri in range(len(cond_pos)):
                #if (orderedpos2[i,3]>cond_midpos[ri] and orderedpos2[i,3]<cond_midpos[ri+1]) :
                if (orderedpos2_old[i,3]>cond_midpos[ri] and orderedpos2_old[i,3]<cond_midpos[ri+1]):
                    ro=ri
# =============================================================================
#             if orderedpos2[i,4]<h_pos[19,r]:
#                 h=20
#             elif  orderedpos2[i,4]>h_pos[0,r]:
#                 h=0
#             for hi in range(19):
#                 if orderedpos2[i,4]>h_pos[hi+1,r] and orderedpos2[i,4]<h_pos[hi,r]:
#                     h=hi+1
# =============================================================================
                    
            
            if orderedpos2[i,4]<h_pos[15,r]:
                h=16
            elif  orderedpos2[i,4]>h_pos[0,r]:
                h=0
            for hi in range(15):
                if (orderedpos2[i,4]>h_pos[hi+1,r] and orderedpos2[i,4]<h_pos[hi,r]):
                    h=hi+1
                    
            if orderedpos2_old[i,4]<h_pos[15,r]:
                ho=16
            elif  orderedpos2_old[i,4]>h_pos[0,r]:
                ho=0
            for hi in range(15):
                if (orderedpos2_old[i,4]>h_pos[hi+1,r] and orderedpos2_old[i,4]<h_pos[hi,r]):
                    ho=hi+1
                    
            if np.abs(orderedpos2[i,3]-orderedpos2_old[i,3])<200:
                    
                vel_x[h*len(cond_pos) + r,l]+=(orderedpos2[i,3]-orderedpos2_old[i,3])/dt
                vel_x2[h, r,l]+=(orderedpos2[i,3]-orderedpos2_old[i,3])/dt
                vel_y[h*len(cond_pos) + r,l]+=(orderedpos2[i,4]-orderedpos2_old[i,4])/dt
                vel_y2[h, r,l]+=(orderedpos2[i,4]-orderedpos2_old[i,4])/dt
                no_xy[h*len(cond_pos) + r,l]+=1
                no_xy2[h,r,l]+=1
                
                vel_x[ho*len(cond_pos) + ro,l]+=(orderedpos2[i,3]-orderedpos2_old[i,3])/dt
                vel_x2[ho, ro,l]+=(orderedpos2[i,3]-orderedpos2_old[i,3])/dt
                vel_y[ho*len(cond_pos) + ro,l]+=(orderedpos2[i,4]-orderedpos2_old[i,4])/dt
                vel_y2[ho, ro,l]+=(orderedpos2[i,4]-orderedpos2_old[i,4])/dt
                no_xy[ho*len(cond_pos) + ro,l]+=1
                no_xy2[ho,ro,l]+=1
             
# =============================================================================
#             ro=0
#             for ri in range(len(cond_pos)):
#                 if orderedpos2_old[i,3]>cond_midpos[ri] and orderedpos2_old[i,3]<cond_midpos[ri+1]:
#                 #if 0.5*(orderedpos2[i,3]+ orderedpos2_old[i,3])>cond_midpos[ri] and 0.5*(orderedpos2[i,3]+orderedpos2_old[i,3])<cond_midpos[ri+1]:
#                     ro=ri
#             
#             
#             if orderedpos2_old[i,4]<h_pos[19,r]:
#                 ho=20
#             elif  orderedpos2_old[i,4]>h_pos[0,r]:
#                 ho=0
#             for hi in range(19):
#                 if orderedpos2_old[i,4]>h_pos[hi+1,r] and orderedpos2_old[i,4]<h_pos[hi,r]:
#                     ho=hi+1
#                 
#             if np.abs(orderedpos2[i,3]-orderedpos2_old[i,3])<20:
#                     
#                 velo_x[ho*len(cond_pos) + ro,l]+=(orderedpos2[i,3]-orderedpos2_old[i,3])/(250000*1e-6)
#                 velo_x2[ho, ro,l]+=(orderedpos2[i,3]-orderedpos2_old[i,3])/(250000*1e-6)
#                 velo_y[ho*len(cond_pos) + ro,l]+=(orderedpos2[i,4]-orderedpos2_old[i,4])/(250000*1e-6)
#                 velo_y2[ho, ro,l]+=(orderedpos2[i,4]-orderedpos2_old[i,4])/(250000*1e-6)
#                 noo_xy[ho*len(cond_pos) + ro,l]+=1
#                 noo_xy2[ho,ro,l]+=1
# =============================================================================


                                
                    
for h in range((h_pos.shape[0]+1)):
    for r in range(len(cond_pos)):
        no_posx[(h*len(cond_pos))+r]=cond_pos[r]
        if h<16:
            no_posy[(h*len(cond_pos))+r]=h_pos[h,r]
            no_posxy[h,r]=[cond_pos[r],h_pos[h,r]]
        else:
            no_posy[(h*len(cond_pos))+r]=h_pos[h-1,r]-2
            no_posxy[h,r]=[cond_pos[r],h_pos[h-1,r]-2]
        for l in range(1,len(steps)):
            if no_xy[(h*len(cond_pos))+r,l]>50 :
                vel_x[(h*len(cond_pos))+r,l]=vel_x[(h*len(cond_pos))+r,l]/no_xy[(h*len(cond_pos))+r,l]
                vel_y[(h*len(cond_pos))+r,l]=vel_y[(h*len(cond_pos))+r,l]/no_xy[(h*len(cond_pos))+r,l]
                vel_x2[h,r,l]=vel_x2[h,r,l]/no_xy2[h,r,l]
                vel_y2[h,r,l]=vel_y2[h,r,l]/no_xy2[h,r,l]
                
            else:
                vel_x[(h*len(cond_pos))+r,l]=0
                vel_y[(h*len(cond_pos))+r,l]=0
                vel_x2[h,r,l]=0
                vel_y2[h,r,l]=0
                
# =============================================================================
#         for l in range(1,len(steps)):
#             if no_xy[(h*len(cond_pos))+r,l]>50 :
#                 vel_x[(h*len(cond_pos))+r,l]=vel_x[(h*len(cond_pos))+r,l]/no_xy[(h*len(cond_pos))+r,l]
#                 vel_y[(h*len(cond_pos))+r,l]=vel_y[(h*len(cond_pos))+r,l]/no_xy[(h*len(cond_pos))+r,l]
#                 vel_x2[h,r,l]=vel_x2[h,r,l]/no_xy2[h,r,l]
#                 vel_y2[h,r,l]=vel_y2[h,r,l]/no_xy2[h,r,l]
#                 
#             else:
#                 vel_x[(h*len(cond_pos))+r,l]=0
#                 vel_y[(h*len(cond_pos))+r,l]=0
#                 vel_x2[h,r,l]=0
#                 vel_y2[h,r,l]=0
# =============================================================================
# =============================================================================
#             if noo_xy[(h*len(cond_pos))+r,l]>10 :
#                 velo_x[(h*len(cond_pos))+r,l]=velo_x[(h*len(cond_pos))+r,l]/noo_xy[(h*len(cond_pos))+r,l]
#                 velo_y[(h*len(cond_pos))+r,l]=velo_y[(h*len(cond_pos))+r,l]/noo_xy[(h*len(cond_pos))+r,l]
#                 velo_x2[h,r,l]=velo_x2[h,r,l]/noo_xy2[h,r,l]
#                 velo_y2[h,r,l]=velo_y2[h,r,l]/noo_xy2[h,r,l]
# =============================================================================

# =============================================================================
# vel_x1=np.zeros(vel_x.shape,dtype=float)
# vel_y1=np.zeros(vel_y.shape,dtype=float)
# 
# for h in range(21):
#     for r in range(len(cond_pos)):
#         for l in range(4,len(steps)):
#             if no_xy[(h*len(cond_pos))+r,l]>0 or noo_xy[(h*len(cond_pos))+r,l]>0:
#                 vel_x1[(h*len(cond_pos))+r,l]=((vel_x[(h*len(cond_pos))+r,l]*no_xy[(h*len(cond_pos))+r,l])+ (velo_x[(h*len(cond_pos))+r,l]*noo_xy[(h*len(cond_pos))+r,l]))/(no_xy[(h*len(cond_pos))+r,l] + noo_xy[(h*len(cond_pos))+r,l])
#                 vel_y1[(h*len(cond_pos))+r,l]=((vel_y[(h*len(cond_pos))+r,l]*no_xy[(h*len(cond_pos))+r,l])+ (velo_y[(h*len(cond_pos))+r,l]*noo_xy[(h*len(cond_pos))+r,l]))/(no_xy[(h*len(cond_pos))+r,l] + noo_xy[(h*len(cond_pos))+r,l])
# 
# =============================================================================
# vel_x1=np.zeros(vel_x.shape,dtype=float)
# vel_y1=np.zeros(vel_y.shape,dtype=float)
no_posx2=no_posx.reshape((h_pos.shape[0]+1),len(cond_pos))
no_posy2=no_posy.reshape((h_pos.shape[0]+1),len(cond_pos))

vel_sm_x=np.zeros(vel_x.shape,dtype=float)
vel_sm_y=np.zeros(vel_y.shape,dtype=float)

vel_sm_x2=np.zeros(vel_x2.shape,dtype=float)
vel_sm_y2=np.zeros(vel_y2.shape,dtype=float)

#Smoothen Velocity
sm_l=5
noxy=np.zeros(((h_pos.shape[0]+1)*len(cond_pos)),dtype=float)
noxy2=np.zeros(no_xy2.shape,dtype=float)
for l in range(sm_l,len(steps)-sm_l):
    #vel_x[:,l]=0
    #vel_y[:,l]=0
    noxy[:]=0
    for lm in range(2*sm_l + 1):
        for ln in range((h_pos.shape[0]+1)*len(cond_pos)):
            vel_sm_x[ln,l]+=(vel_x[ln,l-sm_l+lm]*no_xy[ln,l-sm_l+lm])
            vel_sm_y[ln,l]+=(vel_y[ln,l-sm_l+lm]*no_xy[ln,l-sm_l+lm])
            noxy[ln]+=no_xy[ln,l-sm_l+lm]
        vel_sm_x2[:,:,l]+=vel_x2[:,:,l-sm_l+lm]*no_xy2[:,:,l-sm_l+lm]
        vel_sm_y2[:,:,l]+=vel_y2[:,:,l-sm_l+lm]*no_xy2[:,:,l-sm_l+lm]
        noxy2[:,:,l]+=no_xy2[:,:,l-sm_l+lm]
    vel_sm_x2[:,:,l]/=noxy2[:,:,l]
    vel_sm_y2[:,:,l]/=noxy2[:,:,l]
    for ln in range((h_pos.shape[0]+1)*len(cond_pos)):
        if noxy[ln]>30:#(no_xy[ln,l-1] +no_xy[ln,l] +no_xy[ln,l+1])>0:
            #if ln%10==0:
            #    print(noxy[ln])
            vel_sm_x[ln,l]/=noxy[ln]#(no_xy[ln,l-1] +no_xy[ln,l] +no_xy[ln,l+1])
            vel_sm_y[ln,l]/=noxy[ln]#(no_xy[ln,l-1] +no_xy[ln,l] +no_xy[ln,l+1])
vel_sm_x[:,:sm_l]=vel_x[:,:sm_l]
vel_sm_x[:,-sm_l:]=vel_x[:,-sm_l:]
vel_sm_y[:,:sm_l]=vel_y[:,:sm_l]
vel_sm_y[:,-sm_l:]=vel_y[:,-sm_l:]

vel_sm_x2[:,:,:sm_l]=vel_x2[:,:,:sm_l]
vel_sm_x2[:,:,-sm_l:]=vel_x2[:,:,-sm_l:]
vel_sm_y2[:,:,:sm_l]=vel_y2[:,:,:sm_l]
vel_sm_y2[:,:,-sm_l:]=vel_y2[:,:,-sm_l:]


#Vorticity
#Spatial derivatives
dvx_dy=np.zeros(vel_x2.shape,dtype=float)
dvy_dx=np.zeros(vel_y2.shape,dtype=float)
w=np.zeros(vel_x.shape,dtype=float) #Vorticity
for l in range(len(steps)-1):
    for i in range(1,vel_x2.shape[0]-1):
        for j in range(1,vel_y2.shape[1]-1):
            dvx_dy[i,j,l]=(vel_sm_x2[i+1,j,l]-vel_sm_x2[i-1,j,l])/(no_posy2[i+1,j]-no_posy2[i-1,j])
            vy1=((vel_sm_y2[i,j+1,l]-vel_sm_y2[i-1,j+1,l])*(no_posy2[i,j+1]-no_posy2[i,j])/(no_posy2[i,j+1]-no_posy2[i-1,j+1]))+ vel_sm_y2[i-1,j+1,l]
            vy2=((vel_sm_y2[i,j-1,l]-vel_sm_y2[i-1,j-1,l])*(no_posy2[i,j-1]-no_posy2[i,j])/(no_posy2[i,j-1]-no_posy2[i-1,j-1]))+ vel_sm_y2[i-1,j-1,l]
            #y = ((y2-y1)*(x1-x)/(x1-x2)) + y1
            #vy1=vel_sm_y2[i,j+1,l]
            #vy2=vel_sm_y2[i,j-1,l]
            dvy_dx[i,j,l]=(vy1-vy2)/(no_posx2[i,j+1]-no_posx2[i,j-1])

w=dvx_dy+dvy_dx



                      

SMALL_SIZE = 80
MEDIUM_SIZE = 20
BIGGER_SIZE = 80

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

colors1=np.zeros(((h_pos.shape[0]+1)*len(cond_pos)),dtype=float)
# =============================================================================
# for i in range(len(colors1)):
#     if i%12==0:
#         colors1[i]=0.9
#     elif i%12==1:
#         colors1[i]=0.5
#     elif i%12==2:
#         colors1[i]=0.25
# =============================================================================
        
colors1[:len(cond_pos)]=0.9
colors1[len(cond_pos):len(cond_pos)*2]=0.5
colors1[len(cond_pos)*2:len(cond_pos)*3]=0.25

colors2=colors1.reshape((h_pos.shape[0]+1),len(cond_pos))


colormap=cm.nipy_spectral
colors3=[cm.nipy_spectral(0.9),cm.nipy_spectral(0.5),cm.nipy_spectral(0.25)]     


def symlog(x):
    """ Returns the symmetric log10 value """
    return np.sign(x) * np.log(1+np.abs(x))

for l in range(18,20):#int(len(steps)/5)):

    plt.figure(figsize=(50,17))
    #plt.plot(val_perA[2200:26000,0],(val[2200:13000,5]))
    #plt.plot(val[2200:26000,0],(val[2200:52000,4]+85000))
    #plt.plot(val[2200:52000,0],(val[2200:52000,3]))
    #plt.plot(val_perA[:,0],(val_perA[:,5]))
    #plt.colorbar()
    ax=plt.quiver(no_posx,no_posy,(vel_sm_x[:,(l*10)+2]),(vel_sm_y[:,(l*10)+2]),color=colormap(colors1),width=0.0018,pivot='mid',label='%.2f-%.2fns'%(steps[(l*10)+2-1]*1e-6,(steps[(l*10)+2]+50000)*1e-6),scale=500)#, norm_vel)
    plt.xlim(-15,115)
    plt.ylim(-45,10)
    #plt.colorbar(ax)
    #plt.legend()
    plt.ylabel('Z(A)')
    plt.xlabel('Y(A)')
    plt.show()
    
    vel_sm_x22=vel_sm_x[:,l].reshape((h_pos.shape[0]+1),-1)
    vel_sm_y22=vel_sm_y[:,l].reshape((h_pos.shape[0]+1),-1)
    
# =============================================================================
#     plt.figure(figsize=(50,17))
#     #plt.plot(val_perA[2200:26000,0],(val[2200:13000,5]))
#     #plt.plot(val[2200:26000,0],(val[2200:52000,4]+85000))
#     #plt.plot(val[2200:52000,0],(val[2200:52000,3]))
#     #plt.plot(val_perA[:,0],(val_perA[:,5]))
#     #plt.colorbar()
#     ax=plt.quiver(no_posx2,no_posy2,(vel_sm_x2[:,:,l]),(vel_sm_y2[:,:,l]),width=0.002,pivot='mid',label='%.2f-%.2fns'%(steps[(l*1)-0-1]*1e-6,(steps[(l*1)-0]+50000)*1e-6),scale=500)#, norm_vel)
#     plt.xlim(-15,115)
#     plt.ylim(-45,10)
#     #plt.colorbar(ax)
#     #plt.legend()
#     plt.ylabel('Z(A)')
#     plt.xlabel('Y(A)')
#     plt.show()
# =============================================================================
    
# =============================================================================
#     plt.figure(figsize=(50,17))
#     #plt.plot(val_perA[2200:26000,0],(val[2200:13000,5]))
#     #plt.plot(val[2200:26000,0],(val[2200:52000,4]+85000))
#     #plt.plot(val[2200:52000,0],(val[2200:52000,3]))
#     #plt.plot(val_perA[:,0],(val_perA[:,5]))
#     #plt.colorbar()
#     ax=plt.quiver(no_posx2,no_posy2,(w[:,:,l*10]),(np.zeros((17,15),dtype=float)),width=0.002,pivot='mid',label='%.2f-%.2fns'%(steps[(l*10)-0-1]*1e-6,(steps[(l*10)-0]+50000)*1e-6),scale=100)#, norm_vel)
#     plt.xlim(-15,115)
#     plt.ylim(-45,10)
#     #plt.colorbar(ax)
#     #plt.legend()
#     plt.ylabel('Z(A)')
#     plt.xlabel('Y(A)')
#     plt.show()
# =============================================================================
    
    #from matplotlib.cm import ScalarMappable
    plt.figure(figsize=(50,17))
    #plt.plot(val_perA[2200:26000,0],(val[2200:13000,5]))
    #plt.plot(val[2200:26000,0],(val[2200:52000,4]+85000))
    #plt.plot(val[2200:52000,0],(val[2200:52000,3]))
    #plt.plot(val_perA[:,0],(val_perA[:,5]))
    #plt.colorbar()
    vmax=10
    vmin=-3
    n=6
    levels = np.linspace(vmin, vmax, n+1)
    ax=plt.contourf(no_posx2,no_posy2,(w[:,:,(l*10)+2]),levels=levels,extend='both',cmap='PuOr',label='%.2f-%.2fns'%(steps[(l*10)+2-1]*1e-6,(steps[(l*10)+2]+100000)*1e-6))#, norm_vel)
    #plt.imshow(no_posx2,no_posy2,(w[:,:,l*10]),origin='lower',interpolation='bilinear')
    plt.xlim(-15,115)
    plt.ylim(-45,10)
    #plt.colorbar(ax)
    #plt.legend()
    cbar=plt.colorbar(ax,ticks=range(vmin, vmax+5, 3))
    plt.ylabel('Z(A)')
    plt.xlabel('Y(A)')
    plt.show()
        
        
        
# =============================================================================
#         plt.figure(figsize=(40,20))
#         #plt.plot(val_perA[2200:26000,0],(val[2200:13000,5]))
#         #plt.plot(val[2200:26000,0],(val[2200:52000,4]+85000))
#         #plt.plot(val[2200:52000,0],(val[2200:52000,3]))
#         #plt.plot(val_perA[:,0],(val_perA[:,5]))
#         #plt.colorbar()
#         ax=plt.quiver(no_posx,no_posy,symlog(vel_sm_x[:,(l*1)-0+l1]), symlog(vel_sm_y[:,(l*1)-0+l1]),width=0.0015,pivot='mid',label='%.2f-%.2fns'%(steps[(l*1)-0+l1-1]*1e-6,steps[(l*1)-0+l1]*1e-6),scale=120)#, norm_vel)
#         #plt.colorbar(ax)
#         plt.legend()
#         plt.ylabel('Z(A)')
#         plt.xlabel('Y(A)')
#         plt.show()
# =============================================================================
