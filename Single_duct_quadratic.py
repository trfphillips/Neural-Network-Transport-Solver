import numpy as np
import os
from random import randint
from backend import import_excel
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Input, Dense,Conv2D, add,multiply,subtract, Cropping2D, ZeroPadding2D, Conv3D, TimeDistributed, RepeatVector, Reshape, DepthwiseConv2D, Lambda
from keras import Model
from tensorflow import keras
import tensorflow as tf
import shape_functions

tf.keras.backend.set_floatx('float64')
config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=4,
                        inter_op_parallelism_threads=4,
                        allow_soft_placement=True)

session = tf.compat.v1.Session(config=config)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=20480)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

#define problem parameters
nx = int((180/4)+3)
ny = int((140/4)+1)
boundary_nodes = 2
half_boundary = int(boundary_nodes/2)
sigmat = np.zeros((1,nx+boundary_nodes,ny+boundary_nodes))
k =np.zeros((1,nx+boundary_nodes,ny+boundary_nodes))
sigma_s_off =np.zeros((1,nx+boundary_nodes,ny+boundary_nodes))
sigma_f  = np.zeros((1,nx+boundary_nodes,ny+boundary_nodes))
dx = 36/nx
dy = 28/ny
dz = 10**15
dt = 10**15
v_n = np.ones((1))
s = np.zeros((1,nx+boundary_nodes,ny+boundary_nodes))
right_tube = int(nx/2)+(int(nx*3/36))+1
left_tube = int(nx/2)-(int(nx*3/36))+1
upper_tube = int(ny/2)+(int(ny*3/28))+1
bottom_tube = int(ny/2)-(int(ny*3/28))+1
extra_x = (nx*3/36)-int(nx*3/36)
extra_y = (ny*3/28)-int(ny*3/28)
s[0,left_tube:right_tube,bottom_tube:upper_tube] = 1
s[0,left_tube-1,bottom_tube:upper_tube] = extra_x
s[0,right_tube,bottom_tube:upper_tube] = extra_x
s[0,left_tube:right_tube,bottom_tube-1] = extra_y
s[0,left_tube:right_tube,upper_tube] = extra_y
sigmat[0,left_tube:right_tube,bottom_tube:upper_tube] = 0.5
sigmat[0,left_tube-1,bottom_tube:upper_tube] = extra_x*0.5
sigmat[0,right_tube,bottom_tube:upper_tube] = extra_x*0.5
sigmat[0,left_tube:right_tube,bottom_tube-1] = extra_y*0.5
sigmat[0,left_tube:right_tube,upper_tube] = extra_y*0.5
sigmat[0,:,:bottom_tube] = 0.5
sigmat[0,:,upper_tube:] = 0.5
sigmat[0,0,:]=0
sigmat[0,-1,:]=0
sigmat[0,:,0]=0
sigmat[0,:,-1]=0
ng = 1


bias_initializer = tf.keras.initializers.constant(np.zeros((1,)))
w1 = np.zeros([1, 2, 2, 1])
w1[0,:,:,0] = 1.0
w2 = np.zeros([1, 2, 2, 1])
w2[0][:][:][0] = 0.25
kernel_initializer_1 = tf.keras.initializers.constant(w1)
kernel_initializer_2 = tf.keras.initializers.constant(w2)

jacob_it = 4
multigrid_it =2
nface = 8
nside = 4

cube_sn_weight, cube_sn_direction = shape_functions.octant_sn_quadrature(4,4)
cube_sn_weight_120, cube_sn_direction_120 = shape_functions.octant_sn_quadrature(int(nside/2),int(nside/2))
cube_sn_weight_60, cube_sn_direction_60 = shape_functions.octant_sn_quadrature(int(nside/4),int(nside/4))
cube_sn_weight_120 = np.transpose(cube_sn_weight_120)
cube_sn_weight_60 = np.transpose(cube_sn_weight_60)
cube_sn_weight = np.transpose(cube_sn_weight)
u_s = np.transpose(cube_sn_direction[0,:,:,:])
u_s_120 = np.transpose(cube_sn_direction_120[0,:,:,:])
u_s_60 = np.transpose(cube_sn_direction_60[0,:,:,:])
v_s = np.transpose(cube_sn_direction[1,:,:,:])
v_s_120 = np.transpose(cube_sn_direction_120[1,:,:,:])
v_s_60 = np.transpose(cube_sn_direction_60[1,:,:,:])

mult = np.zeros((1,nface,nside,nside,s.shape[1],s.shape[2],1))
mult[0,:,:,:,1:-1,1:-1,:]=-1
half = np.zeros((1,nface,nside,nside,s.shape[1],s.shape[2],1))
half[0,:,:,:,1:-1,1:-1,:]=-0.5
v_small_val = np.zeros((1,nface,nside,nside,s.shape[1],s.shape[2],1))
v_small_val[0,:,:,:,1:-1,1:-1,:] = 0.01
mult_120 = np.zeros((1,nface,int(nside/2),int(nside/2),int((s.shape[1]-boundary_nodes)/2)+boundary_nodes,int((s.shape[2]-boundary_nodes)/2)+boundary_nodes,1))
mult_120[0,:,:,:,1:-1,1:-1,:]=-1
mult_60 = np.zeros((1,nface,int(nside/4),int(nside/4),int((s.shape[1]-boundary_nodes)/4)+boundary_nodes,int((s.shape[2]-boundary_nodes)/4)+boundary_nodes,1))
mult_60[0,:,:,:,1:-1,1:-1,:]=-1
u_s_p = np.zeros((1,nface,nside,nside,s.shape[1],s.shape[2],1))
u_s_m = np.zeros((1,nface,nside,nside,s.shape[1],s.shape[2],1))
v_s_p = np.zeros((1,nface,nside,nside,s.shape[1],s.shape[2],1))
v_s_m = np.zeros((1,nface,nside,nside,s.shape[1],s.shape[2],1))
u_s_full2 = np.zeros((1,nface,nside,nside,s.shape[1],s.shape[2],1))
v_s_full2 = np.zeros((1,nface,nside,nside,s.shape[1],s.shape[2],1))
u_s_p_120 = np.zeros((1,nface,int(nside/2),int(nside/2),int((s.shape[1]-boundary_nodes)/2)+boundary_nodes,int((s.shape[2]-boundary_nodes)/2)+boundary_nodes,1))
u_s_m_120 = np.zeros((1,nface,int(nside/2),int(nside/2),int((s.shape[1]-boundary_nodes)/2)+boundary_nodes,int((s.shape[2]-boundary_nodes)/2)+boundary_nodes,1))
v_s_p_120 = np.zeros((1,nface,int(nside/2),int(nside/2),int((s.shape[1]-boundary_nodes)/2)+boundary_nodes,int((s.shape[2]-boundary_nodes)/2)+boundary_nodes,1))
v_s_m_120 = np.zeros((1,nface,int(nside/2),int(nside/2),int((s.shape[1]-boundary_nodes)/2)+boundary_nodes,int((s.shape[2]-boundary_nodes)/2)+boundary_nodes,1))
u_s_p_60 = np.zeros((1,nface,int(nside/4),int(nside/4),int((s.shape[1]-boundary_nodes)/4)+boundary_nodes,int((s.shape[2]-boundary_nodes)/4)+boundary_nodes,1))
u_s_m_60 = np.zeros((1,nface,int(nside/4),int(nside/4),int((s.shape[1]-boundary_nodes)/4)+boundary_nodes,int((s.shape[2]-boundary_nodes)/4)+boundary_nodes,1))
v_s_p_60 = np.zeros((1,nface,int(nside/4),int(nside/4),int((s.shape[1]-boundary_nodes)/4)+boundary_nodes,int((s.shape[2]-boundary_nodes)/4)+boundary_nodes,1))
v_s_m_60 = np.zeros((1,nface,int(nside/4),int(nside/4),int((s.shape[1]-boundary_nodes)/4)+boundary_nodes,int((s.shape[2]-boundary_nodes)/4)+boundary_nodes,1))
u_s_full = np.zeros((1,nface,nside,nside,s.shape[1],s.shape[2],1))
v_s_full = np.zeros((1,nface,nside,nside,s.shape[1],s.shape[2],1))
u_s_max = np.zeros((1,nface,nside,nside,s.shape[1],s.shape[2],1))
u_s_max_x = np.zeros((1,nface,nside,nside,s.shape[1],s.shape[2],1))
u_s_max_y = np.zeros((1,nface,nside,nside,s.shape[1],s.shape[2],1))
v_s_max_x = np.zeros((1,nface,nside,nside,s.shape[1],s.shape[2],1))
v_s_max_y = np.zeros((1,nface,nside,nside,s.shape[1],s.shape[2],1))
u_s_full1 = np.zeros((1,nface,nside,nside,s.shape[1],s.shape[2],1))
v_s_full1 = np.zeros((1,nface,nside,nside,s.shape[1],s.shape[2],1))
v_s_max = np.zeros((1,nface,nside,nside,s.shape[1],s.shape[2],1))
weights = np.zeros((1,nface,nside,nside,s.shape[1],s.shape[2],1))
weights_120 = np.zeros((1,nface,int(nside/2),int(nside/2),int((s.shape[1]-boundary_nodes)/2)+boundary_nodes,int((s.shape[2]-boundary_nodes)/2)+boundary_nodes,1))
weights_60 = np.zeros((1,nface,int(nside/4),int(nside/4),int((s.shape[1]-boundary_nodes)/4)+boundary_nodes,int((s.shape[2]-boundary_nodes)/4)+boundary_nodes,1))
halo_matrix = np.ones((1,nface,nside,nside,s.shape[1],s.shape[2],1))
halo_matrix[0,:,:,:,0,:,0] = 0
halo_matrix[0,:,:,:,:,0,0] = 0
halo_matrix[0,:,:,:,-1,:,0] = 0
halo_matrix[0,:,:,:,:,-1,0] = 0
halo_matrix[0,0,:,:,0,:int(s.shape[1]/2),0] = 1
halo_matrix[0,0,:,:,:int(s.shape[1]/2),0,0] = 1
halo_matrix[0,1,:,:,-1,:int(s.shape[1]/2),0] = 1
halo_matrix[0,1,:,:,int(s.shape[1]/2):,0,0] = 1
halo_matrix[0,2,:,:,0,int(s.shape[1]/2):,0] = 1
halo_matrix[0,2,:,:,:int(s.shape[1]/2),-1,0] = 1
halo_matrix[0,3,:,:,-1,int(s.shape[1]/2):,0] = 1
halo_matrix[0,3,:,:,int(s.shape[1]/2):,-1,0] = 1
halo_matrix[0,4,:,:,0,:int(s.shape[1]/2),0] = 1
halo_matrix[0,4,:,:,:int(s.shape[1]/2),0,0] = 1
halo_matrix[0,5,:,:,-1,:int(s.shape[1]/2),0] = 1
halo_matrix[0,5,:,:,int(s.shape[1]/2):,0,0] = 1
halo_matrix[0,6,:,:,0,int(s.shape[1]/2):,0] = 1
halo_matrix[0,6,:,:,:int(s.shape[1]/2),-1,0] = 1
halo_matrix[0,7,:,:,-1,int(s.shape[1]/2):,0] = 1
halo_matrix[0,7,:,:,int(s.shape[1]/2):,-1,0] = 1
from math import pi
for i in range(nface):
    for j in range(nside):
        for k in range(nside):
            u_s_p[0, i, j, k, 1:-1, 1:-1] = u_s[i, j, k] / (dx)
            u_s_m[0, i, j, k, 1:-1, 1:-1] = u_s[i, j, k] / (dx)
            v_s_p[0, i, j, k, 1:-1, 1:-1] = v_s[i, j, k] / (dy)
            v_s_m[0, i, j, k, 1:-1, 1:-1] = v_s[i, j, k] / (dy)
            u_s_full[0, i, j, k, :, :,:] = u_s[i, j, k]
            v_s_full[0, i, j, k, :, :,:] = v_s[i, j, k]
            u_s_full1[0, i, j, k, :, :,:] = u_s[i, j, k] / dx
            v_s_full1[0, i, j, k, :, :,:] = v_s[i, j, k] / dy
            u_s_full2[0, i, j, k, :, :,:] = u_s[i, j, k] / (2 * dx)
            v_s_full2[0, i, j, k, :, :,:] = v_s[i, j, k] / (2 * dy)
            u_s_max_x[0,i,j,k,:,:]=abs(u_s[i,j,k])
            u_s_max_y[0, i, j, k, :, :] = abs(v_s[i,j,k])
            v_s_max_x[0, i, j, k, :, :] = -abs(u_s[i, j, k])
            v_s_max_y[0, i, j, k, :, :] = -abs(v_s[i, j, k])
            v_s_max[0,i,j,k,:,:]=(np.sqrt((u_s[i,j,k]**2)+(v_s[i,j,k]**2)))*((dy+dx)/2)*-1
            weights[0,i,j,k,:,:]=(cube_sn_weight[i,j,k])
for i in range(nface):
    for j in range(nside):
        for k in range(nside):
            u_s_p[0, i, j, k, 1:-1, 1:-1] = u_s[i, j, k] * (1 / dx)
            u_s_m[0, i, j, k, 1:-1, 1:-1] = u_s[i, j, k] * (1 / dx)
            v_s_p[0, i, j, k, 1:-1, 1:-1] = v_s[i, j, k] * (1 / dy)
            v_s_m[0, i, j, k, 1:-1, 1:-1] = v_s[i, j, k] * (1 / dy)
            weights[0, i, j, k, :, :] = (cube_sn_weight[i, j, k]) / (4 * pi)
for i in range(nface):
    for j in range(int(nside/2)):
        for k in range(int(nside/2)):
            u_s_p_120[0, i, j, k,1:-1,1:-1] = u_s_120[i, j, k] * (1 / (2 * dx))
            u_s_m_120[0, i, j, k, 1:-1,1:-1] = u_s_120[i, j, k] * (1 / (2 * dx))
            v_s_p_120[0, i, j, k,1:-1,1:-1] =v_s_120[i, j, k] * (1 / (2 * dx))
            v_s_m_120[0, i, j, k, 1:-1,1:-1]= v_s_120[i, j, k] * (1 / (2 * dx))
            weights_120[0, i, j, k,1:-1,1:-1] = (cube_sn_weight_120[i, j, k]) / (4 * pi)
for i in range(nface):
    for j in range(int(nside/4)):
        for k in range(int(nside/4)):
            u_s_p_60[0, i, j, k, 1:-1,1:-1] = u_s_60[i, j, k] * (1 / (4 * dx))
            u_s_m_60[0, i, j, k,1:-1,1:-1] = u_s_60[i, j, k] * (1 / (4 * dx))
            v_s_p_60[0, i, j, k, 1:-1,1:-1] = v_s_60[i, j, k] * (1 / (4 * dx))
            v_s_m_60[0, i, j, k, 1:-1,1:-1]= v_s_60[i, j, k] * (1 / (4 * dx))
            weights_60[0, i, j, k,1:-1,1:-1] = (cube_sn_weight_60[i, j, k]) / (4 * pi)
u_s_p[:,:,:,:,:,:]= np.where(u_s_p[:,:,:,:,:,:]<0,0,u_s_p[:,:,:,:,:,:])
u_s_m[:,:,:,:,:,:]= np.where(u_s_m[:,:,:,:,:,:]>0,0,u_s_m[:,:,:,:,:,:])
v_s_p[:,:,:,:,:,:]= np.where(v_s_p[:,:,:,:,:,:]<0,0,v_s_p[:,:,:,:,:,:])
v_s_m[:,:,:,:,:,:]= np.where(v_s_m[:,:,:,:,:,:]>0,0,v_s_m[:,:,:,:,:,:])
u_s_p_120[:,:,:,:,:,:]= np.where(u_s_p_120[:,:,:,:,:,:]<0,0,u_s_p_120[:,:,:,:,:,:])
u_s_m_120[:,:,:,:,:,:]= np.where(u_s_m_120[:,:,:,:,:,:]>0,0,u_s_m_120[:,:,:,:,:,:])
v_s_p_120[:,:,:,:,:,:]= np.where(v_s_p_120[:,:,:,:,:,:]<0,0,v_s_p_120[:,:,:,:,:,:])
v_s_m_120[:,:,:,:,:,:]= np.where(v_s_m_120[:,:,:,:,:,:]>0,0,v_s_m_120[:,:,:,:,:,:])
u_s_p_60[:,:,:,:,:,:]= np.where(u_s_p_60[:,:,:,:,:,:]<0,0,u_s_p_60[:,:,:,:,:,:])
u_s_m_60[:,:,:,:,:,:]= np.where(u_s_m_60[:,:,:,:,:,:]>0,0,u_s_m_60[:,:,:,:,:,:])
v_s_p_60[:,:,:,:,:,:]= np.where(v_s_p_60[:,:,:,:,:,:]<0,0,v_s_p_60[:,:,:,:,:,:])
v_s_m_60[:,:,:,:,:,:]= np.where(v_s_m_60[:,:,:,:,:,:]>0,0,v_s_m_60[:,:,:,:,:,:])
weights_res = np.copy(weights)*dx*dy
weights_res_120 = np.copy(weights_120)*(2*dx)*(2*dy)
weights_res_60 = np.copy(weights_60)*(4*dx)*(4*dy)

u_s_p_inv = 1/(u_s_p*2)
u_s_p_inv[np.isinf(u_s_p_inv)]=0
u_s_m_inv = 1/(u_s_m*2)
u_s_m_inv[np.isinf(u_s_m_inv)]=0
v_s_p_inv = 1/(v_s_p*2)
v_s_p_inv[np.isinf(v_s_p_inv)]=0
v_s_m_inv = 1/(v_s_m*2)
v_s_m_inv[np.isinf(v_s_m_inv)]=0

w2 = np.zeros([1 ,1,3, 3])
w2[0, 0, 0, 1] = 0
w2[0, 0,2, 1] = 0
w2[0, 0,1, 2] = 0
w2[0, 0,1, 0] = -1
print('x+',w2[0,0,:,:])
kernel_initializer_UP1 = tf.keras.initializers.constant(w2)
w2 = np.zeros([1 ,1,3, 3])
w2[0, 0, 0, 1] = 0
w2[0, 0,2, 1] = 0
w2[0, 0,1, 2] = 1
w2[0, 0,1, 0] = 0
print('x-',w2[0,0,:,:])
kernel_initializer_UM1 = tf.keras.initializers.constant(w2)
w2 = np.zeros([1 ,1,3, 3])
w2[0, 0, 0, 1] = 0
w2[0, 0,2, 1] = -1
w2[0, 0,1, 2] = 0
w2[0, 0,1, 0] = 0
print('y+',w2[0,0,:,:])
kernel_initializer_VP1 = tf.keras.initializers.constant(w2)
w2 = np.zeros([1 ,1,3, 3])
w2[0, 0, 0, 1] = 1
w2[0, 0,2, 1] = 0
w2[0, 0,1, 2] = 0
w2[0, 0,1, 0] = 0
print('y-',w2[0,0,:,:])
kernel_initializer_VM1 = tf.keras.initializers.constant(w2)
filt_5 = import_excel('5_filt')
w2 = np.zeros([1, 1, 5, 5])
w2[0,0,:,:]= filt_5[:5,:]* (1 / ((dy)))
kernel_initializer_diff5 = tf.keras.initializers.constant(w2)
w2 = np.zeros([1, 1, 5, 5])
w2[0,0,:,:]= filt_5[5:10,:]
adv_scal = 1#sum(sum(w2[0,0,:,2:]))
w2 = w2*(1/adv_scal)*(1/(dx))
kernel_initializer_diffx5 = tf.keras.initializers.constant(w2)
w2 = np.zeros([1, 1, 5, 5])
w2[0,0,:,:]= filt_5[10:15,:]* (-1 / (dy))
kernel_initializer_diffy5 = tf.keras.initializers.constant(w2)
w2 = np.zeros([1, 1, 5, 5])
w2[0,0,:,:]= filt_5[20:25,:]
adv_scal = 1#sum(sum(w2[0,0,:,2:]))
w2 = w2*(1/adv_scal)* (1 / (dx))
kernel_initializer_diff5_x = tf.keras.initializers.constant(w2)
w2 = np.zeros([1, 1, 5, 5])
w2[0,0,:,:]= filt_5[25:30,:]* (1 / (dy))
w2 =w2*(1/adv_scal)
print('diff_y',w2[0,0,:,:])
kernel_initializer_diff5_y = tf.keras.initializers.constant(w2)
w2 = np.zeros([1, 1, 5, 5])
w2[0,0,:,:]= filt_5[15:20,:]
print('mass',w2[0,0,:,:])
w2[0,0,2,2]=0
kernel_initializer_mass5 = tf.keras.initializers.constant(w2)

inv_a = np.zeros((1,nface,nside,nside,s.shape[1],s.shape[2],ng))
inv_a_2 = np.zeros((1,nface,int(nside/2),int(nside/2),int((s.shape[1]-boundary_nodes)/2)+boundary_nodes,int((s.shape[2]-boundary_nodes)/2)+boundary_nodes,ng))
inv_a_4 = np.zeros((1,nface,int(nside/4),int(nside/4),int((s.shape[1]-boundary_nodes)/4)+boundary_nodes,int((s.shape[2]-boundary_nodes)/4)+boundary_nodes,ng))
normal_a = np.zeros((1,nface,nside,nside,s.shape[1],s.shape[2],ng))

for i in range(nface):
    for j in range(nside):
        for k in range(nside):
            inv_a[0,i,j,k,half_boundary:-half_boundary,half_boundary:-half_boundary,0]=1/sigmat[0,half_boundary:-half_boundary,half_boundary:-half_boundary]
            normal_a[0,i,j,k,half_boundary:-half_boundary,half_boundary:-half_boundary,0]=sigmat[0,half_boundary:-half_boundary,half_boundary:-half_boundary]

inv_a[np.isinf(inv_a)]=0

res_240 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(int(((s.shape[1]-boundary_nodes))), int(((s.shape[2]-boundary_nodes))), 1)),
         tf.keras.layers.Conv2D(1, kernel_size=2, strides=2, padding='VALID',  # restriction
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer),  ])
res_120 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(int(((s.shape[1]-boundary_nodes)/2)), int(((s.shape[2]-boundary_nodes)/2)), 1)),
         tf.keras.layers.Conv2D(1, kernel_size=2, strides=2, padding='VALID',  # restriction
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer),  ])

half_boundary = 1
beta=2
boundary_nodes =2
diagonal_vals = np.zeros((1,nface,nside,nside,s.shape[1],s.shape[2],ng))
diagonal_vals_2 = np.zeros((1,nface,int(nside/2),int(nside/2),int((s.shape[1]-boundary_nodes)/2)+boundary_nodes,int((s.shape[2]-boundary_nodes)/2)+boundary_nodes,ng))
diagonal_vals_4 = np.zeros((1,nface,int(nside/4),int(nside/4),int((s.shape[1]-boundary_nodes)/4)+boundary_nodes,int((s.shape[2]-boundary_nodes)/4)+boundary_nodes,ng))
for i in range(nface):
    for j in range(int(nside/2)):
        for k in range(int(nside/2)):
            for g in range(ng):
                diagonal_vals_2[0,i,j,k,half_boundary:-half_boundary,half_boundary:-half_boundary,g] =  ((beta*(abs(u_s_120[i,j,k])+beta*abs(v_s_120[i,j,k])))/dx)+ beta*(np.array(res_240(sigmat[0,half_boundary:-half_boundary,half_boundary:-half_boundary].reshape(1, int(s.shape[1]-boundary_nodes), int(s.shape[2]-boundary_nodes), 1))).reshape(
                        int((s.shape[1]-boundary_nodes) / 2), int((s.shape[2]-boundary_nodes) / 2))  + abs(u_s_p_120[0,i, j, k,0,0]/(2 * dx)) + abs(v_s_p_120[0,i, j, k,0,0]/(2 * dy))+abs(u_s_m_120[0,i, j, k,0,0]/(2 * dx)) + abs(v_s_m_120[0,i, j, k,0,0]/(2 * dy))) + beta*(abs(u_s_120[i,j,k])/(dx**2)) + beta*(abs(v_s_120[i,j,k])/(dy**2))
                inv_a_2[0, i, j, k, half_boundary:-half_boundary, half_boundary:-half_boundary, g] = 1 / (np.array(res_240(
                    sigmat[0, half_boundary:-half_boundary, half_boundary:-half_boundary].reshape(1, int(
                        s.shape[1] - boundary_nodes), int(s.shape[2] - boundary_nodes), 1))).reshape(
                    int((s.shape[1] - boundary_nodes) / 2), int((s.shape[2] - boundary_nodes) / 2)) + abs(
                    u_s_p_120[0, i, j, k, 0, 0] / (2 * dx)) + abs(v_s_p_120[0, i, j, k, 0, 0] / (2 * dy)) + abs(
                    u_s_m_120[0, i, j, k, 0, 0] / (2 * dx)) + abs(
                    v_s_m_120[0, i, j, k, 0, 0] / (2 * dy)) + diagonal_vals_2[0,i,j,k,half_boundary:-half_boundary,half_boundary:-half_boundary,g])

for i in range(nface):
    for j in range(int(nside/4)):
        for k in range(int(nside/4)):
            for g in range(ng):
                diagonal_vals_4[0, i, j, k, half_boundary:-half_boundary, half_boundary:-half_boundary, g] = ((beta*(abs(u_s_60[i,j,k])+beta*abs(v_s_60[i,j,k])))/dx)+ beta*(np.array(res_120(res_240(sigmat[0,half_boundary:-half_boundary,half_boundary:-half_boundary].reshape(1, int(s.shape[1]-boundary_nodes), int(s.shape[2]-boundary_nodes), 1)))).reshape(
                        int((s.shape[1]-boundary_nodes) / 4), int((s.shape[2]-boundary_nodes) / 4)) + abs(u_s_p_60[0,i, j, k,0,0]/(4 * dx)) + abs(v_s_p_60[0,i, j, k,0,0]/(4 * dy)) + abs(u_s_m_120[0,i, j, k,0,0]/(4 * dx))  + abs(v_s_m_60[0,i, j, k,0,0]/(4 * dy))) + beta*(abs(u_s_60[i,j,k])/(dx**2)) + beta*(abs(v_s_60[i,j,k])/(dy**2))
                inv_a_4[0,i,j,k,half_boundary:-half_boundary,half_boundary:-half_boundary,g]=1/ (np.array(res_120(res_240(sigmat[0,half_boundary:-half_boundary,half_boundary:-half_boundary].reshape(1, int(s.shape[1]-boundary_nodes), int(s.shape[2]-boundary_nodes), 1)))).reshape(
                        int((s.shape[1]-boundary_nodes) / 4), int((s.shape[2]-boundary_nodes) / 4)) + abs(u_s_p_60[0,i, j, k,0,0]/(4 * dx)) + abs(v_s_p_60[0,i, j, k,0,0]/(4 * dy)) + abs(u_s_m_120[0,i, j, k,0,0]/(4 * dx))  + abs(v_s_m_60[0,i, j, k,0,0]/(4 * dy))+diagonal_vals_4[0, i, j, k, half_boundary:-half_boundary, half_boundary:-half_boundary, g])

for i in range(nface):
    for j in range(nside):
        for k in range(nside):
            # inv_a[0, i, j, k, half_boundary:-half_boundary,half_boundary:-half_boundary, 0] = 1 / (sigmat[0,half_boundary:-half_boundary,half_boundary:-half_boundary]+ abs(u_s[i, j, k] / dx) + abs(v_s[i, j, k] / dy)+diagonal_val)
            normal_a[0, i, j, k, half_boundary:-half_boundary,half_boundary:-half_boundary, 0] = sigmat[0,half_boundary:-half_boundary,half_boundary:-half_boundary]#+ abs(u_s[i, j, k] / dx) + abs(v_s[i, j, k] / dy)
            diagonal_vals[0, i, j, k, half_boundary:-half_boundary,half_boundary:-half_boundary, 0] = ((beta*(abs(u_s[i,j,k])+beta*abs(v_s[i,j,k])))/dx)+ beta*(
                        sigmat[0, half_boundary:-half_boundary, half_boundary:-half_boundary] + abs(
                    u_s[i, j, k] / dx) + abs(v_s[i, j, k] / dy)) + beta*(abs(u_s[i,j,k])/(dx**2)) + beta*(abs(v_s[i,j,k])/(dy**2))
            inv_a[0, i, j, k, half_boundary:-half_boundary, half_boundary:-half_boundary, 0] = 1 / (
                        sigmat[0, half_boundary:-half_boundary, half_boundary:-half_boundary] + abs(
                    u_s[i, j, k] / dx) + abs(v_s[i, j, k] / dy) + diagonal_vals[0, i, j, k, half_boundary:-half_boundary,half_boundary:-half_boundary, 0] )
inv_a[np.isinf(inv_a)]=0
inv_a_4[np.isinf(inv_a_4)]=0
#####Make model for 240x240
Input_img = Input(shape=(nface,4,4,s.shape[1], s.shape[2],1), name='flux_old')
Input_img2 = Input(shape=(nface,4,4,s.shape[1], s.shape[2],1), name='s')
Input_img5 = Input(shape=(nface,4,4,s.shape[1], s.shape[2],1), name='inv_a')

UPFlux = multiply([Input_img, u_s_p], name='u_s_p')
UMFlux = multiply([Input_img, u_s_m], name='u_m_p')
VPFlux = multiply([Input_img, v_s_p], name='v_s_p')
VMFlux = multiply([Input_img, v_s_m], name='v_s_m')

UPFlux =TimeDistributed(TimeDistributed(TimeDistributed(Conv2D(1, kernel_size=(3,3), strides=1, padding='SAME',  # A matrix
                  kernel_initializer=kernel_initializer_UP1, data_format='channels_last',
                  bias_initializer=bias_initializer, name='UPFlux'))))(UPFlux)
UMFlux =TimeDistributed(TimeDistributed(TimeDistributed(Conv2D(1, kernel_size=(3,3), strides=1, padding='SAME',  # A matrix
                  kernel_initializer=kernel_initializer_UM1, data_format='channels_last',
                  bias_initializer=bias_initializer, name='UMFlux'))))(UMFlux)
VPFlux =TimeDistributed(TimeDistributed(TimeDistributed(Conv2D(1, kernel_size=(3,3), strides=1, padding='SAME',  # A matrix
                  kernel_initializer=kernel_initializer_VP1, data_format='channels_last',
                  bias_initializer=bias_initializer, name='VPFlux'))))(VPFlux)
VMFlux =TimeDistributed(TimeDistributed(TimeDistributed(Conv2D(1, kernel_size=(3,3), strides=1, padding='SAME',  # A matrix
                  kernel_initializer=kernel_initializer_VM1, data_format='channels_last',
                  bias_initializer=bias_initializer, name='VMFlux'))))(VMFlux)


I_J_conv = add([UPFlux, UMFlux, VPFlux, VMFlux], name='I_J')
I_J_conv = multiply([I_J_conv, mult], name='mult5')
add_diagonal = multiply([Input_img, diagonal_vals], name='mult6')
I_J_conv = add([I_J_conv, Input_img2,add_diagonal], name='I_J2')
I_J_conv = multiply([I_J_conv, Input_img5], name='I_J_d')
model_240 = Model([Input_img, Input_img2, Input_img5], I_J_conv, name='model_240')


#####Make model for 120x120
Input_img = Input(shape=(nface,2,2,int((s.shape[1]-boundary_nodes)/2)+boundary_nodes,int((s.shape[2]-boundary_nodes)/2)+boundary_nodes,1), name='flux_old_120')
Input_img2 = Input(shape=(nface,2,2,int((s.shape[1]-boundary_nodes)/2)+boundary_nodes,int((s.shape[2]-boundary_nodes)/2)+boundary_nodes,1), name='s_120')
Input_img5 = Input(shape=(nface,2,2,int((s.shape[1]-boundary_nodes)/2)+boundary_nodes,int((s.shape[2]-boundary_nodes)/2)+boundary_nodes,1), name='inv_a_120')


UPFlux = multiply([Input_img, u_s_p_120], name='u_s_p_120')
UMFlux = multiply([Input_img, u_s_m_120], name='u_m_p_120')
VPFlux = multiply([Input_img, v_s_p_120], name='v_s_p_120')
VMFlux = multiply([Input_img, v_s_m_120], name='v_s_m_120')

UPFlux =TimeDistributed(TimeDistributed(TimeDistributed(Conv2D(1, kernel_size=(3,3), strides=1, padding='SAME',  # A matrix
                  kernel_initializer=kernel_initializer_UP1, data_format='channels_last',
                  bias_initializer=bias_initializer, name='UPFlux_120'))))(UPFlux)
UMFlux =TimeDistributed(TimeDistributed(TimeDistributed(Conv2D(1, kernel_size=(3,3), strides=1, padding='SAME',  # A matrix
                  kernel_initializer=kernel_initializer_UM1, data_format='channels_last',
                  bias_initializer=bias_initializer, name='UMFlux_120'))))(UMFlux)
VPFlux =TimeDistributed(TimeDistributed(TimeDistributed(Conv2D(1, kernel_size=(3,3), strides=1, padding='SAME',  # A matrix
                  kernel_initializer=kernel_initializer_VP1, data_format='channels_last',
                  bias_initializer=bias_initializer, name='VPFlux_120'))))(VPFlux)
VMFlux =TimeDistributed(TimeDistributed(TimeDistributed(Conv2D(1, kernel_size=(3,3), strides=1, padding='SAME',  # A matrix
                  kernel_initializer=kernel_initializer_VM1, data_format='channels_last',
                  bias_initializer=bias_initializer, name='VMFlux_120'))))(VMFlux)


I_J_conv = add([UPFlux, UMFlux, VPFlux, VMFlux], name='I_J_120')
I_J_conv = multiply([I_J_conv, mult_120], name='mult5_120')
add_diagonal = multiply([Input_img, diagonal_vals_2], name='mult6_120')
I_J_conv = add([I_J_conv, Input_img2,add_diagonal], name='I_J2_120')
I_J_conv = multiply([I_J_conv, Input_img5], name='I_J_d_120')
model_120 = Model([Input_img, Input_img2, Input_img5], I_J_conv, name='model_120')

######Make model for 60x60
Input_img = Input(shape=(nface,1,1,int((s.shape[1]-boundary_nodes)/4)+boundary_nodes,int((s.shape[2]-boundary_nodes)/4)+boundary_nodes,1), name='flux_old_60')
Input_img2 = Input(shape=(nface,1,1,int((s.shape[1]-boundary_nodes)/4)+boundary_nodes,int((s.shape[2]-boundary_nodes)/4)+boundary_nodes,1), name='s_60')
Input_img5 = Input(shape=(nface,1,1,int((s.shape[1]-boundary_nodes)/4)+boundary_nodes,int((s.shape[2]-boundary_nodes)/4)+boundary_nodes,1), name='inv_a_60')
UPFlux = multiply([Input_img, u_s_p_60], name='u_s_p_60')
UMFlux = multiply([Input_img, u_s_m_60], name='u_m_p_60')
VPFlux = multiply([Input_img, v_s_p_60], name='v_s_p_60')
VMFlux = multiply([Input_img, v_s_m_60], name='v_s_m_60')

print(UPFlux.shape)
UPFlux =TimeDistributed(TimeDistributed(TimeDistributed(Conv2D(1, kernel_size=(3,3), strides=1, padding='SAME',  # A matrix
                  kernel_initializer=kernel_initializer_UP1, data_format='channels_last',
                  bias_initializer=bias_initializer, name='UPFlux_60'))))(UPFlux)
UMFlux =TimeDistributed(TimeDistributed(TimeDistributed(Conv2D(1, kernel_size=(3,3), strides=1, padding='SAME',  # A matrix
                  kernel_initializer=kernel_initializer_UM1, data_format='channels_last',
                  bias_initializer=bias_initializer, name='UMFlux_60'))))(UMFlux)
VPFlux =TimeDistributed(TimeDistributed(TimeDistributed(Conv2D(1, kernel_size=(3,3), strides=1, padding='SAME',  # A matrix
                  kernel_initializer=kernel_initializer_VP1, data_format='channels_last',
                  bias_initializer=bias_initializer, name='VPFlux_60'))))(VPFlux)
VMFlux =TimeDistributed(TimeDistributed(TimeDistributed(Conv2D(1, kernel_size=(3,3), strides=1, padding='SAME',  # A matrix
                  kernel_initializer=kernel_initializer_VM1, data_format='channels_last',
                  bias_initializer=bias_initializer, name='VMFlux_60'))))(VMFlux)


I_J_conv = add([UPFlux, UMFlux, VPFlux, VMFlux], name='I_J_60')
I_J_conv = multiply([I_J_conv, mult_60], name='mult5_60')
add_diagonal = multiply([Input_img, diagonal_vals_4], name='mult6_60')
I_J_conv = add([I_J_conv, Input_img2,add_diagonal], name='I_J2_60')
I_J_conv = multiply([I_J_conv, Input_img5], name='I_J_d_60')
model_60 = Model([Input_img, Input_img2, Input_img5], I_J_conv, name='model_60')

############Make model to calculate residual
Input_img = Input(shape=(nface,4,4,s.shape[1], s.shape[2],1), name='flux_old_res')
Input_img2 = Input(shape=(nface,4,4,s.shape[1], s.shape[2],1), name='source_res')
Input_img3 = Input(shape=(nface,4,4,s.shape[1], s.shape[2],1), name='a_res')
from tensorflow.keras.layers import Layer
class Set_halo(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Set_halo, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        None

    def call(self, x, mask=None):
        self.W = tf.Variable(lambda: x)
        self.W[:,:,:,:,0,:,:].assign(x[:,:,:,:,1,:,:])
        self.W[:,:,:,:,-1,:,:].assign(x[:,:,:,:,-2,:,:])
        self.W[:,:,:,:,:,0,:].assign(x[:,:,:,:,:,1,:])
        self.W[:,:,:,:,:,-1,:].assign(x[:,:,:,:,:,-2,:])

        return self.W*halo_matrix
Input_img_mod = Set_halo((nface,4,4,s.shape[1], s.shape[2],1))(Input_img)
XFlux =TimeDistributed(TimeDistributed(TimeDistributed(Conv2D(1, kernel_size=(5,5), strides=1, padding='SAME',  # A matrix
                  kernel_initializer=kernel_initializer_diffx5, data_format='channels_last',
                  bias_initializer=bias_initializer, name='UPFlux'))))(Input_img_mod)
YFlux =TimeDistributed(TimeDistributed(TimeDistributed(Conv2D(1, kernel_size=(5,5), strides=1, padding='SAME',  # A matrix
                  kernel_initializer=kernel_initializer_diffy5, data_format='channels_last',
                  bias_initializer=bias_initializer, name='UMFlux'))))(Input_img_mod)

UPFlux = multiply([Input_img, u_s_p], name='u_s_p')
UMFlux = multiply([Input_img, u_s_m], name='u_m_p')
VPFlux = multiply([Input_img, v_s_p], name='v_s_p')
VMFlux = multiply([Input_img, v_s_m], name='v_s_m')


uXflux = multiply([XFlux, u_s_full], name='u_s_p')
vYflux = multiply([YFlux, v_s_full], name='u_m_p')
diag_term = multiply([Input_img, Input_img3], name='diag_res')
# diag_term = multiply([diag_term, mult], name='mult1_res')
central = add([uXflux,vYflux, diag_term], name='I_J1_res')
central = multiply([central, mult], name='mult1_res')
# central = multiply([central, mult], name='mult2_resasd')
r_actual = add([central, Input_img2], name='I_J_res_act')

gamma = (2**2)
# gamma = 0.0086
alpha_1y = (1/16)*gamma
alpha_1x= (1/16)*gamma
alpha_2x = (1/2)* gamma
alpha_2y = (1/2)*gamma

XFlux = tf.keras.backend.abs(XFlux)
YFlux = tf.keras.backend.abs(YFlux)

r_x =TimeDistributed(TimeDistributed(TimeDistributed(Conv2D(1, kernel_size=(5,5), strides=1, padding='SAME',  # A matrix
                  kernel_initializer=kernel_initializer_mass5, data_format='channels_last',
                  bias_initializer=bias_initializer, name='DYYFlux'))))(uXflux)
r_y =TimeDistributed(TimeDistributed(TimeDistributed(Conv2D(1, kernel_size=(5,5), strides=1, padding='SAME',  # A matrix
                  kernel_initializer=kernel_initializer_mass5, data_format='channels_last',
                  bias_initializer=bias_initializer, name='DYYFlux'))))(vYflux)
r_x =  Lambda(lambda x: x*(3/(dx*dy)))(r_x)
r_y =  Lambda(lambda x: x*(3/(dx*dy)))(r_y)
r_x = tf.keras.backend.abs(r_x)
r_y = tf.keras.backend.abs(r_y)

r_2_x = multiply([r_x, r_x], name='r_2i_re12s')
c_2term = tf.keras.backend.abs(uXflux)
c_2term = add([c_2term, v_small_val], name='r_i_res2')
r_2_x = Lambda(lambda x: x[0]/x[1])([r_2_x,c_2term])
r_2_x = Lambda(lambda x: x * alpha_2x)(r_2_x)
r_2_y = multiply([r_y, r_y], name='r_2i_r12es')
c_2term = tf.keras.backend.abs(vYflux)
c_2term = add([c_2term, v_small_val], name='r_i_res5')
r_2_y = Lambda(lambda x: x[0]/x[1])([r_2_y,c_2term])
r_2_y = Lambda(lambda x: x * alpha_2y)(r_2_y)

c_term = add([XFlux, v_small_val], name='r_i_res2')
c_term = tf.keras.backend.abs(c_term)
k_x = Lambda(lambda x: x[0]/x[1])([r_x,c_term])
k_x = Lambda(lambda x: x * alpha_1x)(k_x)
k_x = multiply([k_x, mult], name='k_i_6res')
k_x = add([k_x, u_s_max_x], name='k_i2nef')
k_x = keras.layers.ReLU()(k_x)
k_x = add([k_x, v_s_max_x], name='k_i2ad_resasd')
# k_x = add([k_x, r_2_x], name='k_i_resasd')
# k_x = keras.layers.ReLU()(k_x)
# k_2_x_n= multiply([r_2_x, mult], name='k_2iad_res')
# k_x = add([k_x, k_2_x_n], name='k_i_r7es')
k_x = multiply([k_x,half], name='k_i_end')
k_x = multiply([k_x, mult], name='k_i_end2')

c_term = add([YFlux, v_small_val], name='r_i_res2')
c_term = tf.keras.backend.abs(c_term)
k_y = Lambda(lambda x: x[0]/x[1])([r_y,c_term])
k_y = Lambda(lambda x: x * alpha_1y)(k_y)
k_y = multiply([k_y, mult], name='k_i_6resasd')
k_y = add([k_y, u_s_max_y], name='k_i2nefasd')
k_y = keras.layers.ReLU()(k_y)
k_y = add([k_y, v_s_max_y], name='k_i2ad_resasd')
# k_y = add([k_y, r_2_y], name='k_i_resads')
# k_y = keras.layers.ReLU()(k_y)
# k_2_y_n= multiply([r_2_y, mult], name='k_2iad_resasd')
# k_y = add([k_y, k_2_y_n], name='k_i_r7eadss')
k_y = multiply([k_y,half], name='k_i_enadd')
k_y = multiply([k_y, mult], name='k_i_end2')

ConvFlux = TimeDistributed(TimeDistributed(TimeDistributed(Conv2D(1, kernel_size=5, strides=1, padding='SAME',  # A matrix
                  kernel_initializer=kernel_initializer_diff5_x, data_format='channels_last',
                  bias_initializer=bias_initializer, name='KConvFlux'))))(Input_img_mod)
ConvFlux = multiply([ConvFlux, k_x], name='mult1')
Sumkflux = multiply([Input_img_mod, k_x], name='mult2')
Sumkflux = TimeDistributed(TimeDistributed(TimeDistributed(Conv2D(1, kernel_size=5, strides=1, padding='SAME',  # A matrix
                  kernel_initializer=kernel_initializer_diff5_x, data_format='channels_last',
                  bias_initializer=bias_initializer, name='Sumkflux'))))(Sumkflux)
fluxK = TimeDistributed(TimeDistributed(TimeDistributed(Conv2D(1, kernel_size=5, strides=1, padding='SAME',  # A matrix
                  kernel_initializer=kernel_initializer_diff5_x, data_format='channels_last',
                  bias_initializer=bias_initializer, name='fluxK'))))(k_x)
fluxK = multiply([Input_img_mod, fluxK], name='mult3')
fluxK = multiply([fluxK, mult], name='mult5')
central2 = add([ConvFlux, Sumkflux, fluxK], name='central')
I_J_conv = add([r_actual, central2], name='I_J_res')

ConvFlux = TimeDistributed(TimeDistributed(TimeDistributed(Conv2D(1, kernel_size=5, strides=1, padding='SAME',  # A matrix
                  kernel_initializer=kernel_initializer_diff5_y, data_format='channels_last',
                  bias_initializer=bias_initializer, name='KConvFlux_y'))))(Input_img_mod)
ConvFlux = multiply([ConvFlux, k_y], name='mult1_y')
Sumkflux = multiply([Input_img_mod, k_y], name='mult2_y')
Sumkflux = TimeDistributed(TimeDistributed(TimeDistributed(Conv2D(1, kernel_size=5, strides=1, padding='SAME',  # A matrix
                  kernel_initializer=kernel_initializer_diff5_y, data_format='channels_last',
                  bias_initializer=bias_initializer, name='Sumkflux_y'))))(Sumkflux)
fluxK = TimeDistributed(TimeDistributed(TimeDistributed(Conv2D(1, kernel_size=5, strides=1, padding='SAME',  # A matrix
                  kernel_initializer=kernel_initializer_diff5_y, data_format='channels_last',
                  bias_initializer=bias_initializer, name='fluxK_y'))))(k_y)
fluxK = multiply([Input_img_mod, fluxK], name='mult3_y')
fluxK = multiply([fluxK, mult], name='mult5_y')
central = add([ConvFlux, Sumkflux, fluxK], name='central_y')
I_J_conv = add([I_J_conv, central], name='I_J_res_y')
res_model = Model([Input_img, Input_img2,Input_img3], [I_J_conv,k_x,k_y],name='calc_res')

###### MULTIGRID MODEL
Input_flux_old = Input(shape=(nface,4,4,s.shape[1], s.shape[2],1), name='flux_old')
Input_source =Input(shape=(nface,4,4,s.shape[1], s.shape[2],1), name='source')
Input_inv_a= Input(shape=(nface,4,4,s.shape[1], s.shape[2],1), name='Input_inv_a')
Input_a= Input(shape=(nface,4,4,s.shape[1], s.shape[2],1), name='Input_a')
Input_inv_a_120= Input(shape=(nface,2,2,int(((s.shape[1]-boundary_nodes)/2)+boundary_nodes), int(((s.shape[2]-boundary_nodes)/2)+boundary_nodes), 1), name='Input_inv_a_120')
Input_inv_a_60= Input(shape=(nface,1,1,int(((s.shape[1]-boundary_nodes)/4)+boundary_nodes), int(((s.shape[2]-boundary_nodes)/4)+boundary_nodes), 1), name='Input_inv_a_60')


Residual_act, k_i, central = res_model([Input_flux_old, Input_source,Input_a])
Residual = multiply([Residual_act, weights_res], name='Residual')
Residual = TimeDistributed(TimeDistributed(TimeDistributed(Cropping2D(cropping=((half_boundary, half_boundary), (half_boundary, half_boundary))))))(Residual)
Residual_120 = TimeDistributed(TimeDistributed(TimeDistributed(Conv2D(1, kernel_size=2, strides=2, padding='VALID',  # A matrix
                  kernel_initializer=kernel_initializer_1,
                  bias_initializer=bias_initializer, name='64_res'))))(Residual)

Residual_120 = tf.transpose(Residual_120,perm=[0,1,5,4,3,2,6])
Residual_120 = TimeDistributed(TimeDistributed(TimeDistributed(Conv2D(1, kernel_size=2, strides=2, padding='VALID',  # A matrix
                  kernel_initializer=kernel_initializer_1,
                  bias_initializer=bias_initializer, name='64_res'))))(Residual_120)
Residual_120 = tf.transpose(Residual_120,perm=[0,1,5,4,3,2,6])
Residual_120 = TimeDistributed(TimeDistributed(TimeDistributed(ZeroPadding2D(padding=((half_boundary, half_boundary), (half_boundary, half_boundary))))))(Residual_120)
Residual_120_weights = multiply([Residual_120, weights_res_120], name='Residual_120_weights')
Residual_120_weights = TimeDistributed(TimeDistributed(TimeDistributed(Cropping2D(cropping=((half_boundary, half_boundary), (half_boundary, half_boundary))))))(Residual_120_weights)
Residual_60 = TimeDistributed(TimeDistributed(TimeDistributed(Conv2D(1, kernel_size=2, strides=2, padding='VALID',  # A matrix
                    kernel_initializer=kernel_initializer_1,
                    bias_initializer=bias_initializer, name='64_res'))))(Residual_120_weights)
Residual_60 = tf.transpose(Residual_60,perm=[0,1,5,4,3,2,6])
Residual_60 = TimeDistributed(TimeDistributed(TimeDistributed(Conv2D(1, kernel_size=2, strides=2, padding='VALID',  # A matrix
                    kernel_initializer=kernel_initializer_1,
                    bias_initializer=bias_initializer, name='64_res'))))(Residual_60)
Residual_60 = tf.transpose(Residual_60,perm=[0,1,5,4,3,2,6])
Residual = TimeDistributed(TimeDistributed(TimeDistributed(ZeroPadding2D(padding=((half_boundary, half_boundary), (half_boundary, half_boundary))))))(Residual)
Residual_60 = TimeDistributed(TimeDistributed(TimeDistributed(ZeroPadding2D(padding=((half_boundary, half_boundary), (half_boundary, half_boundary))))))(Residual_60)
Input_60 = tf.zeros(shape=(1,nface,1,1,int(((s.shape[1]-boundary_nodes)/4)+boundary_nodes), int(((s.shape[2]-boundary_nodes)/4)+boundary_nodes), 1))
delta_flux_60 = model_60([Input_60, Residual_60,Input_inv_a_60])
for i in range(jacob_it-1):
    delta_flux_60 = model_60([delta_flux_60, Residual_60,Input_inv_a_60])
delta_flux_60 =  TimeDistributed(TimeDistributed(TimeDistributed(Cropping2D(cropping=((half_boundary, half_boundary), (half_boundary, half_boundary))))))(delta_flux_60)
delta_flux_120 =  TimeDistributed(TimeDistributed(TimeDistributed(tf.keras.layers.UpSampling2D(size=(2, 2)))))(delta_flux_60)
delta_flux_120 = tf.transpose(delta_flux_120,perm=[0,1,5,4,3,2,6])
delta_flux_120 =  TimeDistributed(TimeDistributed(TimeDistributed(tf.keras.layers.UpSampling2D(size=(2, 2)))))(delta_flux_120)

delta_flux_120 = tf.transpose(delta_flux_120,perm=[0,1,5,4,3,2,6])
delta_flux_120 = TimeDistributed(TimeDistributed(TimeDistributed(ZeroPadding2D(padding=((half_boundary, half_boundary), (half_boundary, half_boundary))))))(delta_flux_120)
print(u_s_m_120.shape, u_s_p_120.shape, v_s_p_120.shape, v_s_m_120.shape)
delta_flux_120 = model_120([delta_flux_120, Residual_120,Input_inv_a_120])
for i in range(jacob_it-1):
    delta_flux_120 = model_120([delta_flux_120, Residual_120,Input_inv_a_120])
delta_flux_120 =  TimeDistributed(TimeDistributed(TimeDistributed(Cropping2D(cropping=((half_boundary, half_boundary), (half_boundary, half_boundary))))))(delta_flux_120)
delta_flux_240 =  TimeDistributed(TimeDistributed(TimeDistributed(tf.keras.layers.UpSampling2D(size=(2, 2)))))(delta_flux_120)
delta_flux_240 = tf.transpose(delta_flux_240,perm=[0,1,5,4,3,2,6])
delta_flux_240 =  TimeDistributed(TimeDistributed(TimeDistributed(tf.keras.layers.UpSampling2D(size=(2, 2)))))(delta_flux_240)

delta_flux_240 = tf.transpose(delta_flux_240,perm=[0,1,5,4,3,2,6])
delta_flux_240 = TimeDistributed(TimeDistributed(TimeDistributed(ZeroPadding2D(padding=((half_boundary, half_boundary), (half_boundary, half_boundary))))))(delta_flux_240)
print(delta_flux_240.shape)
print(Residual.shape)
print(Input_inv_a.shape)
delta_flux = model_240([delta_flux_240, Residual_act, Input_inv_a])
for i in range(jacob_it-1):
    delta_flux = model_240([delta_flux, Residual_act, Input_inv_a])
flux = add([Input_flux_old, delta_flux], name='flux')
flux = model_240([flux, Input_source,Input_inv_a])
print(flux.shape)
multi_model = Model([Input_flux_old,Input_source,Input_inv_a,Input_a,Input_inv_a_120,Input_inv_a_60],[flux,Residual_act,k_i,central],name='multi_model')
multi_model.summary()
source = np.zeros((1,nface,nside,nside,s.shape[1],s.shape[2],1))
flux = np.zeros((1,nface,nside,nside,s.shape[1],s.shape[2],1))
flux[0,0,:]=0
flux[0,-1,:]=0
flux[0,:,0]=0
flux[0,:,-1]=0
Input_inv_a = inv_a
Input_a = normal_a
Input_inv_a_120 = inv_a_2
Input_inv_a_60 = inv_a_4
for i in range(nface):
    for j in range(nside):
        for k in range(nside):
            source[0,i,j,k,:,:,0]=s[0,:,:]
            # flux[0,i,j,k,:,:,0]=np.load('Figures/Single_duct_norm_hh_mass/flux_1.npy')
try:
    flux = np.load('Figures/Single_duct_norm_h_mass010/flux_ang.npy')
except:
    print('could not load old flux')
    None

residual_all = np.zeros((1000))
k_i_all= np.zeros((1000))
flux, residual,k_i, central = multi_model([flux, source, (Input_inv_a),(Input_a),(Input_inv_a_120),(Input_inv_a_60)])
residual_all[0] = sum(sum(sum(sum(sum(sum(np.array(residual)))))))
k_i_all[0] = sum(sum(sum(sum(sum(sum(np.array(k_i)))))))
print('res:',residual_all[0])
print('k:',k_i_all[0])
print('central:',sum(sum(sum(sum(sum(sum(np.array(central))))))))
for i in range(1000-1):
    flux_new, residual,k_i, central = multi_model([flux, source, (Input_inv_a),(Input_a),(Input_inv_a_120),(Input_inv_a_60)])
    residual_all[i] = sum(sum(sum(sum(sum(sum(np.array(residual)))))))
    k_i_all[i] = sum(sum(sum(sum(sum(sum(np.array(k_i)))))))
    print('res:',residual_all[i])
    print('x_diff:',k_i_all[i])
    print('y_diff:',sum(sum(sum(sum(sum(sum(np.array(central))))))))
    # if abs((residual_all[i])>abs(residual_all[i-1])) and (i >1):
    #     break
    # else:
    flux = np.copy(flux_new)
flux_new = flux*weights
scalar_flux = np.zeros((s.shape[1],s.shape[2]))
for i in range(nface):
    for j in range(nside):
        for k in range(nside):
            scalar_flux = scalar_flux + flux_new[0,i,j,k,:,:,0]
plt.plot(residual_all)
output_dir = 'Figures/Single_duct_norm_h_mass010'
os.system('mkdir '+output_dir)
plt.savefig(output_dir+'/flux_res.png')
plt.close()
os.system('mkdir '+output_dir)
np.save(output_dir+'/flux_1.npy',scalar_flux)
np.save(output_dir+'/flux_ang.npy',flux)
cs = plt.contourf(scalar_flux[:,:],levels=100)
plt.colorbar(cs, shrink=0.9)
plt.savefig(output_dir+'/flux_2d.png')
plt.close()
plt.plot(scalar_flux[int(s.shape[1]/2):-boundary_nodes,int(s.shape[2]/2)])
plt.savefig(output_dir+'/flux_1d.png')
plt.close()
scalar_flux_max = max(np.array(scalar_flux).flatten())
scalar_flux = (np.array(scalar_flux)/scalar_flux_max)*1.755
plt.plot(scalar_flux[int(s.shape[1]/2):-2,int(s.shape[2]/2)])
x = [0.509,2.717,4.755,7.132,8.83,10.528,12.566,13.585,14.604,15.623,15.962,16.981,18.679,22.075,23.774,25.811,28.528,32.943,36,40.755,43.132,46.868,51.962,56.377,61.132,62.83,66.566,70.302,74.717,78.113,80.83,84.566,86]
y = [1.755,1.742,1.718,1.681,1.583,1.509,1.374,1.178,1.018,0.798,0.847,0.687,0.601,0.577,0.54,0.466,0.442,0.368,0.319,0.294,0.258,0.258,0.245,0.209,0.184,0.172,0.16,0.172,0.147,0.135,0.135,0.135,0.135]
plt.plot(x,y)
plt.savefig(output_dir+'/flux_1d_comp.png')
plt.close()
