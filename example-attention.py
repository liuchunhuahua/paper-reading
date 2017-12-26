# date: 20171228
# function: comprare tf.matmul and tf.reduce_sum(tf.multiply()) to realize attention
# conclusion: tf.matmul is obviously simpler to realize and easer to understand!
import tensorflow as tf
import sys
import numpy as np


p = tf.constant([[0.1,0.2,0.3],[0.4,0.1,0.5],[0.2,0.7,0.3]])  #(3,3)
h = tf.constant([[0.1,0.2,0.3],[0.6,0.1,0.4]])                #(2,3)
ph_diff_3d = tf.expand_dims(p,1) - tf.expand_dims(h,0)        #(3,2,3)
ph_diff_2d = tf.reduce_sum(ph_diff_3d,axis=-1)                #(3,2)  

#a= tf.constant([[0,-0.5],[0.4,-0.1],[0.6,0.1]])
a_exp = tf.exp(ph_diff_2d)

a_sum = tf.reduce_sum(a_exp,axis=0,keep_dims=True)

alpha =  a_exp/a_sum      #(3,2)
alpha_trans = tf.transpose(a_exp/a_sum,[1,0])        #(2,3)
#alpha = tf.nn.softmax(a,dim=0)

att_p = tf.matmul(alpha_trans,p)  #(2,3)(3,3)
## notice that the axis of tf.reduce_sum is the lenth dim 1, not -1
att_p1 = tf.reduce_sum(tf.expand_dims(p,0)*tf.expand_dims(alpha_trans,-1),axis=1)   #(1,3,3)(2,3,1)=>(2,3,3)=>(2,3)


with tf.Session() as sess:
  res = sess.run([alpha_trans,att_p,att_p1])
  for x in res:
    print (x)
    print ("\n")



'''
#numpy example
#h = np.array([[0.1,0.2,0.3],[0.6,0.1,0.4]])                #(2,3)
#h_3d = np.array([[[0.1],[0.2],[0.3]],[[0.6],[0.1],[0.4]]])                #(2,3,1)

p = np.array([[0.1,0.2,0.3],[0.4,0.1,0.5],[0.2,0.7,0.3]]) #(3,3) 
p_3d = np.array([[[0.1,0.2,0.3],[0.4,0.1,0.5],[0.2,0.7,0.3]]]) #(1,3,3)
a = np.array([[0.23,0.35,0.42],[0.23,0.35,0.42]])              #(2,3)
a_3d = np.array([[[0.23],[0.35],[0.42]],[[0.23],[0.35],[0.42]]]) #(2,3,1)

r_3d = p_3d*a_3d
r = np.sum(r_3d,axis=1)
print (p_3d,"\n")
print (a_3d,"\n")
print (r_3d,"\n")
print (r)
#print (0.23*p[0] + 0.35*p[1]+0.42*p[2])
sys.exit()
'''
