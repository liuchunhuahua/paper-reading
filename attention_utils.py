import tensorflow as tf

# dot-product way1
# x_sen:(b,x_len,h)   y_sen:(b,y_len,h)
if att_genre=='dot_product':  
  weight_matrix =tf.matmul(x_sen, tf.transpose(y_sen,perm=[0,2,1]))

#dot-product way2
if att_genre =='dot_product':  
  x_sen_4d = tf.expand_dims(x_sen,2)  
  y_sen_4d = tf.expand_dims(y_sen,1)  
  weight_matrix = tf.reduce_sum(tf.multiply(x_sen_4d,y_sen_4d),axis=-1)
  
  
