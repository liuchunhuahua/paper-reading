'''
date: 20180104
author: chunhua liu
note: works on tf.__version__ = '1.4.0' , '1.3.0' does not work
'''
import numpy as np
import tensorflow as tf

from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import check_ops

def _maybe_mask_score(score, true_sequence_length, score_mask_value,transpose=False):
  '''
  Args：
  transpose: if transpose=True, transpoe the last 2 dims at ethe begining and end
  '''
  if true_sequence_length is None:
    return score
  message = ("All values in true_sequence_length must greater than zero.")

  if transpose ==True: score = tf.transpose(score,[0,2,1])
  with ops.control_dependencies(
      [check_ops.assert_positive(true_sequence_length, message=message)]):

    extend_dim = score.get_shape().as_list()[1]

    score_mask =tf.tile( tf.expand_dims(array_ops.sequence_mask(
        true_sequence_length, maxlen=array_ops.shape(score)[-1]),1),[1,extend_dim,1])

    score_mask_values = score_mask_value * array_ops.ones_like(score)

    print ("score_mask",score_mask)
    #print ("score",score)
    print ("score",tf.convert_to_tensor(score))
    print ("score_mask_values",score_mask_values)
    result =  array_ops.where(score_mask, score, score_mask_values)
    if transpose==True: result = tf.transpose(result,[0,2,1])
    #return result, score_mask,score,score_mask_values
    return result 

def score_mask(score,true_len):
  '''
  Function: mask the final dim of score
  Args:
    score:(b,maxlen1,maxlen2)
    true_len1:(b,)
    true_len2:(b,)
  Returns:
    masked score
  '''
  #mask_value = -np.inf
  mask_value = -1e40

  score_tras = array_ops.transpose(score,perm = [0,2,1]) 
  col_mask = _maybe_mask_score(score= score_tras,true_sequence_length=true_len2,score_mask_value =mask_value)
  #col_mask = _maybe_mask_score(score= row_mask_tras,true_sequence_length=true_len2,score_mask_value =mask_value)
  col_mask_tras = array_ops.transpose(col_mask,perm = [0,2,1])

  #return col_mask_tras
  return col_mask_tras

def score_sub(x,y,x_true_len,y_true_len): 
  '''
  x:(b,xmax_len,d)
  y:(b,ymax_len,d)
  x_true_len:(b,)
  y_true_len:(b,)
  '''
  x = tf.expand_dims(x_raw,axis=2)          #(b,xmax_len,1,d)               
  y = tf.expand_dims(y_raw,axis=1)          #(b,1,ymax_len,d)               

  x_sub_y = x-y                             #(b,xmax_len,ymax_len,d) 
  sub_score_raw = tf.abs( tf.reduce_sum(x_sub_y,axis=-1))  #(b,xmax_len,ymax_len)

  sub_score_mask = _maybe_mask_score(score= sub_score_raw,true_sequence_length=y_true_len,score_mask_value = -1e40)

  #sub_score_mask = score_mask(sub_score_raw ,x_true_len,y_true_len)
  sub_score_norm = tf.nn.softmax(sub_score_mask)
  
  score_mask = _maybe_mask_score(score= sub_score_norm,
                                 true_sequence_length=x_true_len,
                                 score_mask_value = 0,
                                 transpose=True)
  #return sub_score_norm
  return score_mask


x_raw =tf.constant( [[[0.6,0.2],[0.1,0.2],[0.9,0.5],[0.0,0.0]]])      #(1,4,2)
y_raw = tf.constant([[[0.6,0.2],[0.7,0.3],[0.0,0.0]]] )               #(1,3,2)

x_true_len = tf.constant( [3])
y_true_len = tf.constant( [2])


diff_score_xy = score_sub(x_raw,y_raw,x_true_len,y_true_len)
diff_weighted_y = tf.matmul(diff_score_xy,y_raw)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  #res = sess.run([res,res2])
  res = sess.run([diff_score_xy,diff_weighted_y])
  #res = sess.run([diff_score_xy])
  for r in res:
    if len(r)>1:
      for x in r:
        print (x)
    else:
      print (r)



