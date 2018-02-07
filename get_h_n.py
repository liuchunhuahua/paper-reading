  def get_h_n(self,lstm_y_output ,true_y_length):
    ''' lstm_y_output: A 3D Tensor of shape(sey_len, batch_size,hidden_units)(list will work)
      true_y_length:(batch_size)
    '''
    hn_list=[]
    for i in range(self.config.batch_size):
      i=tf.cast(i,tf.int32)

      last_step=tf.cast(true_y_length[i]-1,tf.int32)

      hn_list.append(lstm_y_output[last_step, i, :])
      
    hn_tensor=tf.convert_to_tensor(hn_list)
    h_n=tf.reshape(hn_tensor,[tf.shape(hn_tensor)[0], tf.shape(hn_tensor)[-1]])         
    return h_n

  
 
import tensorflow as tf

def last_hidden(inp,true_len,max_len ):
  ''' 
  inp: a tensor of [b,l,d]
  true_len: a tensor of [b]
  max_len: a scalar of max sequence len
  ''' 
  batch_size = inp.get_shape().as_list()[0] 
  len_mask = tf.cast(tf.one_hot(indices=true_len-1,depth=max_len),tf.bool) #(b,l)
  # be care for that the true_len must be less than max_len, or else tf.boolean_mask will miss some data 
  last_flatten = tf.boolean_mask(tensor= inp, mask=len_mask) 
  last = tf.reshape(last_flatten,[batch_size,-1])
  return last

l = tf.constant([2,3,4])
inp = tf.constant([[1,2,0,0,0],[2,2,1,0,0],[3,3,3,5,0]])

xmaxlen=5
res = last_hidden(inp=inp, true_len=l,max_len=xmaxlen)

with tf.Session() as sess:
  print (sess.run(res))                        
