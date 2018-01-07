

#https://github.com/YichenGong/Densely-Interactive-Inference-Network/blob/master/python/models/DIIN.py
## Fucntion for embedding lookup and dropout at embedding layer
def emb_drop(E, x):
  emb = tf.nn.embedding_lookup(E, x)
  emb_drop = tf.cond(self.is_train, lambda: tf.nn.dropout(emb, config.keep_rate), lambda: emb)
  return emb_drop
# huahua: 忽然觉得利用tf.cond可以使代码变得很整洁。
