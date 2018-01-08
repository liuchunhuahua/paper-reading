
# https://github.com/YichenGong/Densely-Interactive-Inference-Network/blob/master/python/my/tensorflow/nn.py

def highway_layer(arg, bias, bias_start=0.0, scope=None, wd=0.0, input_keep_prob=1.0, is_train=None, output_size = None):
    with tf.variable_scope(scope or "highway_layer"):
        if output_size is not None:
            d = output_size
        else:
            d = arg.get_shape()[-1]
        trans = linear([arg], d, bias, bias_start=bias_start, scope='trans', wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)

        trans = tf.nn.relu(trans)
        gate = linear([arg], d, bias, bias_start=bias_start, scope='gate', wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)
        gate = tf.nn.sigmoid(gate)
        if d != arg.get_shape()[-1]:
            arg = linear([arg], d, bias, bias_start=bias_start, scope='arg_resize', wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)
        out = gate * trans + (1 - gate) * arg
        return out


def highway_network(arg, num_layers, bias, bias_start=0.0, scope=None, wd=0.0, input_keep_prob=1.0, is_train=None, output_size = None):
    with tf.variable_scope(scope or "highway_network"):
        prev = arg
        cur = None
        for layer_idx in range(num_layers):
            cur = highway_layer(prev, bias, bias_start=bias_start, scope="layer_{}".format(layer_idx), wd=wd,
                                input_keep_prob=input_keep_prob, is_train=is_train, output_size = output_size)
            prev = cur
        return cur
