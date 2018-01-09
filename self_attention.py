from tensorflow.contrib.rnn.python.ops.rnn_cell import _linear
from tensorflow.python.util import nest
import tensorflow as tf

#copy from https://github.com/YichenGong/Densely-Interactive-Inference-Network/blob/master/python/models/DIIN.py
'''
# call in DIIN.py
 with tf.variable_scope("prepro") as scope:
            pre = premise_in
            hyp = hypothesis_in
            for i in range(config.self_att_enc_layers):
                with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                    # pre:[N,L,2d] ,prem_mask:[N,L,1]
                    p = self_attention_layer(config, self.is_train, pre, p_mask=prem_mask, scope="{}_layer_self_att_enc".format(i)) # [N, len, dim]    
                    h = self_attention_layer(config, self.is_train, hyp, p_mask=hyp_mask, scope="{}_layer_self_att_enc_h".format(i))
                    pre = p
                    hyp = h
                    variable_summaries(p, "p_self_enc_summary_layer_{}".format(i))
                    variable_summaries(h, "h_self_enc_summary_layer_{}".format(i))
'''
def self_attention_layer(config, is_train, p, p_mask=None, scope=None):
    with tf.variable_scope(scope or "self_attention_layer"):
        PL = tf.shape(p)[1]
        # HL = tf.shape(h)[1]
        # if config.q2c_att or config.c2q_att:
        self_att = self_attention(config, is_train, p, p_mask=p_mask)

        print("self_att shape")
        print(self_att.get_shape())
        
        p0 = fuse_gate(config, is_train, p, self_att, scope="self_att_fuse_gate")
        
        return p0

def self_attention(config, is_train, p, p_mask=None, scope=None): #[N, L, 2d]
    with tf.variable_scope(scope or "self_attention"):
        PL = p.get_shape().as_list()[1]
        dim = p.get_shape().as_list()[-1]
        # HL = tf.shape(h)[1]
        p_aug_1 = tf.tile(tf.expand_dims(p, 2), [1,1,PL,1])
        p_aug_2 = tf.tile(tf.expand_dims(p, 1), [1,PL,1,1]) #[N, PL, HL, 2d]

        if p_mask is None:
            ph_mask = None
        else:
            p_mask_aug_1 = tf.reduce_any(tf.cast(tf.tile(tf.expand_dims(p_mask, 2), [1, 1, PL, 1]), tf.bool), axis=3)
            p_mask_aug_2 = tf.reduce_any(tf.cast(tf.tile(tf.expand_dims(p_mask, 1), [1, PL, 1, 1]), tf.bool), axis=3)
            self_mask = p_mask_aug_1 & p_mask_aug_2


        h_logits = get_logits([p_aug_1, p_aug_2], None, True, wd=config.wd, mask=self_mask,
                              is_train=is_train, func=config.self_att_logit_func, scope='h_logits')  # [N, PL, HL]
        self_att = softsel(p_aug_2, h_logits) 

        return self_att

def get_logits(args, size, bias, bias_start=0.0, scope=None, mask=None, wd=0.0, input_keep_prob=1.0, is_train=None, func=None):
    if func is None:
        func = "sum"
    if func == 'sum':
        return sum_logits(args, mask=mask, name=scope)
    elif func == 'linear':
        return linear_logits(args, bias, bias_start=bias_start, scope=scope, mask=mask, wd=wd, input_keep_prob=input_keep_prob,
                             is_train=is_train)
    elif func == 'double':
        return double_linear_logits(args, size, bias, bias_start=bias_start, scope=scope, mask=mask, wd=wd, input_keep_prob=input_keep_prob,
                                    is_train=is_train)
    elif func == 'dot':
        assert len(args) == 2
        arg = args[0] * args[1]
        return sum_logits([arg], mask=mask, name=scope)
    elif func == 'scaled_dot':
        assert len(args) == 2
        dim = args[0].get_shape().as_list()[-1]
        arg = args[0] * args[1]
        arg = arg / tf.sqrt(tf.constant(dim, dtype=tf.float32))
        return sum_logits([arg], mask=mask, name=scope)
    elif func == 'mul_linear':
        assert len(args) == 2
        arg = args[0] * args[1]
        return linear_logits([arg], bias, bias_start=bias_start, scope=scope, mask=mask, wd=wd, input_keep_prob=input_keep_prob,
                             is_train=is_train)
    elif func == 'proj':
        assert len(args) == 2
        d = args[1].get_shape()[-1]
        proj = linear([args[0]], d, False, bias_start=bias_start, scope=scope, wd=wd, input_keep_prob=input_keep_prob,
                      is_train=is_train)
        return sum_logits([proj * args[1]], mask=mask)
    elif func == 'tri_linear':
        assert len(args) == 2
        new_arg = args[0] * args[1]
        return linear_logits([args[0], args[1], new_arg], bias, bias_start=bias_start, scope=scope, mask=mask, wd=wd, input_keep_prob=input_keep_prob,
                             is_train=is_train)
    else:
        raise Exception()
    
def linear_logits(args, bias, bias_start=0.0, scope=None, mask=None, wd=0.0, input_keep_prob=1.0, is_train=None):
    with tf.variable_scope(scope or "Linear_Logits"):
        logits = linear(args, 1, bias, bias_start=bias_start, squeeze=True, scope='first',
                        wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)
        if mask is not None:
            logits = exp_mask(logits, mask)
        return logits

def linear(args, output_size, bias, bias_start=0.0, scope=None, squeeze=False, wd=0.0, input_keep_prob=1.0,
           is_train=None):
    with tf.variable_scope(scope or "linear"):
        if args is None or (nest.is_sequence(args) and not args):
            raise ValueError("`args` must be specified")
        if not nest.is_sequence(args):
            args = [args]

        flat_args = [flatten(arg, 1) for arg in args]
        # if input_keep_prob < 1.0:
        assert is_train is not None
        flat_args = [tf.cond(is_train, lambda: tf.nn.dropout(arg, input_keep_prob), lambda: arg)
                         for arg in flat_args]
        flat_out = _linear(flat_args, output_size, bias)
        out = reconstruct(flat_out, args[0], 1)
        if squeeze:
            out = tf.squeeze(out, [len(args[0].get_shape().as_list())-1])
        if wd:
            add_wd(wd)

    return out
  
  def fuse_gate(config, is_train, lhs, rhs, scope=None):
    with tf.variable_scope(scope or "fuse_gate"):
        dim = lhs.get_shape().as_list()[-1]
        # z
        # if config.fuse_gate_KR_1_0:
        #     lhs_1 = linear(lhs, dim ,True, bias_start=0.0, scope="lhs_1", squeeze=False, wd=config.wd, input_keep_prob=1.0, is_train=is_train)
        #     rhs_1 = linear(rhs, dim ,True, bias_start=0.0, scope="rhs_1", squeeze=False, wd=0.0, input_keep_prob=1.0, is_train=is_train)
        # else:
        lhs_1 = linear(lhs, dim ,True, bias_start=0.0, scope="lhs_1", squeeze=False, wd=config.wd, input_keep_prob=config.keep_rate, is_train=is_train)
        rhs_1 = linear(rhs, dim ,True, bias_start=0.0, scope="rhs_1", squeeze=False, wd=0.0, input_keep_prob=config.keep_rate, is_train=is_train)
        if config.self_att_fuse_gate_residual_conn and config.self_att_fuse_gate_relu_z:
            z = tf.nn.relu(lhs_1 + rhs_1)
        else:
            z = tf.tanh(lhs_1 + rhs_1)
        # f
        # if config.fuse_gate_KR_1_0:
        #     lhs_2 = linear(lhs, dim ,True, bias_start=0.0, scope="lhs_2", squeeze=False, wd=config.wd, input_keep_prob=1.0, is_train=is_train)
        #     rhs_2 = linear(rhs, dim ,True, bias_start=0.0, scope="rhs_2", squeeze=False, wd=config.wd, input_keep_prob=1.0, is_train=is_train)
        # else:
        lhs_2 = linear(lhs, dim ,True, bias_start=0.0, scope="lhs_2", squeeze=False, wd=config.wd, input_keep_prob=config.keep_rate, is_train=is_train)
        rhs_2 = linear(rhs, dim ,True, bias_start=0.0, scope="rhs_2", squeeze=False, wd=config.wd, input_keep_prob=config.keep_rate, is_train=is_train)
        f = tf.sigmoid(lhs_2 + rhs_2)

        if config.two_gate_fuse_gate:
            lhs_3 = linear(lhs, dim ,True, bias_start=0.0, scope="lhs_3", squeeze=False, wd=config.wd, input_keep_prob=config.keep_rate, is_train=is_train)
            rhs_3 = linear(rhs, dim ,True, bias_start=0.0, scope="rhs_3", squeeze=False, wd=config.wd, input_keep_prob=config.keep_rate, is_train=is_train)
            f2 = tf.sigmoid(lhs_3 + rhs_3)
            out = f * lhs + f2 * z
        else:   
            out = f * lhs + (1 - f) * z

        return out

