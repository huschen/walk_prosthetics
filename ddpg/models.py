import tensorflow as tf
import tensorflow.contrib as tc


class Model(object):
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        ''' Used for target actor/critic updates '''
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        ''' Used for MPI gradient calculation '''
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        ''' Used for parameter noise '''
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]

    @property
    def reg_vars(self):
        ''' Used in parameter regularisation '''
        # LayerNorm variables (/beta:0, /gamma:0) have no 'kernel'
        return [var for var in self.trainable_vars if 'kernel' in var.name]


def double_sigmoid(x, y_default):
    ''' piecewise sigmoid, maximum(sigmoid_a, sigmoid_b)
        example:
        y_default = 0.05
        igmoid_a: sigmoid(x)*0.1, sigmoid_a(0) = 0.05
        sigmoid_b: sigmoid(x)*1.9-0.9, sigmoid_b(0) = 0.05
        maximum(sigmoid(x)*0.1,  sigmoid(x)*1.9-0.9) for y(0) = 0.05'''
    scale = y_default * 2
    return tf.maximum(tf.nn.sigmoid(x) * scale, tf.nn.sigmoid(x) * (2 - scale) - 1 + scale)


class Actor(Model):
    def __init__(self, nb_actions, nb_demo_kine, name='actor', layer_norm=True, sigm_default=0.05):
        super(Actor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.nb_demo_kine = nb_demo_kine
        self.sigm_default = sigm_default

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs[:, 0:self.nb_demo_kine]
            # 'demo' needed in the name
            x = tf.layers.dense(x, 16, name='demo_mid')
            x = tf.nn.leaky_relu(x)
            # 'demo' needed in the name
            demo = x = tf.layers.dense(x, self.nb_demo_kine, name='demo_aprx',
                                       kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))

            x = tf.concat([x, obs], axis=-1)
            x = tf.layers.dense(x, 96)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.leaky_relu(x)

            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.leaky_relu(x)

            x = tf.layers.dense(x, 32)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.leaky_relu(x)

            x = tf.layers.dense(x, self.nb_actions,
                                kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            # x = tf.nn.tanh(x)
            x = double_sigmoid(x, self.sigm_default)
        return x, demo

    @property
    def active_vars(self):
        ''' demo vars become inactive during RL '''
        return [var for var in self.trainable_vars if 'demo' not in var.name]

    @property
    def perturbable_vars(self):
        ''' for applying parameter noise, demo vars excluded '''
        return [var for var in super().perturbable_vars if 'demo' not in var.name]

    @property
    def demo_reg_vars(self):
        return [var for var in super().reg_vars if 'demo_mid' in var.name]

    @property
    def dbg_vars(self):
        ''' only used in dbg_tf_init(), for sanity check of tensorflow initialisation '''
        return self.demo_reg_vars[:1] + self.perturbable_vars[:1]


class Critic(Model):
    def __init__(self, nb_preds, name='critic', layer_norm=True):
        super(Critic, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.nb_preds = nb_preds

    def __call__(self, obs, action, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs
            x = tf.layers.dense(x, 128)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.leaky_relu(x)

            x = tf.concat([x, action], axis=-1)
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.leaky_relu(x)

            x = tf.layers.dense(x, 32)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.leaky_relu(x)

            q = tf.layers.dense(x, 1, name='output',
                                kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            # though target network does not need preds..
            pred_obs = tf.layers.dense(x, self.nb_preds,
                                       kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            pred_rwd = tf.layers.dense(x, 1,
                                       kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return q, pred_rwd, pred_obs

    @property
    def output_vars(self):
        ''' used in ddpg.setup_popart() '''
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars

    @property
    def reg_vars(self):
        return [var for var in super().reg_vars if 'output' not in var.name]

    @property
    def dbg_vars(self):
        ''' only used in dbg_tf_init(), for sanity check of tensorflow initialisation '''
        return self.perturbable_vars[:1]
