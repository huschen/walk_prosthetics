import os
from copy import copy

import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
from mpi4py import MPI

from common import logger
from common.misc_util import boolean_flag
from common.mpi_adam import MpiAdam
from common.mpi_running_mean_std import RunningMeanStd
import common.tf_util as tf_util

from ddpg.models import Actor, Critic
from ddpg.memory import Memory
import ddpg.noise as Noise


def obs_norm_partial(x, stats, nb_org):
    ''' partially normalize observation data '''
    if stats is None:
        return x
    y1 = x[:, :nb_org]
    y2 = (x[:, nb_org:] - stats.mean[nb_org:]) / stats.std[nb_org:]
    y = tf.concat([y1, y2], axis=-1)
    return y


def ret_normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / stats.std


def ret_denormalize(x, stats):
    if stats is None:
        return x
    return x * stats.std + stats.mean


def reduce_std(x, axis=None, keepdims=False):
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))


def reduce_var(x, axis=None, keepdims=False):
    m = tf.reduce_mean(x, axis=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)


def act_norm(x):
    ''' normalise action, from [0, 1] to [-1, 1], as input to critic network '''
    return (x - 0.5) * 2


def act_denorm(x):
    # Action output is in [0,1] range, function not used!
    return x


def get_target_updates(vvars, target_vars, tau):
    logger.debug('setting up target updates ...')
    soft_updates = []
    init_updates = []
    assert len(vvars) == len(target_vars)
    for var, target_var in zip(vvars, target_vars):
        logger.debug('  {} <- {}'.format(target_var.name, var.name))
        init_updates.append(tf.assign(target_var, var))
        soft_updates.append(tf.assign(target_var, (1. - tau) * target_var + tau * var))
    assert len(init_updates) == len(vvars)
    assert len(soft_updates) == len(vvars)
    return tf.group(*init_updates), tf.group(*soft_updates)


def get_perturbed_actor_updates(actor, perturbed_actor, param_noise_stddev):
    assert len(actor.vars) == len(perturbed_actor.vars)
    assert len(actor.perturbable_vars) == len(perturbed_actor.perturbable_vars)

    updates = []
    for var, perturbed_var in zip(actor.vars, perturbed_actor.vars):
        if var in actor.perturbable_vars:
            logger.debug('  {} <- {} + noise'.format(perturbed_var.name, var.name))
            updates.append(tf.assign(perturbed_var,
                                     var + tf.random_normal(tf.shape(var), mean=0., stddev=param_noise_stddev)))
        else:
            logger.debug('  {} <- {}'.format(perturbed_var.name, var.name))
            updates.append(tf.assign(perturbed_var, var))
    assert len(updates) == len(actor.vars)
    return tf.group(*updates)


def process_noise_type(noise_type, nb_actions):
    param_noise = None
    action_noise = None
    for current_noise_type in noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'adaptive-param' in current_noise_type:
            _, stddev_init, stddev = current_noise_type.split('_')
            param_noise = Noise.AdaptiveParamNoiseSpec(initial_stddev=float(stddev_init),
                                                       desired_action_stddev=float(stddev))
        elif 'normal' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = Noise.NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        elif 'ou' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = Noise.OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions),
                                                              sigma=float(stddev) * np.ones(nb_actions))
        else:
            raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))
    return param_noise, action_noise


class DDPG(object):
    def __init__(self, observation_shape, action_shape, nb_demo_kine, nb_key_states,
                 batch_size=128, noise_type='', actor=None, critic=None,
                 layer_norm=True, observation_range=(-5., 5.), action_range=(-1., 1.), return_range=(-np.inf, np.inf),
                 normalize_returns=False, normalize_observations=True, reward_scale=1., clip_norm=None,
                 demo_l2_reg=0., critic_l2_reg=0., actor_lr=1e-4, critic_lr=1e-3, demo_lr=5e-3, gamma=0.99, tau=0.001,
                 enable_popart=False, save_ckpt=True):

        # Noise
        nb_actions = action_shape[-1]
        param_noise, action_noise = process_noise_type(noise_type, nb_actions)

        logger.info('param_noise', param_noise)
        logger.info('action_noise', action_noise)

        # States recording
        self.memory = Memory(limit=int(2e5), action_shape=action_shape, observation_shape=observation_shape)

        # Models
        self.nb_demo_kine = nb_demo_kine
        self.actor = actor or Actor(nb_actions, nb_demo_kine, layer_norm=layer_norm)
        self.nb_key_states = nb_key_states
        self.critic = critic or Critic(nb_key_states, layer_norm=layer_norm)
        self.nb_obs_org = nb_key_states

        # Inputs.
        self.obs0 = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs0')
        self.obs1 = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs1')
        self.terminals1 = tf.placeholder(tf.float32, shape=(None, 1), name='terminals1')
        self.rewards = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
        self.actions = tf.placeholder(tf.float32, shape=(None,) + action_shape, name='actions')
        # self.critic_target_Q: value assigned by self.target_Q_obs0
        self.critic_target_Q = tf.placeholder(tf.float32, shape=(None, 1), name='critic_target_Q')
        self.param_noise_stddev = tf.placeholder(tf.float32, shape=(), name='param_noise_stddev')

        # change in observations
        self.obs_delta_kine = (self.obs1 - self.obs0)[:, :self.nb_demo_kine]
        self.obs_delta_kstates = (self.obs1 - self.obs0)[:, :self.nb_key_states]

        # Parameters.
        self.gamma = gamma
        self.tau = tau
        self.normalize_observations = normalize_observations
        self.normalize_returns = normalize_returns
        self.action_noise = action_noise
        self.param_noise = param_noise
        self.action_range = action_range
        self.return_range = return_range
        self.observation_range = observation_range

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.demo_lr = demo_lr
        self.clip_norm = clip_norm
        self.enable_popart = enable_popart
        self.reward_scale = reward_scale
        self.batch_size = batch_size
        self.stats_sample = None
        self.critic_l2_reg = critic_l2_reg
        self.demo_l2_reg = demo_l2_reg

        # Observation normalization.
        if self.normalize_observations:
            with tf.variable_scope('obs_rms'):
                self.obs_rms = RunningMeanStd(shape=observation_shape)
        else:
            self.obs_rms = None

        self.normalized_obs0 = tf.clip_by_value(obs_norm_partial(self.obs0, self.obs_rms, self.nb_obs_org),
                                                self.observation_range[0], self.observation_range[1])
        normalized_obs1 = tf.clip_by_value(obs_norm_partial(self.obs1, self.obs_rms, self.nb_obs_org),
                                           self.observation_range[0], self.observation_range[1])

        # Return normalization.
        if self.normalize_returns:
            with tf.variable_scope('ret_rms'):
                self.ret_rms = RunningMeanStd()
        else:
            self.ret_rms = None

        # Create target networks.
        target_actor = copy(self.actor)
        target_actor.name = 'target_actor'
        self.target_actor = target_actor
        target_critic = copy(self.critic)
        target_critic.name = 'target_critic'
        self.target_critic = target_critic

        # Create networks and core TF parts that are shared across set-up parts.
        # the actor output is [0,1], need to normalised to [-1,1] before feeding into critic
        self.actor_tf, self.demo_aprx = self.actor(self.normalized_obs0)

        # critic loss
        # normalized_critic_tf, pred_rwd, pred_obs_delta: critic_loss
        self.normalized_critic_tf, self.pred_rwd, self.pred_obs_delta = self.critic(self.normalized_obs0,
                                                                                    act_norm(self.actions))
        # self.critic_tf: only in logging [reference_Q_mean/std]
        self.critic_tf = ret_denormalize(tf.clip_by_value(self.normalized_critic_tf,
                                                          self.return_range[0], self.return_range[1]), self.ret_rms)

        # actor loss
        normalized_critic_with_actor_tf = self.critic(self.normalized_obs0, act_norm(self.actor_tf), reuse=True)[0]
        # self.critic_with_actor_tf: actor loss, and logging [reference_Q_tf_mean/std]
        self.critic_with_actor_tf = ret_denormalize(
            tf.clip_by_value(normalized_critic_with_actor_tf, self.return_range[0], self.return_range[1]),
            self.ret_rms)

        # target Q
        self.target_action = tf.clip_by_value(target_actor(normalized_obs1)[0],
                                              self.action_range[0], self.action_range[1])
        self.target_Q_obs1 = ret_denormalize(target_critic(normalized_obs1, act_norm(self.target_action))[0],
                                             self.ret_rms)
        self.target_Q_obs0 = self.rewards + (1. - self.terminals1) * gamma * self.target_Q_obs1

        # Set up parts.
        if self.param_noise is not None:
            self.setup_param_noise(self.normalized_obs0)

        self.setup_actor_optimizer()
        self.setup_critic_optimizer()
        if self.normalize_returns and self.enable_popart:
            self.setup_popart()
        self.setup_stats()
        self.setup_target_network_updates()
        self.dbg_vars = self.actor.dbg_vars + self.critic.dbg_vars

        self.sess = None
        # Set up checkpoint saver
        self.save_ckpt = save_ckpt
        if save_ckpt:
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
        else:
            # saver for loading ckpt
            self.saver = tf.train.Saver()

        self.main_summaries = tf.summary.merge_all()
        logdir = logger.get_dir()
        if logdir:
            self.train_writer = tf.summary.FileWriter(os.path.join(logdir, 'tb'), tf.get_default_graph())
        else:
            self.train_writer = None

    def setup_target_network_updates(self):
        actor_init_updates, actor_soft_updates = get_target_updates(self.actor.vars, self.target_actor.vars, self.tau)
        critic_init_updates, critic_soft_updates = get_target_updates(self.critic.vars, self.target_critic.vars,
                                                                      self.tau)
        self.target_init_updates = [actor_init_updates, critic_init_updates]
        self.target_soft_updates = [actor_soft_updates, critic_soft_updates]

    def setup_param_noise(self, normalized_obs0):
        assert self.param_noise is not None

        # Configure perturbed actor.
        param_noise_actor = copy(self.actor)
        param_noise_actor.name = 'param_noise_actor'
        self.perturbed_actor_tf = param_noise_actor(normalized_obs0)[0]
        logger.debug('setting up param noise')
        self.perturb_policy_ops = get_perturbed_actor_updates(self.actor, param_noise_actor, self.param_noise_stddev)

        # Configure separate copy for stddev adoption.
        adaptive_param_noise_actor = copy(self.actor)
        adaptive_param_noise_actor.name = 'adaptive_param_noise_actor'
        adaptive_actor_tf = adaptive_param_noise_actor(normalized_obs0)[0]
        self.perturb_adaptive_policy_ops = get_perturbed_actor_updates(self.actor, adaptive_param_noise_actor,
                                                                       self.param_noise_stddev)
        self.adaptive_policy_distance = tf.sqrt(tf.reduce_mean(tf.square(self.actor_tf - adaptive_actor_tf)))

    def setup_actor_optimizer(self):
        logger.info('setting up actor optimizer')
        # loss_normed = -tf.reduce_mean(self.normalized_critic_with_actor_tf)
        self.actor_Q = tf.reduce_mean(self.critic_with_actor_tf)
        self.actor_loss = -self.actor_Q
        tf.summary.scalar('actor/Q', self.actor_Q)

        # setting up actor vars/grads/optimizer
        self.actor_vars = self.actor.active_vars
        self.actor_grads = tf_util.flatgrad(self.actor_loss, self.actor_vars, clip_norm=self.clip_norm)
        self.actor_optimizer = MpiAdam(var_list=self.actor_vars, beta1=0.9, beta2=0.999, epsilon=1e-08)

        actor_shapes = [var.get_shape().as_list() for var in self.actor.trainable_vars]
        self.actor_params = actor_params = [0] * (len(self.actor.trainable_vars) + 1)
        for i, shape in enumerate(actor_shapes):
            actor_params[i + 1] = actor_params[i] + np.prod(shape)
        n_inact = len(actor_shapes) - len(self.actor_vars)
        active_params = actor_params[n_inact:] - actor_params[n_inact]
        logger.info('  actor shapes: {}'.format(actor_shapes))
        logger.info('  actor params: {}'.format(actor_params))
        logger.info('  actor total: {}'.format(actor_params[-1]))
        logger.info('  actor active: {}'.format(active_params))

        grad = self.actor_grads[active_params[0]:active_params[1]]
        tf.summary.scalar('grads/actor_layer%d_%d' % (n_inact // 2, active_params[1] - active_params[0]),
                          tf.reduce_mean(grad))

        grad = self.actor_grads[active_params[-3]:active_params[-2]]
        tf.summary.scalar('grads/actor_layer%d_%d' % (-1, active_params[-2] - active_params[-3]), tf.reduce_mean(grad))

        # for train_demo()
        self.demo_loss = tf.reduce_mean(tf.square(self.obs_delta_kine - self.demo_aprx))
        self.demo_max_loss = tf.reduce_max(tf.square(self.obs_delta_kine - self.demo_aprx))
        if self.demo_l2_reg > 0.:
            demo_reg_vars = self.actor.demo_reg_vars
            for var in demo_reg_vars:
                logger.info('  regularizing: {}'.format(var.name))
            logger.info('  applying l2 regularization for demo_aprx with {}'.format(self.demo_l2_reg))
            self.demo_reg = tc.layers.apply_regularization(
                tc.layers.l2_regularizer(self.demo_l2_reg),
                weights_list=demo_reg_vars
            )
            self.demo_loss += self.demo_reg
        else:
            self.demo_reg = None

        self.demo_grads = tf_util.flatgrad(self.demo_loss, self.actor.trainable_vars, clip_norm=self.clip_norm)
        self.demo_optimizer = MpiAdam(var_list=self.actor.trainable_vars, beta1=0.9, beta2=0.999, epsilon=1e-08)

        # mimic rwd
        self.mimic_rwd = -self.demo_loss
        tf.summary.scalar('actor/mimic_rwd', self.mimic_rwd)

    def setup_critic_optimizer(self):
        logger.info('setting up critic optimizer')

        self.normalized_critic_target_tf = tf.clip_by_value(ret_normalize(self.critic_target_Q, self.ret_rms),
                                                            self.return_range[0], self.return_range[1])
        self.critic_loss = tf.reduce_mean(tf.square(self.normalized_critic_tf - self.normalized_critic_target_tf))
        tf.summary.scalar('critic_loss/Q_diff', self.critic_loss)
        if self.normalize_returns:
            tf.summary.scalar('critic_loss/Q_normed_critic', tf.reduce_mean(self.normalized_critic_tf))
            tf.summary.scalar('critic_loss/Q_normed_target', tf.reduce_mean(self.normalized_critic_target_tf))

        self.critic_loss_step = 0
        diff_rwd = tf.reduce_mean(tf.square(self.pred_rwd - self.rewards))
        self.critic_loss_step += diff_rwd
        tf.summary.scalar('critic_loss/step_rwd', self.critic_loss_step)

        critic_kine_factor = 100
        diff_obs = tf.reduce_mean(tf.square(self.pred_obs_delta - self.obs_delta_kstates), axis=0)
        diff_obs_kine = tf.reduce_mean(diff_obs[:self.nb_demo_kine]) * critic_kine_factor
        diff_obs_rest = tf.reduce_mean(diff_obs[self.nb_demo_kine:])
        self.critic_loss_step += (diff_obs_kine + diff_obs_rest)
        tf.summary.scalar('critic_loss/step_kstates_kine_x%d' % critic_kine_factor, diff_obs_kine)
        tf.summary.scalar('critic_loss/step_kstates_rest', diff_obs_rest)
        tf.summary.scalar('critic_loss/step_total', self.critic_loss_step)

        self.critic_loss += self.critic_loss_step

        if self.critic_l2_reg > 0.:
            critic_reg_vars = self.critic.reg_vars
            for var in critic_reg_vars:
                logger.debug('  regularizing: {}'.format(var.name))
            logger.info('  applying l2 regularization with {}'.format(self.critic_l2_reg))
            critic_reg = tc.layers.apply_regularization(
                tc.layers.l2_regularizer(self.critic_l2_reg),
                weights_list=critic_reg_vars
            )
            self.critic_loss += critic_reg
            tf.summary.scalar('critic_loss/reg', critic_reg)

        critic_shapes = [var.get_shape().as_list() for var in self.critic.trainable_vars]

        critic_params = [0] * (len(self.critic.trainable_vars) + 1)
        for i, shape in enumerate(critic_shapes):
            critic_params[i + 1] = critic_params[i] + np.prod(shape)

        logger.info('  critic shapes: {}'.format(critic_shapes))
        logger.info('  critic params: {}'.format(critic_params))
        logger.info('  critic total: {}'.format(critic_params[-1]))
        self.critic_grads = tf_util.flatgrad(self.critic_loss, self.critic.trainable_vars, clip_norm=self.clip_norm)
        self.critic_optimizer = MpiAdam(var_list=self.critic.trainable_vars, beta1=0.9, beta2=0.999, epsilon=1e-08)

        # todo: make the following general
        grad = self.critic_grads[critic_params[0]:critic_params[1]]
        tf.summary.scalar('grads/critic_layer%d_%d' % (0, critic_params[1] - critic_params[0]), tf.reduce_mean(grad))
        grad = self.critic_grads[critic_params[-3]:critic_params[-2]]
        tf.summary.scalar('grads/critic_layer%d_rwd_%d' % (-1, critic_params[-2] - critic_params[-3]),
                          tf.reduce_mean(grad))
        grad = self.critic_grads[critic_params[-7]:critic_params[-6]]
        tf.summary.scalar('grads/critic_layer%d_q_%d' % (-1, critic_params[-6] - critic_params[-7]),
                          tf.reduce_mean(grad))

    def setup_popart(self):
        # See https://arxiv.org/pdf/1602.07714.pdf for details.
        self.old_std = tf.placeholder(tf.float32, shape=[1], name='old_std')
        new_std = self.ret_rms.std
        self.old_mean = tf.placeholder(tf.float32, shape=[1], name='old_mean')
        new_mean = self.ret_rms.mean

        self.renormalize_Q_outputs_op = []
        for vs in [self.critic.output_vars, self.target_critic.output_vars]:
            assert len(vs) == 2
            M, b = vs
            assert 'kernel' in M.name
            assert 'bias' in b.name
            assert M.get_shape()[-1] == 1
            assert b.get_shape()[-1] == 1
            self.renormalize_Q_outputs_op += [M.assign(M * self.old_std / new_std)]
            self.renormalize_Q_outputs_op += [b.assign((b * self.old_std + self.old_mean - new_mean) / new_std)]

    def setup_stats(self):
        ops = []
        names = []

        if self.normalize_returns:
            ops += [self.ret_rms.mean, self.ret_rms.std]
            names += ['zrms/ret_mean', 'zrms/ret_std']

        if self.normalize_observations:
            ops += [tf.reduce_mean(self.obs_rms.mean[:self.nb_demo_kine]),
                    tf.reduce_mean(self.obs_rms.std[:self.nb_demo_kine])]
            names += ['zrms/obs_kine_mean', 'zrms/obs_kine_std']

            ops += [tf.reduce_mean(self.obs_rms.mean[:self.nb_key_states]),
                    tf.reduce_mean(self.obs_rms.std[:self.nb_key_states])]
            names += ['zrms/obs_kstates_mean', 'zrms/obs_kstates_std']

            ops += [tf.reduce_mean(self.obs_rms.mean), tf.reduce_mean(self.obs_rms.std)]
            names += ['zrms/obs_mean', 'zrms/obs_std']

            # for debugging partial normalisation
            for o_i in [self.nb_obs_org - 1, self.nb_obs_org]:
                ops += [self.obs0[0, o_i], self.normalized_obs0[0, o_i]]
                names += ['zobs_dbg_%d' % o_i, 'zobs_dbg_%d_normalized' % o_i]

        ops += [tf.reduce_mean(self.critic_tf)]
        names += ['zref/Q_mean']
        ops += [reduce_std(self.critic_tf)]
        names += ['zref/Q_std']

        ops += [tf.reduce_mean(self.critic_with_actor_tf)]
        names += ['zref/Q_tf_mean']
        ops += [reduce_std(self.critic_with_actor_tf)]
        names += ['zref/Q_tf_std']

        ops += [tf.reduce_mean(self.actor_tf)]
        names += ['zref/action_mean']
        ops += [reduce_std(self.actor_tf)]
        names += ['zref/action_std']

        ops += [tf.reduce_mean(self.mimic_rwd)]
        names += ['zref/mimic_rwd']

        if self.param_noise:
            ops += [tf.reduce_mean(self.perturbed_actor_tf)]
            names += ['zref/action_ptb_mean']
            ops += [reduce_std(self.perturbed_actor_tf)]
            names += ['zref/action_ptb_std']

        self.stats_ops = ops
        self.stats_names = names

    def pi(self, obs, step, apply_param_noise=True, apply_action_noise=True, compute_Q=True, rollout_log=False):
        if self.param_noise is not None and apply_param_noise:
            actor_tf = self.perturbed_actor_tf
            info = 'ptb'
        else:
            actor_tf = self.actor_tf
            info = 'org'
        feed_dict = {self.obs0: [obs]}
        if compute_Q:
            action, q = self.sess.run([actor_tf, self.critic_with_actor_tf], feed_dict=feed_dict)
        else:
            action = self.sess.run(actor_tf, feed_dict=feed_dict)
            q = None
        action = action.flatten()
        # actor output is [0,1], no need to denormalise.
        # action = act_denorm(action)
        if rollout_log:
            summary_list = [('the_action/%d_rollout_%s' % (i, info), a) for i, a in enumerate(action)]

        if self.action_noise is not None and apply_action_noise:
            noise = self.action_noise()
            assert noise.shape == action.shape
            action += noise
        else:
            noise = None
        action = np.clip(action, self.action_range[0], self.action_range[1])

        if rollout_log:
            if noise is not None:
                summary_list += [('the_action/%d_rollout_noise' % i, a) for i, a in enumerate(noise)]
            self.add_list_summary(summary_list, step)
        return action, q

    def store_transition(self, storage, obs0, action, reward, obs1, terminal1):
        '''store one experience'''
        reward *= self.reward_scale
        storage.append(obs0, action, reward, obs1, terminal1)
        if self.normalize_observations:
            self.obs_rms.update(np.array([obs0]))

    def store_multrans(self, storage, obs0, action, reward, obs1, terminal1):
        '''store multiple experiences'''
        for i in range(len(reward)):
            storage.append(obs0[i], action[i], reward[i] * self.reward_scale, obs1[i], terminal1[i])
        if self.normalize_observations:
            self.obs_rms.update(np.vstack(obs0))

    def train_demo(self, obs0_pos, obs1_pos, obs0_neg, obs1_neg, step, neg_pct=1.0, lr_decay=1.0):
        # gradients calculated for pos and neg data separately, then combined for gradient update,
        # because only positive data are used in eval modes

        # the loss evaluated here are those before gradient update
        ops = [self.demo_grads, self.demo_loss, self.demo_max_loss, self.actor_Q]
        pos_grads, demo_loss, max_loss, actor_Q = self.sess.run(ops, feed_dict={
            self.obs0: obs0_pos,
            self.obs1: obs1_pos,
        })
        ops = [self.demo_grads, self.demo_loss]
        neg_grads, neg_loss = self.sess.run(ops, feed_dict={
            self.obs0: obs0_neg,
            self.obs1: obs1_neg,
        })

        comb_grads = pos_grads - neg_grads * neg_pct
        self.demo_optimizer.update(comb_grads, stepsize=self.demo_lr * lr_decay)

        if self.demo_reg is not None:
            demo_reg = self.sess.run(self.demo_reg)
        else:
            demo_reg = 0

        # sanity check the training
        pos_g = pos_grads[self.actor_params[2]:self.actor_params[3]]
        neg_g = neg_grads[self.actor_params[2]:self.actor_params[3]]
        comb_g = comb_grads[self.actor_params[2]:self.actor_params[3]]
        summary_list = [('demo_loss/train_pos', demo_loss),
                        ('demo_loss/train_max', max_loss),
                        ('demo_loss/train_neg', neg_loss),
                        ('grads/demo_pos_layer%d_%d' % (1, len(pos_g)), np.mean(pos_g)),
                        ('grads/demo_neg_layer%d_%d' % (1, len(neg_g)), np.mean(neg_g)),
                        ('grads/demo_comb_layer%d_%d' % (1, len(comb_g)), np.mean(comb_g)),
                        ('actor/Q', actor_Q),
                        ('demo_loss/reg', demo_reg)]
        self.add_list_summary(summary_list, step)

        return demo_loss

    def test_demo(self, obs0, obs1):
        loss_mean, loss_max = self.sess.run([self.demo_loss, self.demo_max_loss], feed_dict={
            self.obs0: obs0,
            self.obs1: obs1,
        })
        return loss_mean, loss_max

    def eval_demo(self, obs0):
        return self.sess.run(self.demo_aprx, feed_dict={self.obs0: obs0})

    def get_mimic_rwd(self, obs0, obs1):
        mimic_rwd, demo_aprx = self.sess.run([self.mimic_rwd, self.demo_aprx], feed_dict={
            self.obs0: obs0,
            self.obs1: obs1
        })
        return mimic_rwd, demo_aprx

    def train_main(self, step):
        batch = self.memory.sample(batch_size=self.batch_size)

        if self.normalize_returns and self.enable_popart:
            ops = [self.ret_rms.mean, self.ret_rms.std, self.target_Q_obs0, self.target_Q_obs1]
            old_mean, old_std, target_Q_obs0, target_Q_obs1 = self.sess.run(ops, feed_dict={
                self.obs1: batch['obs1'],
                self.rewards: batch['rewards'],
                self.terminals1: batch['terminals1'].astype('float32'),
            })
            self.ret_rms.update(target_Q_obs0.flatten())
            self.sess.run(self.renormalize_Q_outputs_op, feed_dict={
                self.old_std: np.array([old_std]),
                self.old_mean: np.array([old_mean]),
            })

            # Run sanity check. Disabled by default since it slows down things considerably.
            # print('running sanity check')
            # target_Q_new, new_mean, new_std = self.sess.run([self.target_Q_obs0, self.ret_rms.mean, self.ret_rms.std],
            # feed_dict={
            #     self.obs1: batch['obs1'],
            #     self.rewards: batch['rewards'],
            #     self.terminals1: batch['terminals1'].astype('float32'),
            # })
            # print(target_Q_new, target_Q_obs0, new_mean, new_std)
            # assert (np.abs(target_Q_obs0 - target_Q_new) < 1e-3).all()
        else:
            ops = [self.target_Q_obs0, self.target_Q_obs1]
            target_Q_obs0, target_Q_obs1 = self.sess.run(ops, feed_dict={
                self.obs1: batch['obs1'],
                self.rewards: batch['rewards'],
                self.terminals1: batch['terminals1'].astype('float32')
            })

        summary_list = [('critic_loss/Q_target_obs1_mean', np.mean(target_Q_obs1)),
                        ('critic_loss/Q_target_obs1_std', np.std(target_Q_obs1)),
                        ('critic_loss/Q_target_obs0_mean', np.mean(target_Q_obs0)),
                        ('critic_loss/Q_target_obs0_std', np.std(target_Q_obs0))]
        self.add_list_summary(summary_list, step)

        # Get all gradients and perform a synced update.
        ops = [self.main_summaries, self.actor_grads, self.actor_loss, self.critic_grads, self.critic_loss]
        main_summaries, actor_grads, actor_loss, critic_grads, critic_loss = self.sess.run(ops, feed_dict={
            self.obs0: batch['obs0'],
            self.actions: batch['actions'],
            self.critic_target_Q: target_Q_obs0,
            self.rewards: batch['rewards'],
            self.obs1: batch['obs1']
        })
        self.actor_optimizer.update(actor_grads, stepsize=self.actor_lr)
        self.critic_optimizer.update(critic_grads, stepsize=self.critic_lr)

        if self.train_writer:
            self.train_writer.add_summary(main_summaries, step)

        return critic_loss, actor_loss

    def initialize(self, sess, start_ckpt=None):
        self.sess = sess
        if start_ckpt:
            self.saver.restore(sess, start_ckpt)
        else:
            self.sess.run(tf.global_variables_initializer())
        self.actor_optimizer.sync()
        self.demo_optimizer.sync()
        self.critic_optimizer.sync()
        self.sess.run(self.target_init_updates)

    def store_ckpt(self, save_path, epoch):
        if self.save_ckpt:
            self.saver.save(self.sess, save_path, global_step=epoch)

    def update_target_net(self):
        self.sess.run(self.target_soft_updates)

    def get_stats(self, storage):
        if self.stats_sample is None:
            # Get a sample and keep that fixed for all further computations.
            # This allows us to estimate the change in value for the same set of inputs.
            self.stats_sample = storage.sample(batch_size=self.batch_size)
        values = self.sess.run(self.stats_ops, feed_dict={
            self.obs0: self.stats_sample['obs0'],
            self.obs1: self.stats_sample['obs1'],
            self.actions: self.stats_sample['actions'],
        })

        names = self.stats_names[:]
        assert len(names) == len(values)
        stats = dict(zip(names, values))

        if self.param_noise is not None:
            stats = {**stats, **self.param_noise.get_stats()}

        return stats

    def adapt_param_noise(self, step):
        if self.param_noise is None:
            return 0.

        # Perturb a separate copy of the policy to adjust the scale for the next "real" perturbation.
        batch = self.memory.sample(batch_size=self.batch_size)
        self.sess.run(self.perturb_adaptive_policy_ops, feed_dict={
            self.param_noise_stddev: self.param_noise.current_stddev,
        })
        distance = self.sess.run(self.adaptive_policy_distance, feed_dict={
            self.obs0: batch['obs0'],
            self.param_noise_stddev: self.param_noise.current_stddev,
        })
        mean_distance = MPI.COMM_WORLD.allreduce(distance, op=MPI.SUM) / MPI.COMM_WORLD.Get_size()
        self.param_noise.adapt(mean_distance)
        self.add_list_summary([('param_noise/distance', mean_distance)], step)
        self.add_list_summary([('param_noise/std', self.param_noise.current_stddev)], step)
        return mean_distance

    def reset(self):
        '''Reset internal state after an episode is complete.'''
        if self.action_noise is not None:
            self.action_noise.reset()
        if self.param_noise is not None:
            self.sess.run(self.perturb_policy_ops, feed_dict={
                self.param_noise_stddev: self.param_noise.current_stddev,
            })

    def add_list_summary(self, summary_raw, step):
        def summary_val(k, v):
            kwargs = {'tag': k, 'simple_value': v}
            return tf.Summary.Value(**kwargs)
        if self.train_writer:
            summary_list = [summary_val(tag, val) for tag, val in summary_raw]
            self.train_writer.add_summary(tf.Summary(value=summary_list), step)


def config(model_group):
    model_group.add_argument('--batch_size', type=int, default=64)
    model_group.add_argument('--noise_type', type=str, default='adaptive-param_0.02_0.05,ou_0.2')
    boolean_flag(model_group, 'layer_norm', default=True)
    boolean_flag(model_group, 'normalize_returns', default=False)
    boolean_flag(model_group, 'normalize_observations', default=True)
    model_group.add_argument('--reward_scale', type=float, default=1.)
    model_group.add_argument('--clip_norm', type=float, default=None)
    model_group.add_argument('--critic_l2_reg', type=float, default=1e-5)
    model_group.add_argument('--demo_l2_reg', type=float, default=2e-7)
    model_group.add_argument('--actor_lr', type=float, default=1e-4)
    model_group.add_argument('--critic_lr', type=float, default=1e-3)
    model_group.add_argument('--demo_lr', type=float, default=5e-3)
    boolean_flag(model_group, 'enable_popart', default=False)
    model_group.add_argument('--gamma', type=float, default=0.9)
