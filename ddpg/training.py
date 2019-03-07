import os
import time
import datetime

import numpy as np
from mpi4py import MPI

from common import logger
from common.misc_util import dbg_tf_init
import common.tf_util as tf_util
from ddpg.pretrain import pretrain_demo, load_history


def train(env, eval_env, agent, render=False, render_eval=False, sanity_run=False,
          nb_epochs=500, nb_epoch_cycles=20, nb_rollout_steps=100, nb_train_steps=50,
          param_noise_adaption_interval=50, hist_files=None, start_ckpt=None, demo_files=None):

    rank = MPI.COMM_WORLD.Get_rank()
    mpi_size = MPI.COMM_WORLD.Get_size()
    if rank == 0:
        logdir = logger.get_dir()
    else:
        logdir = None

    memory = agent.memory
    batch_size = agent.batch_size

    with tf_util.single_threaded_session() as sess:
        # Prepare everything.
        agent.initialize(sess, start_ckpt=start_ckpt)
        sess.graph.finalize()
        agent.reset()
        dbg_tf_init(sess, agent.dbg_vars)

        total_nb_train = 0
        total_nb_rollout = 0
        total_nb_eval = 0

        # pre-train demo and critic_step
        # train_params: (nb_steps, lr_scale)
        total_nb_train = pretrain_demo(agent, env, demo_files, total_nb_train,
                                       train_params=[(100, 1.0)], start_ckpt=start_ckpt)
        load_history(agent, env, hist_files)

        # main training
        obs = env.reset()
        reset = False
        episode_step = 0
        last_episode_step = 0

        for i_epoch in range(nb_epochs):
            t_epoch_start = time.time()
            logger.info('\n%s epoch %d starts:' % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), i_epoch))
            for i_cycle in range(nb_epoch_cycles):
                logger.info('\n%s cycles_%d of epoch_%d' % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                                                            i_cycle, i_epoch))

                # rollout
                rcd_obs, rcd_action, rcd_r, rcd_new_obs, rcd_done = [], [], [], [], []
                if not sanity_run and mpi_size == 1 and last_episode_step != 0:
                    # todo: use mpi_max(last_episode_step)
                    # dynamically set nb_rollout_steps
                    nb_rollout_steps = max(last_episode_step * 4, batch_size)
                logger.info('[%d, %d] rollout for %d steps.' % (total_nb_rollout, memory.nb_entries, nb_rollout_steps))
                t_rollout_start = time.time()

                for i_rollout in range(nb_rollout_steps):
                    rollout_log = i_cycle == 0
                    # 50% param_noise, 40% action_noise
                    action, q = agent.pi(obs, total_nb_rollout, compute_Q=True, rollout_log=rollout_log,
                                         apply_param_noise=i_rollout % 10 < 5, apply_action_noise=i_rollout % 10 > 5)
                    assert action.shape == env.action_space.shape
                    new_obs, r, done, reset, info = env.step(action)

                    if rank == 0 and render:
                        env.render()

                    episode_step += 1
                    total_nb_rollout += 1

                    if rollout_log:
                        summary_list = [('rollout/%s' % tp, info[tp]) for tp in ['rwd_walk', 'rwd_total']]
                        tp = 'rwd_agent'
                        summary_list += [('rollout/%s_x%d' % (tp, info['rf_agent']), info[tp] * info['rf_agent'])]
                        summary_list += [('rollout/q', q)]
                        if r != 0:
                            summary_list += [('rollout/q_div_r', q / r)]
                        agent.add_list_summary(summary_list, total_nb_rollout)

                    # store at the end of cycle to speed up MPI rollout
                    # agent.store_transition(obs, action, r, new_obs, done)
                    rcd_obs.append(obs)
                    rcd_action.append(action)
                    rcd_r.append(r)
                    rcd_new_obs.append(new_obs)
                    rcd_done.append(done)

                    obs = new_obs
                    if reset:
                        # Episode done.
                        last_episode_step = episode_step
                        episode_step = 0

                        agent.reset()
                        obs = env.reset()

                agent.store_multrans(memory, rcd_obs, rcd_action, rcd_r, rcd_new_obs, rcd_done)

                t_train_start = time.time()
                steps_per_second = float(nb_rollout_steps) / (t_train_start - t_rollout_start)
                agent.add_list_summary([('rollout/steps_per_second', steps_per_second)], total_nb_rollout)

                # Train.
                if not sanity_run:
                    # dynamically set nb_train_steps
                    if memory.nb_entries > batch_size * 20:
                        # using 1% of data for training every step?
                        nb_train_steps = max(int(memory.nb_entries * 0.01 / batch_size), 1)
                    else:
                        nb_train_steps = 0
                logger.info('[%d] training for %d steps.' % (total_nb_train, nb_train_steps))
                for _ in range(nb_train_steps):
                    # Adapt param noise, if necessary.
                    if memory.nb_entries >= batch_size and total_nb_train % param_noise_adaption_interval == 0:
                        agent.adapt_param_noise(total_nb_train)

                    agent.train_main(total_nb_train)
                    agent.update_target_net()
                    total_nb_train += 1

                if i_epoch == 0 and i_cycle < 5:
                    rollout_duration = t_train_start - t_rollout_start
                    train_duration = time.time() - t_train_start
                    logger.info('rollout_time(%d) = %.3fs, train_time(%d) = %.3fs' % (
                        nb_rollout_steps, rollout_duration, nb_train_steps, train_duration))
                    logger.info('rollout_speed=%.3fs/step, train_speed = %.3fs/step' % (
                        np.divide(rollout_duration, nb_rollout_steps), np.divide(train_duration, nb_train_steps)))

            logger.info('')
            mpi_size = MPI.COMM_WORLD.Get_size()
            # Log stats.
            stats = agent.get_stats(memory)
            combined_stats = stats.copy()

            def as_scalar(x):
                if isinstance(x, np.ndarray):
                    assert x.size == 1
                    return x[0]
                elif np.isscalar(x):
                    return x
                else:
                    raise ValueError('expected scalar, got %s' % x)
            combined_stats_sums = MPI.COMM_WORLD.allreduce(np.array([as_scalar(x) for x in combined_stats.values()]))
            combined_stats = {k: v / mpi_size for (k, v) in zip(combined_stats.keys(), combined_stats_sums)}

            # exclude logging zobs_dbg_%d, zobs_dbg_%d_normalized
            summary_list = [(key, combined_stats[key]) for key, v in combined_stats.items() if 'dbg' not in key]
            agent.add_list_summary(summary_list, i_epoch)

            # only print out train stats for epoch_0 for sanity check
            if i_epoch > 0:
                combined_stats = {}

            # Evaluation and statistics.
            if eval_env is not None:
                logger.info('[%d, %d] run evaluation' % (i_epoch, total_nb_eval))
                total_nb_eval = eval_episode(eval_env, render_eval, agent, combined_stats, total_nb_eval)

            logger.info('epoch %d duration: %.2f mins' % (i_epoch, (time.time() - t_epoch_start) / 60))
            for key in sorted(combined_stats.keys()):
                logger.record_tabular(key, combined_stats[key])
            logger.dump_tabular()
            logger.info('')

            if rank == 0:
                agent.store_ckpt(os.path.join(logdir, '%s.ckpt' % 'ddpg'), i_epoch)


def eval_episode(eval_env, render_eval, agent, combined_stats, total_nb_eval=0):
    steps = 0
    submit_episode_reward = 0.
    reset = False
    obs = eval_env.reset()

    while not reset:
        action, q = agent.pi(obs, total_nb_eval, apply_param_noise=False, apply_action_noise=False, compute_Q=True)
        obs, r, _, reset, info = eval_env.step(action)

        steps += 1
        total_nb_eval += 1
        if render_eval:
            eval_env.render()
        submit_episode_reward += r

        summary_list = [('the_action/%d_eval' % i, a) for i, a in enumerate(action)]
        summary_list += [('eval/q', q)]
        if r != 0:
            summary_list += [('eval/q_div_r', q / r)]

        summary_list += [('eval/%s' % tp, info[tp]) for tp in ['rwd_walk', 'rwd_total',
                                                               'rwd_blnc', 'rwd_dist', 'rwd_enrg']]
        tp = 'rwd_agent'
        summary_list += [('eval/%s_x%d' % (tp, info['rf_agent']), info[tp] * info['rf_agent'])]
        summary_list += [('eval_obs/%d' % i, o) for i, o in enumerate(obs[:agent.nb_demo_kine])]
        agent.add_list_summary(summary_list, total_nb_eval)

    agent.add_list_summary([('eval/episode_steps', steps)], total_nb_eval)
    combined_stats['eval/episode_return'] = submit_episode_reward
    combined_stats['eval/episode_steps'] = steps
    return total_nb_eval


# MPI not used in testing mode
def test(eval_env, agent, render_eval=True, nb_epochs=1, start_ckpt=None, **kwargs):
    logger.info('Start testing:', start_ckpt, '\n')
    with tf_util.single_threaded_session() as sess:
        agent.initialize(sess, start_ckpt=start_ckpt)
        sess.graph.finalize()

        for _ in range(nb_epochs):
            combined_stats = {}
            eval_episode(eval_env, render_eval, agent, combined_stats)

            for key in sorted(combined_stats.keys()):
                logger.record_tabular(key, combined_stats[key])
            logger.dump_tabular()
            logger.info('')


def config(group):
    group.add_argument('--nb_epochs', type=int, default=1)
    group.add_argument('--nb_epoch_cycles', type=int, default=2)
    group.add_argument('--nb_rollout_steps', type=int, default=25)
    group.add_argument('--nb_train_steps', type=int, default=10)
    group.add_argument('--start_ckpt', type=str, default='')
    group.add_argument('--hist_files', type=str, default='')
    group.add_argument('--demo_files', type=str,
                       default='demo/train_pos.csv,demo/train_neg.csv,demo/test0.csv,demo/test1.csv,demo/test2.csv')
