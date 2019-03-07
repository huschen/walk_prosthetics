"""The main script, for training, sampling and testing.
    Command examples:
    rm logs/tb/*; python main.py; tensorboard --logdir logs/tb/
    python main.py --mode 1 --time_limit 10 --nb_epochs 3 --done_mimic_thrsh -0.1 --render_eval --verbose_eval
    python main.py --mode 2  --start-ckpt logs/ddpg.ckpt-77 --render-eval --verbose-eval
"""

import argparse
import time
import os
import datetime

import tensorflow as tf
from mpi4py import MPI

from common import logger
from common.misc_util import set_global_seeds, boolean_flag, mpi_complete
import ddpg.ddpg as ddpg
import ddpg.training as training
import ddpg.sampling as sampling

import cust_env

MODE_TRAIN = 0
MODE_SAMPLE = 1
MODE_TEST = 2
MODE_VALID = [MODE_TRAIN, MODE_SAMPLE, MODE_TEST]


def run(mode, render, render_eval, verbose_eval, sanity_run, env_kwargs, model_kwargs, train_kwargs):
    if sanity_run:
        # Mode to sanity check the basic code.
        # Fixed seed and logging dir.
        # Dynamic setting of nb_rollout_steps and nb_train_steps in training.train() is disabled.
        print('SANITY CHECK MODE!!!')

    # Configure MPI, logging, random seeds, etc.
    mpi_rank = MPI.COMM_WORLD.Get_rank()
    mpi_size = MPI.COMM_WORLD.Get_size()

    if mpi_rank == 0:
        logger.configure(dir='logs' if sanity_run else datetime.datetime.now().strftime("train_%m%d_%H%M"))
        logdir = logger.get_dir()
    else:
        logger.set_level(logger.DISABLED)
        logdir = None
    logdir = MPI.COMM_WORLD.bcast(logdir, root=0)


    start_time = time.time()
    # fixed seed when running sanity check, same seed hourly for training.
    seed = 1000000 * mpi_rank
    seed += int(start_time) // 3600 if not sanity_run else 0

    seed_list = MPI.COMM_WORLD.gather(seed, root=0)
    logger.info('mpi_size {}: seeds={}, logdir={}'.format(mpi_size, seed_list, logger.get_dir()))

    # Create envs.
    envs = []
    if mode in [MODE_TRAIN]:
        train_env = cust_env.ProsEnvMon(visualize=render, seed=seed,
                                        fn_step=None,
                                        fn_epis=logdir and os.path.join(logdir, '%d' % mpi_rank),
                                        reset_dflt_interval=2,
                                        ** env_kwargs)
        logger.info('action, observation space:', train_env.action_space.shape, train_env.observation_space.shape)
        envs.append(train_env)
    else:
        train_env = None

    # Always run eval_env, either in evaluation mode during MODE_TRAIN, or MODE_SAMPLE, MODE_TEST.
    # Reset to random states (reset_dflt_interval=0) in MODE_SAMPLE ,
    # Reset to default state (reset_dflt_interval=1) in evaluation of MODE_TRAIN, or MODE_TEST
    reset_dflt_interval = 0 if mode in [MODE_SAMPLE] else 1
    eval_env = cust_env.ProsEnvMon(visualize=render_eval, seed=seed,
                                   fn_step=logdir and os.path.join(logdir, 'eval_step_%d.csv' % mpi_rank),
                                   fn_epis=logdir and os.path.join(logdir, 'eval_%d' % mpi_rank),
                                   reset_dflt_interval=reset_dflt_interval,
                                   verbose=verbose_eval,
                                   ** env_kwargs)
    envs.append(eval_env)

    # Create DDPG agent
    tf.reset_default_graph()
    set_global_seeds(seed)
    assert (eval_env is not None), 'Empty Eval Environment!'

    action_range = (min(eval_env.action_space.low), max(eval_env.action_space.high))
    logger.info('\naction_range', action_range)
    nb_demo_kine, nb_key_states = eval_env.obs_cust_params
    agent = ddpg.DDPG(eval_env.observation_space.shape, eval_env.action_space.shape,
                      nb_demo_kine, nb_key_states,
                      action_range=action_range, save_ckpt=mpi_rank == 0, **model_kwargs)
    logger.debug('Using agent with the following configuration:')
    logger.debug(str(agent.__dict__.items()))

    # Set up agent mimic reward interface, for environment
    for env in envs:
        env.set_agent_intf_fp(agent.get_mimic_rwd)

    # Run..
    logger.info('\nEnv params:', env_kwargs)
    logger.info('Model params:', model_kwargs)
    if mode == MODE_TRAIN:
        logger.info('Start training', train_kwargs)
        training.train(train_env, eval_env, agent, render=render, render_eval=render_eval, sanity_run=sanity_run,
                       **train_kwargs)

    elif mode == MODE_SAMPLE:
        sampling.sample(eval_env, agent, render=render_eval, **train_kwargs)
    else:
        training.test(eval_env, agent, render_eval=render_eval, **train_kwargs)

    # Close up.
    if train_env:
        train_env.close()
    if eval_env:
        eval_env.close()

    mpi_complete(start_time, mpi_rank, mpi_size, non_blocking_mpi=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    main_gname = 'main parameters'
    main_group = parser.add_argument_group(main_gname)
    main_group.add_argument('--mode', type=int, default=0, help='train=0, sample=1, test=2')
    boolean_flag(main_group, 'render', default=False)
    boolean_flag(main_group, 'render_eval', default=False)
    boolean_flag(main_group, 'verbose_eval', default=False)
    boolean_flag(main_group, 'sanity_run', default=True, help='sanity check run')

    env_gname = 'env parameters'
    env_group = parser.add_argument_group(env_gname)
    cust_env.config(env_group)

    model_gname = 'model parameters'
    model_group = parser.add_argument_group(model_gname)
    ddpg.config(model_group)

    train_gname = 'train parameters'
    train_group = parser.add_argument_group(train_gname)
    training.config(train_group)

    args = parser.parse_args()
    arg_groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = group_dict

    assert args.mode in MODE_VALID, 'Invalid mode! Only supports train, sample and test.'
    # Run actual script.
    run(args.mode, args.render, args.render_eval, args.verbose_eval, args.sanity_run,
        arg_groups[env_gname], arg_groups[model_gname], arg_groups[train_gname])
