import time
import datetime

import numpy as np
from mpi4py import MPI

from common import logger
from common.misc_util import dbg_tf_init
import common.tf_util as tf_util
from ddpg.pretrain import pretrain_demo


DEFAULT_ACT_VALUE = 0.05


# sample fd1norm2, single action, value [0.8, 0.4]
def sample_fd1norm2(eval_env, render_eval=False, target_nb_resets=-1):
    logger.info('Start sampling')
    mpi_rank = MPI.COMM_WORLD.Get_rank()

    eval_env.reset()
    reset = False

    nb_samples = 0
    nb_resets = 0
    nb_entries = eval_env.nb_reset_entries
    if target_nb_resets == -1:
        target_nb_resets = nb_entries
    logger.info('sampling fd1norm2, nb_init_entries=%d, target_nb_resets=%d' % (nb_entries, target_nb_resets))

    nb_actions = eval_env.action_space.shape[0]
    i_action = 0
    t_start = time.time()
    while nb_resets < target_nb_resets:
        logger.info('\n%s [target=%d] nb_resets=%d, nb_samples=%d:' % (
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), target_nb_resets, nb_resets, nb_samples))

        # Perform rollouts.
        for mean, std in [(0.5, 0.1), (0, 0.05)]:
            for _ in range(5):
                action = np.ones(nb_actions) * DEFAULT_ACT_VALUE
                action[i_action] = np.clip(np.random.normal(mean, std), 0, 1)
                _, _, _, reset, _ = eval_env.step(action)
                nb_samples += 1
                if mpi_rank == 0 and render_eval:
                    eval_env.render()

                if reset:
                    nb_resets += 1
                    eval_env.reset()
                    break

        i_action = (i_action + 1) % nb_actions

    logger.info('nb_samples=%d, nb_resets=%d, duration=%.3fs' % (nb_samples, nb_resets, time.time() - t_start))


def sample(eval_env, agent, render_eval=False, start_ckpt=None, demo_files=None, nb_epochs=1000, **kwargs):
    with tf_util.single_threaded_session() as sess:
        # Run demo network
        agent.initialize(sess, start_ckpt=start_ckpt)
        sess.graph.finalize()
        agent.reset()
        dbg_tf_init(sess, agent.dbg_vars)
        total_nb_train = 0
        pretrain_demo(agent, eval_env, demo_files, total_nb_train, train_params=[(100, 1.0)], start_ckpt=start_ckpt)

        # run sampling
        # agent reward can be disabled.
        # eval_env.set_intf_fp(None)
        sample_fd1norm2(eval_env, render_eval=render_eval, target_nb_resets=nb_epochs)
