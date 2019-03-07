import os
import time

import numpy as np
from mpi4py import MPI

from common import logger
from common.dataset import DataSet
from common.misc_util import array_hash, rad2degree


class DemoTrainer():
    def __init__(self, agent, env, fn_demos):
        self.agent = agent
        self.env = env
        self.load_files(fn_demos)

    def load_files(self, fn_demos):
        assert fn_demos is not '', 'demo files are missing!'
        fn_pos, fn_neg, fn_tests = [], [], []
        for fn in fn_demos.split(','):
            if 'train_pos' in fn:
                fn_pos.append(fn)
            elif 'train_neg' in fn:
                fn_neg.append(fn)
            else:
                fn_tests.append(fn)
        logger.info('\nloaded demo train_pos/train_neg/test data: ', fn_pos, fn_neg, fn_tests)

        train_in, train_out = self.env.read_demo_csv(fn_pos[0])
        self.train_set = DataSet([train_in, train_out], nb_ref=2)
        logger.info('hash=%s\ndemo_train.nb_entries: %d, shape: train_in=%s, train_out=%s' % (
            array_hash(train_in[0:min(100, self.train_set.nb_entries)]),
            self.train_set.nb_entries,
            str(train_in.shape), str(train_out.shape)))

        neg_in_out = self.env.read_demo_csv(fn_neg[0])
        self.neg_set = DataSet(list(neg_in_out), nb_ref=2)
        logger.info('demo_neg.nb_entries: %d' % self.neg_set.nb_entries)

        self.test_sets = []
        for test_file in fn_tests:
            test_in_out = self.env.read_demo_csv(test_file)
            self.test_sets.append(DataSet(list(test_in_out), nb_ref=2))
        logger.info('demo_test.nb_entries: ', [t.nb_entries for t in self.test_sets])

    def train_demo(self, step, lr_decay=1.0, batch_size=128):
        train_in, train_out = self.train_set.next_batch(batch_size=batch_size)
        neg_in, neg_out = self.neg_set.next_batch(batch_size=batch_size)
        demo_loss = self.agent.train_demo(train_in, train_out, neg_in, neg_out, step, neg_pct=0.1, lr_decay=lr_decay)

        demo_test_loss = []
        summary_list = []
        for i, testset in enumerate(self.test_sets):
            test_in, test_out = testset.next_batch(batch_size=batch_size * 2)
            loss_mean, loss_max = self.agent.test_demo(test_in, test_out)
            demo_test_loss.append(loss_mean)
            summary_list += [('demo_loss/test_%d' % i, loss_mean)]
            summary_list += [('demo_loss/test_max_%d' % i, loss_max)]

        self.agent.add_list_summary(summary_list, step)
        return demo_loss, demo_test_loss

    def eval_ref(self, verbose=False):
        nb_demo_kine = self.agent.nb_demo_kine
        obs0, obs1 = self.train_set.fixed_sample
        obs_delta = obs1 - obs0
        result_01 = self.agent.eval_demo(obs0)
        new_obs = obs0 + np.hstack([result_01, np.zeros_like(obs0[:, nb_demo_kine:])])
        result_12 = self.agent.eval_demo(new_obs)

        if verbose:
            logger.info('ref', rad2degree(obs_delta[0, :nb_demo_kine]),
                        '\t', rad2degree(obs_delta[1, :nb_demo_kine]))
        logger.info('RES', rad2degree(result_01[0]), '\t', rad2degree(result_12[0]))


def pretrain_demo(agent, env, demo_files, total_nb_train, train_params, start_ckpt=None):
    demo_traner = DemoTrainer(agent, env, demo_files)
    if not start_ckpt:
        t_start = time.time()
        batch_size = agent.batch_size
        for nb_steps, lr_decay in train_params:
            logger.info('\n[%d] train_demonstrator for %d steps, lr_decay=%.2f' % (
                total_nb_train, nb_steps, lr_decay))
            train_losses = []
            test_losses = []

            demo_traner.eval_ref(verbose=True)
            for _ in range(nb_steps):
                train_ls, test_ls = demo_traner.train_demo(total_nb_train, lr_decay=lr_decay, batch_size=batch_size)
                train_losses.append(train_ls)
                test_losses.append(test_ls)
                total_nb_train += 1

            demo_traner.eval_ref(verbose=False)
            logger.info('demo_train_loss: final=%.2e, first=%.2e, avg=%.2e' % (
                train_losses[-1], train_losses[0], np.mean(train_losses)))
            test_losses = np.array(test_losses).T
            for test_ls in test_losses:
                logger.info('demo_test_loss: final=%.2e, first=%.2e, avg=%.2e' % (
                    test_ls[-1], test_ls[0], np.mean(test_ls)))
        logger.info('pretrain_demo duration=%.3fs' % (time.time() - t_start))
    else:
        demo_traner.eval_ref(verbose=True)
    return total_nb_train


def load_history(agent, env, hist_files):
    logger.info()
    # load data
    if hist_files:
        for fn_tmp in hist_files.split(','):
            fn_base, fn_ext = os.path.splitext(fn_tmp)
            # this relay on the file names following the format base_rank.ext
            # todo: when mpi_size>1, not enough experience, there might be hand shake problems.
            fn_rank = '%s_%d%s' % (fn_base, MPI.COMM_WORLD.Get_rank(), fn_ext)
            rcd = env.read_step_csv(fn_rank)
            if rcd:
                start = max(0, len(rcd[0]) - int(agent.memory.limit))
                rcd = [r[start:] for r in rcd]
                agent.store_multrans(agent.memory, *rcd)
            logger.info('loaded experiences from %s, memory.nb_entries=%d' % (fn_rank, agent.memory.nb_entries))

    if agent.memory.nb_entries > 0:
        # states
        stats = agent.get_stats(agent.memory)
        combined_stats = stats.copy()
        for key in sorted(combined_stats.keys()):
            logger.record_tabular(key, combined_stats[key])
        logger.dump_tabular()

        print('rank%d, loaded experiences from %s, memory.nb_entries=%d' % (
            MPI.COMM_WORLD.Get_rank(), hist_files, agent.memory.nb_entries))
