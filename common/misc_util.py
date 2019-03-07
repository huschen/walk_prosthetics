import random
import time
import datetime
import platform
import hashlib
import json

import numpy as np
from mpi4py import MPI

from common import logger

MPI_DBG = False


def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)


def boolean_flag(parser, name, default=False, help=None):
    """Add a boolean flag to argparse parser.

    Parameters
    ----------
    parser: argparse.Parser
        parser to add the flag to
    name: str
        --<name> will enable the flag, while --no-<name> will disable it
    default: bool or None
        default value of the flag
    help: str
        help string for the flag
    """
    dest = name.replace('-', '_')
    parser.add_argument("--" + name, action="store_true", default=default, dest=dest, help=help)
    parser.add_argument("--no_" + name, action="store_false", dest=dest)


def dbg_tf_init(sess, var):
    # for x in tf.global_variables():
        # print(x.op.name)
    try:
        values = sess.run(var)
        for i, val in enumerate(values):
            if len(val.shape) == 2:
                d = min(val.shape[1], 5)
                if MPI_DBG:
                    print(MPI.COMM_WORLD.Get_rank(), 'init of %s[0,:%d]:' % (var[i].op.name, d), val[0, :d])
                else:
                    logger.info('init of %s[0,:%d]:' % (var[i].op.name, d), val[0, :d])
    except KeyError:
        pass


def mpi_complete(start_time, rank, mpi_size, non_blocking_mpi=True):
    if non_blocking_mpi:
        # non-blocking, release CPU after completion, time inaccurate
        if rank == 0:
            logger.info('%s rank %d completes: %.3fs' % (
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 0, time.time() - start_time))
            completed = [0] * mpi_size
            completed[0] = 1
            # first probe always return EMPTY..
            MPI.COMM_WORLD.Iprobe()
            while sum(completed) != mpi_size:
                time.sleep(10)
                for i, done in enumerate(completed):
                    if done == 0:
                        if MPI.COMM_WORLD.Iprobe(source=i):
                            duration = MPI.COMM_WORLD.recv(source=i)
                            completed[i] = 1
                            logger.info('%s rank %d completes: %.3fs' % (
                                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), i, duration))
                print('completed', completed)
            logger.info('total training time: %.3fs' % (time.time() - start_time))
        else:
            MPI.COMM_WORLD.send(time.time() - start_time, dest=0)
    else:
        # blocking
        x = MPI.COMM_WORLD.allreduce(rank)
        if rank == 0:
            print(x)
            logger.info('total training time: %.3fs' % (time.time() - start_time))


# convert to list for logging
def rad2degree(raw):
    return np.round(np.array(raw) / np.pi * 180.0).astype(int).tolist()


def array_hash(array):
    data = array.tolist() if isinstance(array, np.ndarray) else array
    return hashlib.md5(json.dumps(data).encode('utf-8')).hexdigest()
