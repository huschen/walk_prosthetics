""" Prosthetics Walk Environment class and its logging wrapper.
"""
import csv
import json
import time
import numpy as np

from tensorflow.python.platform import gfile
from osim.env import ProstheticsEnv
from opensim import Vector as opensim_vect

from common import logger
from common.dataset import DataList
from common.misc_util import rad2degree


ACTIVATIONS = ['abd_r', 'add_r',
               'hamstrings_r', 'bifemsh_r', 'glut_max_r', 'iliopsoas_r', 'rect_fem_r', 'vasti_r',
               'abd_l', 'add_l',
               'hamstrings_l', 'bifemsh_l', 'glut_max_l', 'iliopsoas_l', 'rect_fem_l', 'vasti_l',
               'gastroc_l', 'soleus_l', 'tib_ant_l']

MUSCLE_NAMES = ['abd_r', 'add_r', 'hamstrings_r', 'bifemsh_r', 'glut_max_r', 'iliopsoas_r',
                'rect_fem_r', 'vasti_r', 'abd_l', 'add_l', 'hamstrings_l', 'bifemsh_l', 'glut_max_l',
                'iliopsoas_l', 'rect_fem_l', 'vasti_l', 'gastroc_l', 'soleus_l', 'tib_ant_l']

JOINT_NAMES = ['ground_pelvis_0', 'ground_pelvis_1', 'ground_pelvis_2',
               'ground_pelvis_3', 'ground_pelvis_4', 'ground_pelvis_5',
               'hip_r_0', 'hip_r_1', 'hip_r_2', 'knee_r_0', 'ankle_r_0',
               'hip_l_0', 'hip_l_1', 'hip_l_2', 'knee_l_0', 'ankle_l_0',
               'back_0']

JOINT_POS_LOCKED = ['ground_pelvis_3', 'ground_pelvis_5', 'hip_r_2', 'hip_l_2', 'back_0']

NB_JOINTS = len(JOINT_NAMES)
JOINT_POS_ACTIVE = [n for n in JOINT_NAMES if n not in JOINT_POS_LOCKED]

# weighted sum of square error
JOINT_ERR_WEIGHT = [1, 1, 1, 0.5,
                    2, 1, 2, 1,
                    2, 1, 2, 1]

# the percentage (on each side) to increase the demo joint pos range
JOINT_INC_FACTOR = [5.5, 0.5, 0.5, 1,
                    0.75, 0.15, 0.5, 0.5,
                    0.75, 0.15, 0.5, 0.5]

STATE_RESTART = -1

DEBUG_ENV = False
DEBUG_DONE = False
DEBUG_RWD_WALK = False


def flatten_keys(d, p=''):
    if not isinstance(d, dict):
        return []
    res = []
    for key, val in sorted(d.items()):
        if isinstance(val, dict):
            res.extend(flatten_keys(val, '%s_%s' % (p, key)))
        elif isinstance(val, list):
            res.extend(['%s_%s_%d' % (p, key, i) for i in range(len(val))])
        else:
            res.extend(['%s_%s_%d' % (p, key, 0)])
    return res


def flatten(d):
    res = []
    if isinstance(d, dict):
        for _, val in sorted(d.items()):
            res.extend(flatten(val))
    elif isinstance(d, list):
        res = d
    else:
        res = [d]
    return res


def gen_obs_cust(dict_k2i=None):
    if dict_k2i:
        def append(idx, item, idict=dict_k2i):
            # no check of dictionary key, in order to catch key change
            idx.append(idict[item])
    else:
        # to calculate the size of observation space
        def append(idx, item):
            idx.append(0)

    idx = []

    for joint in JOINT_POS_ACTIVE:
        append(idx, '_joint_pos_%s' % (joint))

    # nb kinematic features for imitation learning
    nb_demo_kine = len(idx)

    for body in ['pelvis', 'pros_foot_r', 'calcn_l', 'toes_l']:
        append(idx, '_body_vel_%s_0' % body)
    # append(idx, dict_k2i, '_misc_mass_center_vel_0')

    for muscle in MUSCLE_NAMES:
        for key in ['activation', 'fiber_length']:
            # _muscles_abd_l_activation_0
            append(idx, '_muscles_%s_%s_0' % (muscle, key))

    # nb features for critic to learn to model/predict
    # also those are features that do not need to normalise
    nb_key_states = len(idx)

    for joint in JOINT_NAMES:
        append(idx, '_joint_vel_%s' % (joint))

    append(idx, '_forces_ankleSpring_0')
    for i in range(18):
        # '_forces_pros_foot_r_0_0'
        append(idx, '_forces_pros_foot_r_0_%d' % i)

    for i in range(24):
        # '_forces_foot_l_0
        append(idx, '_forces_foot_l_%d' % i)

    return idx, nb_demo_kine, nb_key_states


def gen_rwd_walk(obs_flat, dict_k2i, obs_old_flat, info=None,
                 rf_vel=0, rf_dist=50, rf_enrg=0.002, rf_blnc=0.1, rf_live=0.5):

    def rd(name, obs=obs_flat, d=dict_k2i):
        return obs[d[name]]

    # tvel = rd('_target_vel_0')
    # rwd_vel = (tvel ** 2 - (rd('_body_vel_pelvis_0') - tvel) ** 2) / tvel * rf_vel
    rwd_vel = min(rd('_body_vel_pelvis_0'), 1) * rf_vel

    # dist = rd('_misc_mass_center_pos_0') - rd('_misc_mass_center_pos_0', obs=obs_old_flat)
    rwd_dist = (rd('_body_pos_pelvis_0') - rd('_body_pos_pelvis_0', obs=obs_old_flat)) * rf_dist

    rwd_enrg = -sum([rd('_muscles_%s_activation_0' % m) ** 2 for m in MUSCLE_NAMES]) * rf_enrg

    penalty = [0] * 6
    penalty[0] = rd('_misc_mass_center_vel_1') ** 2
    penalty[1] = abs(rd('_misc_mass_center_pos_1') - rd('_body_pos_torso_1'))
    penalty[2] = abs((rd('_body_pos_calcn_l_0') + rd('_body_pos_pros_foot_r_0')) * 0.5 - rd('_misc_mass_center_pos_0'))
    penalty[3] = abs((rd('_body_pos_calcn_l_2') + rd('_body_pos_pros_foot_r_2')) * 0.5 - rd('_misc_mass_center_pos_2'))

    if DEBUG_RWD_WALK:
        print('p[0]', penalty[0], 'vel_ver(cmo)', rd('_misc_mass_center_pos_1') * rf_blnc)
        print('p[1]', penalty[1], 'pos(com-torso)', rd('_misc_mass_center_pos_1') * rf_blnc,
              rd('_body_pos_torso_1') * rf_blnc)
        print('p[2]', penalty[2], 'pos_x(com-feet)', rd('_body_pos_calcn_l_0') * rf_blnc,
              rd('_body_pos_pros_foot_r_0') * rf_blnc, rd('_misc_mass_center_pos_0') * rf_blnc)
        print('p[3]', penalty[3], 'pos_z(com-feet)', rd('_body_pos_calcn_l_2') * rf_blnc,
              rd('_body_pos_pros_foot_r_2') * rf_blnc, rd('_misc_mass_center_pos_2') * rf_blnc)

    rwd_blnc = -sum(penalty) * rf_blnc

    reward = (rwd_vel + rwd_dist + rwd_enrg + rwd_blnc + rf_live) * 1
    if info is not None:
        info['rwd_vel'] = rwd_vel
        info['rwd_dist'] = rwd_dist
        info['rwd_enrg'] = rwd_enrg
        info['rwd_blnc'] = rwd_blnc
    return reward


class ProsEnv(ProstheticsEnv):
    """ Env with customised observation, reward and done_state.
        Support random/fixed state resetting.
        Temporarily removed arguments: mpi_rank, mpi_size.
    """
    def __init__(self, model='3D', prosthetic=True, visualize=True, integrator_accuracy=5e-5, seed=0, difficulty=0,
                 time_limit=300, done_height=0.6, done_mimic_thrsh=-1e-2,
                 rf_agent=0, rf_walk='rf_vel=0,rf_dist=20', verbose=False):
        # init env
        super().__init__(visualize=visualize, integrator_accuracy=integrator_accuracy, difficulty=difficulty,
                         seed=seed)
        self.change_model(model=model, prosthetic=prosthetic, difficulty=difficulty, seed=seed)

        # osim env variables
        # over-write time_limit
        self.spec.timestep_limit = time_limit
        # cust_env variables
        self.done_height = done_height
        self.done_mimic_thrsh = done_mimic_thrsh
        self.rf_agent = rf_agent
        # reward_walk flags
        self.rf_walk_cfg = {}
        for dw in rf_walk.replace(' ', '').split(','):
            k, v = dw.split('=')
            self.rf_walk_cfg[k] = float(v)
        self.verbose = verbose

        # in osim, reset is needed before calling get_state_desc()
        # in osim, target_velocity is only set after reset.
        super().reset(project=False)
        self.prev_state_desc = None

        osim_init_state = self.osim_model.model.getStateVariableValues(self.osim_model.state)
        self.default_init_state = [osim_init_state.get(i) for i in range(0, len(osim_init_state))]

        # customised observations
        obs_dict = self.get_state_desc()
        # keep self.obs_keys for ProsEnvMon logging
        self.obs_keys = flatten_keys(obs_dict)
        self.dict_k2i = {k: i for i, k in enumerate(self.obs_keys)}
        self.obs_cust_idx, self.nb_demo_kine, self.nb_key_states = gen_obs_cust(self.dict_k2i)
        self.obs_flat = None

        # the order of osim state: joint_i_pos, joint_i_vel ...
        self.osim_state_idx = [JOINT_NAMES.index(n) * 2 for n in JOINT_POS_ACTIVE]

        # to be set by set_agent_intf_fp()
        self.get_mimic_rwd = None
        # to be set by ProsEnvMon(), need fixing?
        self.joint_minmax = None
        self.eplen = 0

    @property
    def obs_cust_params(self):
        logger.info('obs_cust_params: nb_demo_kine=%d, nb_key_states=%d' % (
            self.nb_demo_kine, self.nb_key_states))
        return self.nb_demo_kine, self.nb_key_states

    def set_agent_intf_fp(self, get_mimic_rwd):
        self.get_mimic_rwd = get_mimic_rwd

    def read_osim_angles(self, info=''):
        check_state = self.osim_model.model.getStateVariableValues(self.osim_model.state)
        state = [check_state.get(i) for i in range(0, NB_JOINTS * 2, 2)]
        logger.info('**%s' % info, rad2degree(state))

    def get_cust_rwd_done(self, obs_old_flat, obs_flat, info=None):
        # calculate reward
        rwd_walk = gen_rwd_walk(obs_flat, self.dict_k2i, obs_old_flat, info=info, **self.rf_walk_cfg)
        obs_cust = np.array([[obs_flat[i] for i in self.obs_cust_idx]])
        if self.get_mimic_rwd:
            obs_old_cust = np.array([[obs_old_flat[i] for i in self.obs_cust_idx]])
            rwd_agent, demo_aprx = self.get_mimic_rwd(obs_old_cust, obs_cust)
        else:
            rwd_agent = 0

        rwd_total = rwd_walk + rwd_agent * self.rf_agent
        if info:
            info['rwd_agent'] = rwd_agent
            info['rwd_walk'] = rwd_walk
            info['rwd_total'] = rwd_total
            info['rf_agent'] = self.rf_agent

        if self.verbose and info:
            if self.get_mimic_rwd:
                obs_delta_kine = (obs_cust - obs_old_cust)[:, :self.nb_demo_kine]
                agent_dbg = [rad2degree(obs_old_cust[0, :self.nb_demo_kine]), rad2degree(obs_delta_kine[0]),
                             rad2degree(demo_aprx[0, :self.nb_demo_kine] - obs_delta_kine[0])]
            else:
                agent_dbg = None

            logger.info('%.3f, [%.3f %.3f], %.3f, [%.3f,%.3f], %.2e' % (
                info['rwd_total'], info['rwd_vel'], info['rwd_dist'], info['rwd_enrg'], info['rwd_blnc'],
                rwd_agent * self.rf_agent, rwd_agent), '\t', agent_dbg)

        # calculate state_based is_done
        actual_height = obs_flat[self.dict_k2i['_body_pos_pelvis_1']]
        done = actual_height < self.done_height
        done = done or (rwd_agent < self.done_mimic_thrsh if rwd_agent is not None else False)

        joint_pos_min, joint_pos_max = self.joint_minmax
        upper_done = np.maximum(obs_cust[0, 0:self.nb_demo_kine] - joint_pos_max, 0)
        lower_done = np.maximum(joint_pos_min - obs_cust[0, 0:self.nb_demo_kine], 0)
        range_done = np.sum(upper_done) + np.sum(lower_done) > 0
        done = done or range_done

        if DEBUG_DONE:
            if done:
                logger.info('done=%d,eplen=%d: height=%.2f, rwd_agent=%.2e, lower_done=%d,upper_done=%d ' % (
                    int(done), self.eplen, actual_height, rwd_agent, np.sum(lower_done) > 0, np.sum(upper_done) > 0))
                if range_done:
                    logger.info('lower, upper check', lower_done.tolist(), upper_done.tolist())

        return rwd_total, done

    def get_observation(self):
        obs_flat = flatten(self.get_state_desc())
        obs_cust = [obs_flat[i] for i in self.obs_cust_idx]
        return obs_cust

    def get_observation_space_size(self):
        try:
            return len(self.obs_cust_idx)
        except AttributeError:
            # when function is called before reset, self.obs_cust_idx is not defined
            return len(gen_obs_cust(None)[0])

    def step(self, action):
        # self.prev_state_desc is used by ProstheticsEnv.get_prev_state_desc()
        self.prev_state_desc = self.get_state_desc()
        obs_old_flat = flatten(self.prev_state_desc)

        self.osim_model.actuate(action)
        self.osim_model.integrate()
        if DEBUG_ENV and self.eplen == 0:
            self.read_osim_angles(info='a')

        # keep self.obs_flat for ProsEnvMon logging
        self.obs_flat = flatten(self.get_state_desc())
        info = {}
        r, done = self.get_cust_rwd_done(obs_old_flat, self.obs_flat, info=info)
        reset = done or (self.osim_model.istep >= self.spec.timestep_limit)
        obs_cust = [self.obs_flat[i] for i in self.obs_cust_idx]

        return [obs_cust, r, done, reset, info]

    def reset(self, target_state=None):
        assert target_state is not None
        new_state = np.copy(self.default_init_state)
        new_state[self.osim_state_idx] = target_state

        self.osim_model.state = self.osim_model.model.initializeState()
        self.osim_model.model.setStateVariableValues(self.osim_model.state, opensim_vect(new_state))

        self.osim_model.model.equilibrateMuscles(self.osim_model.state)
        self.osim_model.state.setTime(0)
        self.osim_model.istep = 0
        self.osim_model.reset_manager()
        self.osim_model.state_desc_istep = -1

        if DEBUG_ENV:
            logger.info('--', rad2degree(target_state))
            self.read_osim_angles(info='0')

        # keep self.obs_flat for ProsEnvMon logging
        self.obs_flat = flatten(self.get_state_desc())
        obs_cust = [self.obs_flat[i] for i in self.obs_cust_idx]

        if self.verbose:
            logger.info('rwd_total, [vel,dist], rwd_enrg, [blnc,agent_x%d], rwd_agent \
            obs0,\t obs_delta_kine,\t aprx_demo-obs_delta_kine' % (self.rf_agent))

        return obs_cust

    def close(self):
        super().close()
        return


def scale_minmax(minmax_org, scale_rate=1, scale_list=JOINT_INC_FACTOR, pretty_print=lambda x: x):
    logger.info('scale_minmax: scale_rate=%d, scale_list=%s' % (scale_rate, JOINT_INC_FACTOR))
    data_min, data_max = minmax_org
    data_range = data_max - data_min
    logger.info('org min_max range', pretty_print(data_range))
    data_min -= data_range * scale_list * scale_rate
    data_max += data_range * scale_list * scale_rate
    logger.info('scaled min_max range', pretty_print(data_max - data_min))
    return np.vstack((data_min, data_max))


class ProsEnvMon(ProsEnv):
    """Wrapper class for Env, dealing with csv recording and data processing.
    """
    def __init__(self, fn_reset_states=None, reset_dflt_interval=1,
                 fn_demo_range=None, fn_step=None, fn_epis=None, **kwargs):
        super().__init__(**kwargs)

        # reset-state data, for random reset-state
        self.nb_reset = 0
        reset_data = self.read_basic_csv(fn_reset_states, header_check=JOINT_POS_ACTIVE)
        assert reset_data is not None, 'Loading fn_reset_states %s failed!' % fn_reset_states
        self.reset_states = DataList(reset_data)
        self.reset_dflt_interval = int(reset_dflt_interval)
        self.default_reset_state = self.reset_states.fixed_sample()
        logger.info('nb_reset_entries=%d, default reset state:' % self.reset_states.nb_entries,
                    rad2degree(self.default_reset_state), '\n', self.default_reset_state.tolist())

        # demo-range data, for setting joint position min_max
        demo_data = self.read_basic_csv(fn_demo_range, header_check=JOINT_POS_ACTIVE)
        assert demo_data is not None, 'Loading fn_demo_range %s failed!' % fn_demo_range

        # demo-train/test data, for mimic learning
        self.demo_header = JOINT_POS_ACTIVE

        # calculate joint pos min and max for early stopping
        joint_pos_min = np.min(demo_data, axis=0)
        joint_pos_max = np.max(demo_data, axis=0)
        self.joint_minmax_org = np.vstack((joint_pos_min, joint_pos_max))
        logger.info('org demo joint_pos_min', rad2degree(joint_pos_min))
        logger.info('org demo joint_pos_max', rad2degree(joint_pos_max))
        self.joint_minmax = scale_minmax(self.joint_minmax_org, pretty_print=rad2degree)

        # step logging
        # for step csv reading
        self.step_header = ['restart']
        self.step_act_idx = len(self.step_header)
        self.step_header += ['a%d' % i for i in range(self.action_space.shape[0])] + self.obs_keys
        if fn_step:
            self.fp_step = gfile.GFile(fn_step, 'w')
            self.lg_step = csv.writer(self.fp_step, lineterminator='\n')
            self.lg_step.writerow(self.step_header)
        else:
            self.fp_step = self.lg_step = None

        # episode logging
        if fn_epis:
            self.tstart = time.time()
            self.fp_epis = open(fn_epis + '.monitor.csv', "wt")
            self.fp_epis.write('#%s\n' % json.dumps({"t_start": self.tstart}))
            self.lg_epis = csv.DictWriter(self.fp_epis, fieldnames=('r', 'l', 't'))
            self.lg_epis.writeheader()
            self.fp_epis.flush()

            self.eprew = 0.0
            self.eplen = 0
        else:
            self.fp_epis = self.lg_epis = None

    @property
    def nb_reset_entries(self):
        return self.reset_states.nb_entries

    def step(self, action):
        obs_cust, r, done, reset, info = super().step(action)

        if self.lg_step:
            restart = 0
            step_info = [restart] + action.tolist() + self.obs_flat
            assert len(step_info) == len(self.step_header)
            self.lg_step.writerow(step_info)

        if self.lg_epis:
            self.eprew += r
            self.eplen += 1
            if reset:
                # time t, is for matching train and evaluation result, not for calculate episode duration
                epinfo = {"r": round(self.eprew, 6), "l": self.eplen, "t": round(time.time() - self.tstart, 6)}
                self.lg_epis.writerow(epinfo)
                self.fp_epis.flush()
                self.eprew = 0.0
                self.eplen = 0
                self.tstart = time.time()

        return [obs_cust, r, done, reset, info]

    def reset(self):
        if self.reset_dflt_interval > 0 and self.nb_reset % self.reset_dflt_interval == 0:
            state = self.default_reset_state
        else:
            state = self.reset_states.random_sample()
        obs_cust = super().reset(target_state=state)
        self.nb_reset += 1

        if self.lg_step:
            action = [-1.0] * self.action_space.shape[0]
            step_info = [STATE_RESTART] + action + self.obs_flat
            self.lg_step.writerow(step_info)

        return obs_cust

    def close(self):
        super().close()
        if self.fp_step:
            self.fp_step.close()
        if self.fp_epis:
            self.fp_epis.close()

    def read_basic_csv(self, fname, header_check):
        with open(fname, 'r') as fp:
            reader = csv.reader(fp, delimiter=',')
            header = next(reader)
            assert header[1:] == header_check, '(%d, %d), csv of wrong version!' % (len(header), len(header_check))
            # osim require 'float' data type
            data = np.array([line[1:] for line in reader]).astype(np.float)
            return data

    def read_step_csv(self, fname):
        try:
            with open(fname, 'r') as fp:
                rcd_obs, rcd_action, rcd_r, rcd_new_obs, rcd_done = [], [], [], [], []

                reader = csv.reader(fp, delimiter=',')
                header = next(reader)
                nb_items = len(header)
                assert nb_items == len(self.step_header), \
                    '(%d, %d), csv of wrong version!' % (nb_items, len(self.step_header))

                # step_info = [int(restart)] + action.tolist() + obs_flat(new)
                act_idx = self.step_act_idx
                nb_actions = self.action_space.shape[0]
                obs_idx = act_idx + nb_actions
                count_dones = 0

                for line in reader:
                    x = np.array(line).astype(np.float)
                    if len(x) == nb_items:
                        if x[0] == STATE_RESTART:
                            obs_old_flat = x[obs_idx:]
                        else:
                            obs_flat = x[obs_idx:]
                            rwd_total, done = self.get_cust_rwd_done(obs_old_flat, obs_flat)
                            count_dones += done

                            obs_old_cust = obs_old_flat[self.obs_cust_idx].tolist()
                            obs_cust = obs_flat[self.obs_cust_idx].tolist()
                            action = x[act_idx:nb_actions + act_idx].tolist()

                            rcd_obs.append(obs_old_cust)
                            rcd_action.append(action)
                            rcd_r.append(rwd_total)
                            rcd_new_obs.append(obs_cust)
                            rcd_done.append(done)
                            # logger.info(done, rwd_total, action)
                            obs_old_flat = obs_flat

                logger.info('read_step_csv: dones_counted=%d' % count_dones)
                return rcd_obs, rcd_action, rcd_r, rcd_new_obs, rcd_done

        except FileNotFoundError:
            return None

    def read_demo_csv(self, fname):
        with open(fname, 'r') as fp:
            reader = csv.reader(fp, delimiter=',')
            header = next(reader)
            assert header[1:] == self.demo_header, \
                '(%d, %d), csv of wrong version' % (len(header), len(self.demo_header))
            nb_features = len(header) - 1
            pad_size = self.observation_space.shape[0] - nb_features

            data_in = []
            data_out = []
            for line in reader:
                x = np.array(line).astype(np.float)
                if x[0] == 0:
                    s0 = x[1:]
                else:
                    s1 = x[1:]
                    data_in.append(s0)
                    data_out.append(s1)
                    s0 = s1
            obs0 = np.pad(np.array(data_in), ((0, 0), (0, pad_size)), 'constant', constant_values=(0))
            obs1 = np.pad(np.array(data_out), ((0, 0), (0, pad_size)), 'constant', constant_values=(0))
            return obs0, obs1


def config(env_group):
    env_group.add_argument('--difficulty', type=int, default=0)
    env_group.add_argument('--integrator_accuracy', type=float, default=5e-4)
    env_group.add_argument('--done_height', type=float, default=0.875)
    env_group.add_argument('--done_mimic_thrsh', type=float, default=-9e-3)
    env_group.add_argument('--time_limit', type=int, default=300)
    env_group.add_argument('--rf_agent', type=int, default=50)
    env_group.add_argument('--rf_walk', type=str,
                           default='rf_vel=0,rf_dist=50,rf_enrg=0.002,rf_blnc=0.1,rf_live=0.2')
    env_group.add_argument('--fn_reset_states', type=str, default='demo/reset_states.csv')
    env_group.add_argument('--fn_demo_range', type=str, default='demo/state_range.csv')
