from contextlib import ExitStack

import numpy as np
from gym import spaces

from ge_world import mujoco_env
from ge_world.base_envs import MazeCamEnv
from ge_world.mujoco_env import MujocoEnv


def elbow_pos(angles):
    l = 0.02
    x_pos = l * np.cos(angles[0]) + \
            l * np.cos(angles[:2].sum(axis=-1))
    y_pos = l * np.sin(angles[0]) + \
            l * np.sin(angles[:2].sum(axis=-1))
    return x_pos, y_pos


def effector_pos(angles):
    l = 0.02
    x_pos = l * np.cos(angles[0]) + \
            l * np.cos(angles[:2].sum(axis=-1)) + \
            l * np.cos(angles.sum(axis=-1))
    y_pos = l * np.sin(angles[0]) + \
            l * np.sin(angles[:2].sum(axis=-1)) + \
            l * np.sin(angles.sum(axis=-1))
    return x_pos, y_pos


def good_state(state):
    """
    filter for a good goal (state) in the maze.

    :param state:
    :return: bool, True if goal position is good
    """
    x, y = effector_pos(state)
    x_0, y_0 = elbow_pos(state)
    return 0.0 < x and -0.0275 < y < 0.0275 and \
           0.0 < x_0 and -0.0275 < y_0 < 0.0275


good_goal = good_state


def good_state_slot(state):
    """
    filter for a good goal (state) in the maze.

    :param state:
    :return: bool, True if goal position is good
    """
    x, y = effector_pos(state)
    x_0, y_0 = elbow_pos(state)
    return 0.0 < x < 0.0375 and -0.0275 < y < 0.0275 and \
           0.0 < x_0 < 0.0375 and -0.0275 < y_0 < 0.0275


class Peg2DEnv(MujocoEnv, MazeCamEnv):
    """
    2D peg insertion environment.

    There are three variants:
    - Peg2DDiscrete-v0: standard showing the slot on the right
    - Peg2DHiddenDiscrete-v0: makes the slot transparent
    - Peg2DFreeDiscrete-v0: removes the slot for exploration.
    """
    achieved_key = 'x'
    desired_key = 'goal'
    is_good_goal = lambda self, _: good_goal(_)
    is_good_state = lambda self, _: good_state(_)

    def __init__(self,
                 frame_skip=4,
                 obs_keys=(achieved_key, desired_key, "ef_pos"),
                 obj_low=[-np.pi / 2, -np.pi + 0.2, -np.pi + 0.2],
                 obj_high=[np.pi / 2, np.pi - 0.2, np.pi - 0.2],
                 goal_low=-0.02, goal_high=0.02,
                 act_scale=0.5, discrete=False,
                 free=False,  # whether to move the goal out of the way
                 mix_mode=tuple(),
                 view_mode="grey",
                 # hide_slot=False,
                 in_slot=0.1,  # prob. peg to be initialized inside the slot
                 done_on_goal=False,
                 **kwargs
                 ):
        """

        :param frame_skip:
        :param discrete:
        :param id_less:
        :param done_on_goal: False, bool. flag for setting done to True when reaching the goal
        """
        # self.controls = Controls(k_goals=1)
        self.free = free
        self.mix_mode = mix_mode  # cycle through [1, 0.1, -0.1, 0]
        # self.hide_slot = hide_slot
        self.obs_keys = obs_keys
        self.discrete = discrete
        self.done_on_goal = done_on_goal

        self.in_slot = in_slot

        if self.discrete:
            actions = [-act_scale, 0, act_scale]
            self.a_dict = actions
            self.action_space = [spaces.Discrete(3) for _ in range(3)]

        # call super init after initializing the variables.
        import os
        xml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"assets/peg-2d.xml")
        MujocoEnv.__init__(self, model_path=xml_path, frame_skip=frame_skip, set_action_space=True,
                           set_observation_space=False, **kwargs)
        MazeCamEnv.__init__(self, **kwargs)

        # note: Experimental, hard-coded
        _ = dict()
        if 'x' in obs_keys:
            _['x'] = spaces.Box(low=np.array(obj_low), high=np.array(obj_high))
        if 'goal' in obs_keys:
            # todo: double check for agreement with actual goal distribution
            _['goal'] = spaces.Box(low=np.array([goal_low, goal_low]), high=np.array([goal_high, goal_high]))
        if 'img' in obs_keys:
            _['img'] = spaces.Box(
                low=np.zeros((1, self.width, self.height)), high=np.ones((1, self.width, self.height)))
        if 'goal_img' in obs_keys:
            _['goal_img'] = spaces.Box(
                low=np.zeros((1, self.width, self.height)), high=np.ones((1, self.width, self.height)))
        if 'a' in obs_keys:
            _['a'] = spaces.Box(
                low=-act_scale * np.ones((3, self.width, self.height)),
                high=act_scale * np.ones((3, self.width, self.height)))
        self.obj_low = obj_low
        self.obj_high = obj_high
        self.goal_low = goal_low
        self.goal_high = goal_high
        self.view_mode = view_mode
        self.observation_space = spaces.Dict(_)

    def compute_reward(self, achieved, desired, *_):
        success = np.linalg.norm(achieved - desired, axis=-1, ord=2) < 0.02
        return (success - 1).astype(float)

    def step(self, a):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        qvel[:] = 0
        self.set_state(qpos, qvel)

        # todo: remove discrete action support.
        if self.discrete:
            a = [self.a_dict[int(a_i)] for a_i in a]

        for i in range(self.frame_skip):
            self.do_simulation(a, 1)
            # note: return observation *after* simulation. This is how DeepMind Lab does it.
            ob = self._get_obs("x", "goal")
            # do not implement different reward for insertion.
            reward = self.compute_reward(ob['x'], ob['goal'])
            done = bool(1 + reward)
            if reward == 0 or i == (self.frame_skip - 1):
                dist = np.linalg.norm(ob['goal'] - ob['x'], ord=2)
                return self._get_obs(), reward, done, dict(dist=dist, success=float(done))

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[1] = 0
        # side-view
        # self.viewer.cam.lookat[2] = -0.1
        # self.viewer.cam.distance = .7
        # self.viewer.cam.elevation = -55
        # self.viewer.cam.azimuth = 90

        # ortho-view
        self.viewer.cam.lookat[2] = 0
        self.viewer.cam.distance = .74
        self.viewer.cam.elevation = -90
        self.viewer.cam.azimuth = 90

    def _get_goal(self):
        while True:
            goals = self.np_random.uniform(low=self.goal_low, high=self.goal_high, size=(4, 1))
            for goal in goals:
                if good_goal(goal):
                    return goal

    def _get_state(self, slot_y=None):
        if slot_y is not None and self.np_random.rand() < self.in_slot:
            return self._get_goal_state(x=self.np_random.rand() * - 0.01, slot_y=slot_y)

        while True:  # keep it simple.
            states = self.np_random.uniform(low=[0, -2.5, 0], high=[1.5, 0, 2.7], size=(20, 3))

            for state in states:
                if self.free:
                    if good_state(state):
                        return state
                else:
                    if good_state_slot(state):
                        return state

    def _get_goal_state(self, slot_y, x=0., ):
        qpos = np.zeros(3)

        peg_x_y = [x, slot_y / 10]

        base = (0.03 + peg_x_y[0])
        hypo = np.linalg.norm([base, peg_x_y[1]], ord=2)
        a0 = np.arctan(peg_x_y[1] / base)
        a1 = np.arccos(hypo / 0.04)

        qpos[0] = a0 + a1
        qpos[1] = - 2 * a1
        qpos[2] = 0 - qpos[0] - qpos[1]
        return qpos

    def set_slot_pos(self, goal):
        self.model.body_pos[-1, 1] = goal
        self.sim.set_constants()

    def mixed_render(self, *args, **kwargs):
        """A drop-in replacement of default render, but turns the slot on and off
        50% of the time, unless the end-effector intersects the wall."""
        qpos = self.sim.data.qpos.copy()
        flash = self.mix_mode and good_state_slot(qpos)
        if flash:
            slot_pos = self.np_random.choice(self.mix_mode, size=1)
            self.set_slot_pos(slot_pos)
            self.set_state(qpos, self.sim.data.qvel)

        r = self.render(*args, **kwargs)

        if flash:
            self.set_slot_pos(self.slot_pos)
            self.set_state(qpos, self.sim.data.qvel)

        return r

    def reset_model(self, x=None, slot_y=None):
        assert not self.mix_mode or slot_y is None, \
            f"can not set `slot_y={slot_y}` when mixed_mode is {self.mix_mode}"

        if self.free:
            self.slot_pos = 1 if slot_y is None else slot_y
            goal_pos = self._get_state(slot_y=None if self.slot_pos == 1 else self.slot_pos)

            if x is None:  # in free mode, the initial y position should be sampled differently
                diff_goal = self._get_goal()
                x = self._get_state(diff_goal)

        else:
            self.slot_pos = self._get_goal() if slot_y is None else slot_y
            goal_pos = self._get_goal_state(x=-0.003, slot_y=self.slot_pos)

        if x is None:
            x = self._get_state(self.slot_pos)

        self.goal_state = goal_pos.copy()

        self.set_slot_pos(self.slot_pos)
        self.set_state(goal_pos, self.sim.data.qvel)

        img = self.mixed_render(self.view_mode, width=self.width, height=self.height)
        if self.view_mode == "grey":
            img = img[..., None]
        self.goal_img = img.transpose(2, 0, 1)

        # self.set_slot_pos(self.goal)
        self.sim.data.qpos[:] = x
        self.sim.data.qvel[:] = 0  # no velocity
        return self._get_obs()

    def _get_delta(self):
        *delta, _ = self.get_body_com("goal") - self.get_body_com("object")
        return delta

    def _get_obs(self, *obs_keys):
        obs_keys = obs_keys or self.obs_keys
        obs = {}
        qpos = self.sim.data.qpos.flat.copy()
        if 'x' in obs_keys:
            obs['x'] = qpos.copy()
        if 'goal' in obs_keys:
            obs['goal'] = self.goal_state
        if 'ef_pos' in obs_keys:
            obs['ef_pos'] = effector_pos(qpos)
        if 'img' in obs_keys:
            # if self.hide_slot:
            #     self.set_goal_pos(1)
            self.set_state(qpos.copy(), self.sim.data.qvel)

            img = self.mixed_render(self.view_mode, width=self.width, height=self.height)
            if self.view_mode == "grey":
                img = img[..., None]
            obs['img'] = img.transpose(2, 0, 1)
            # if self.hide_slot:
            #     self.set_goal_pos(self.goal)
            self.set_state(qpos, self.sim.data.qvel)
        if 'goal_img' in obs_keys:
            obs['goal_img'] = self.goal_img

        return obs


class MixedPeg2D(Peg2DEnv):
    """this one flips the self.free flag every other environment reset."""

    def __init__(self, *args, free=None, **kwargs):
        assert free is None, "the `free` option is under control by the Mixed environment itself."
        super().__init__(*args, **kwargs)


    def reset_model(self, *args, **kwargs):
        self.free = self.np_random.rand() > 0.5
        return Peg2DEnv.reset_model(self, *args, **kwargs)


if __name__ == "__main__":
    import gym
    from ml_logger import logger
    from time import sleep
    from tqdm import trange

    # env = Peg2DEnv(discrete=True, id_less=False, obs_keys=["x", 'img'])
    env_id = "Peg2D-v0"
    # env_id = "Peg2DFree-v0"
    env = gym.make(env_id)
    seed = 100
    env.seed(seed)

    frames = []

    # while True:
    #     env.reset()
    #     for step in range(10):
    #         act = np.array([0 if step < 5 else 2, 1, 1])
    #         # act = 13
    #         obs, reward, done, info = env.step(act)
    #         env.render()
    #         sleep(0.1)
    #
    # # env.render('human', width=200, height=200)

    for i in trange(100):
        env.reset()
        for step in range(1):
            frame = env.render('rgb', width=200, height=200)
            act = np.random.randint(low=0, high=2, size=3)
            # act = 13
            obs, reward, done, info = env.step(act)
            if i == 0:
                logger.log_image(obs['img'].transpose(1, 2, 0), key=f"../figures/{env_id}_img.png")
                logger.log_image(obs['goal_img'].transpose(1, 2, 0), key=f"../figures/{env_id}_goal_img.png")
                logger.log_image(frame, key=f"../figures/{env_id}.png")
            else:
                env.unwrapped.obs_keys = ["img"]

            frames.append(frame)

    stack = np.stack(frames)
    logger.log_image(stack.min(0), key=f"../figures/{env_id}_spread.png")

    print('done rendering!')

else:
    from gym.envs import register

    # note: kwargs are not passed in to the constructor when entry_point is a function.
    register(
        id="Peg2D-v0",
        entry_point=Peg2DEnv,
        kwargs=dict(discrete=False, view_mode='grey', in_slot=0.1,
                    obs_keys=['x', 'goal', 'img', 'goal_img', 'ef_pos']),
        # max_episode_steps=1000,
    )
    register(
        id="Peg2D-mixed-v0",
        entry_point=MixedPeg2D,
        kwargs=dict(discrete=False, view_mode='grey', in_slot=0.1, mix_mode=(1, 0),
                    obs_keys=['x', 'goal', 'img', 'goal_img', 'ef_pos']),
        # max_episode_steps=1000,
    )
    # register(  # info: not used.
    #     id="Peg2DFixed-v0",
    #     entry_point=Peg2DEnv,
    #     kwargs=dict(discrete=True, view_mode='grey', in_slot=0,
    #                 goal_low=0, goal_high=0,
    #                 obs_keys=['x', 'goal', 'img', 'goal_img', 'a']),
    #     # max_episode_steps=1000,
    # )
    # register(
    #     id="Peg2DFreeSampleRGB-v0",
    #     entry_point=Peg2DEnv,
    #     kwargs=dict(discrete=True, view_mode='rgb', free=True,
    #                 obs_keys=['x', 'goal', 'img', 'a']),
    #     # max_episode_steps=1000,
    # )
    # register(
    #     id="Peg2DFreeSample-v0",
    #     entry_point=Peg2DEnv,
    #     kwargs=dict(discrete=True, view_mode='grey', free=True,
    #                 obs_keys=['x', 'goal', 'img', 'a']),
    #     # max_episode_steps=1000,
    # )
    # register(
    #     id="Peg2DFree-v0",
    #     entry_point=Peg2DEnv,
    #     kwargs=dict(discrete=True, view_mode='grey', free=True,
    #                 obs_keys=['x', 'goal', 'img', 'goal_img', 'a']),
    #     # max_episode_steps=1000,
    # )
