import numpy as np
from gym import spaces

from ge_world import mujoco_env


def good_goal(goal):
    """
    filter for a good goal (state) in the maze.

    :param goal:
    :return: bool, True if goal position is good
    """
    return goal[0] < 0.26 and -0.26 < goal[0]


def good_state(state):
    """
    filter for a good goal (state) in the maze.

    :param state:
    :return: bool, True if goal position is good
    """
    return state[0] < (np.pi / 2) and (- np.pi / 2) < state[0]


class Peg2DEnv(mujoco_env.MujocoEnv):
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

    def __init__(self, frame_skip=10, obs_keys=(achieved_key, desired_key),
                 obj_low=[-np.pi / 2, -np.pi + 0.2, -np.pi + 0.2],
                 obj_high=[np.pi / 2, np.pi - 0.2, np.pi - 0.2],
                 goal_low=-0.02, goal_high=0.02,
                 act_scale=0.5, discrete=False,
                 free=False,  # whether to move the goal out of the way
                 view_mode="grey",
                 in_slot=0.1,  # prob. peg to be initialized inside the slot
                 done_on_goal=False):
        """

        :param frame_skip:
        :param discrete:
        :param id_less:
        :param done_on_goal: False, bool. flag for setting done to True when reaching the goal
        """
        # self.controls = Controls(k_goals=1)
        self.discrete = discrete
        self.done_on_goal = done_on_goal

        self.in_slot = in_slot

        if self.discrete:
            set_spaces = False
            actions = [-act_scale, 0, act_scale]
            self.a_dict = actions
            self.action_space = [spaces.Discrete(3) for _ in range(3)]
        else:
            set_spaces = True

        # call super init after initializing the variables.
        import os
        xml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"assets/peg-2d.xml")

        mujoco_env.MujocoEnv.__init__(self, xml_path, frame_skip=frame_skip, set_spaces=set_spaces)

        # note: Experimental, hard-coded
        self.width = 64
        self.height = 64
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
        self.obs_keys = obs_keys
        self.free = free

    # @property
    # def k(self):
    #     return self.controls.k

    def compute_reward(self, achieved, desired, *_):
        return 1
        # success = np.linalg.norm(achieved - desired, axis=-1) < 0.02
        # return (success - 1).astype(float)

    def step(self, a):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        qvel[:] = 0
        self.set_state(qpos, qvel)
        if self.discrete:
            a = [self.a_dict[int(a_i)] for a_i in a]
        self.do_simulation(a, self.frame_skip)
        # note: return observation *after* simulation. This is how DeepMind Lab does it.
        ob = self._get_obs()

        dist = np.linalg.norm(self.goal_state - qpos, ord=2)
        done = dist < 0.04
        reward = float(done) - 1

        # offer raw action to agent
        if 'a' in self.obs_keys:
            ob['a'] = a.copy()

        return ob, reward, done, dict(dist=dist, success=float(done))

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

    def effector_pos(self, angles):
        l = 0.02
        x_pos = l * np.cos(angles[0]) + \
                l * np.cos(angles[:2].sum(axis=-1)) + \
                l * np.cos(angles.sum(axis=-1))
        y_pos = l * np.sin(angles[0]) + \
                l * np.sin(angles[:2].sum(axis=-1)) + \
                l * np.sin(angles.sum(axis=-1))
        return x_pos, y_pos

    def _get_state(self, goal=None):
        if self.np_random.rand() < self.in_slot:
            return self._get_goal_state(x=self.np_random.rand() * - 0.01, goal=goal)

        while True:
            import numpy as np

            states = self.np_random.uniform(  # info: use single polar for slot mode.
                low=[-1.5 if self.free else 0, 0, -0.5], high=[1.5, 1.5, 0.5], size=(4, 3))
            states[:, 1] = - np.sign(states[:, 0]) * states[:, 1] - states[:, 0]
            states[:, 2] = states[:, 2] - states[:, :2].sum(axis=-1)

            for state in states:
                finger_pos = self.effector_pos(state)
                if not self.free:
                    if 0.02 <= finger_pos[0] < 0.035:
                        return state
                    continue
                if good_state(state):
                    return state

    def _get_goal_state(self, goal, x=0., ):
        # if self.goal_high:
        #     assert goal == 0, "only zero goal position is allowed for fixed slot."
        qpos = np.zeros(3)

        peg_x_y = [x, goal / 10]

        base = (0.03 + peg_x_y[0])
        hypo = np.linalg.norm([base, peg_x_y[1]], ord=2)
        a0 = np.arctan(peg_x_y[1] / base)
        if self.free:
            a1 = np.sign(self.np_random.rand() - 0.5) * np.arccos(hypo / 0.04)
        else:
            a1 = np.arccos(hypo / 0.04)
        qpos[0] = a0 + a1
        qpos[1] = - 2 * a1
        qpos[2] = 0 - qpos[0] - qpos[1]
        return qpos

    def set_goal_pos(self, goal):
        self.model.body_pos[-1, 1] = goal
        self.sim.set_constants()

    def reset_model(self, x=None, goal=None):
        if self.free:
            # assert goal is None, "can not set goal in free mode."
            # assert not self.in_slot, "can not have in_slot probability."
            goal = self._get_goal()
            goal_pos = self._get_state(goal=goal)

            if x is None:  # in free mode, the initial y position should be sampled differently
                diff_goal = self._get_goal()
                x = self._get_state(diff_goal)

            # now move the slot out of the way.
            self.goal = 1

        else:
            self.goal = goal or self._get_goal()
            goal_pos = self._get_goal_state(x=-0.003, goal=self.goal)

            if x is None:
                x = self._get_state(self.goal)

        self.goal_state = goal_pos.copy()

        self.set_goal_pos(1)
        self.set_state(goal_pos, self.sim.data.qvel)

        img = self.render(self.view_mode, width=self.width, height=self.height)
        if self.view_mode == "grey":
            img = img[..., None]
        self.goal_img = img.transpose(2, 0, 1) / 255

        # Now genrate the initial positions

        self.set_goal_pos(self.goal)
        self.sim.data.qpos[:] = x
        self.sim.data.qvel[:] = 0  # no velocity
        return self._get_obs()

    def _get_delta(self):
        *delta, _ = self.get_body_com("goal") - self.get_body_com("object")
        return delta

    def _get_obs(self):
        obs = {}
        qpos = self.sim.data.qpos.flat.copy()
        if 'x' in self.obs_keys:
            obs['x'] = qpos.copy()
        if 'goal' in self.obs_keys:
            obs['goal'] = self.goal
        if 'img' in self.obs_keys:
            self.set_goal_pos(1)
            self.set_state(qpos.copy(), self.sim.data.qvel)

            img = self.render(self.view_mode, width=self.width, height=self.height)
            if self.view_mode == "grey":
                img = img[..., None]
            obs['img'] = img.transpose(2, 0, 1) / 255
            self.set_goal_pos(self.goal)
            self.set_state(qpos, self.sim.data.qvel)
        if 'goal_img' in self.obs_keys:
            obs['goal_img'] = self.goal_img

        return obs


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
        kwargs=dict(discrete=True, view_mode='grey', in_slot=0,
                    obs_keys=['x', 'goal', 'img', 'goal_img', 'a']),
        max_episode_steps=1000,
    )
    register(  # info: not used.
        id="Peg2DFixed-v0",
        entry_point=Peg2DEnv,
        kwargs=dict(discrete=True, view_mode='grey', in_slot=0,
                    goal_low=0, goal_high=0,
                    obs_keys=['x', 'goal', 'img', 'goal_img', 'a']),
        max_episode_steps=1000,
    )
    register(
        id="Peg2DFreeSampleRGB-v0",
        entry_point=Peg2DEnv,
        kwargs=dict(discrete=True, view_mode='rgb', free=True,
                    obs_keys=['x', 'goal', 'img', 'a']),
        max_episode_steps=1000,
    )
    register(
        id="Peg2DFreeSample-v0",
        entry_point=Peg2DEnv,
        kwargs=dict(discrete=True, view_mode='grey', free=True,
                    obs_keys=['x', 'goal', 'img', 'a']),
        max_episode_steps=1000,
    )
    register(
        id="Peg2DFree-v0",
        entry_point=Peg2DEnv,
        kwargs=dict(discrete=True, view_mode='grey', free=True,
                    obs_keys=['x', 'goal', 'img', 'goal_img', 'a']),
        max_episode_steps=1000,
    )
