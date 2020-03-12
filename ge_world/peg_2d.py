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
                 act_scale=1, discrete=False,
                 free=False,  # whether to move the goal out of the way
                 view_mode="grey",
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

    reach_counts = 0

    def step(self, a):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        qvel[-1:] = 0
        if self.free:  # sent the slot out of view
            qpos[-1:] = 1
        self.set_state(qpos, qvel)
        if self.discrete:
            a = [self.a_dict[int(a_i)] for a_i in a]
        a = np.array([*a, *self.goal])
        self.do_simulation(a, self.frame_skip)
        # note: return observation *after* simulation. This is how DeepMind Lab does it.
        ob = self._get_obs()

        dist = np.linalg.norm(self._get_goal_state() - qpos[:-1], ord=2)
        done = dist < 0.04
        reward = float(done) - 1

        # offer raw action to agent
        if 'a' in self.obs_keys:
            ob['a'] = a[:-1].copy()

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
        # return np.array([0.2])
        while True:
            goals = self.np_random.uniform(low=self.goal_low, high=self.goal_high, size=(4, 1))
            for goal in goals:
                if good_goal(goal):
                    return goal

    def _get_state(self):
        if self.np_random.rand() < 0.1:
            return self._get_goal_state(x=self.np_random.rand() * - 0.01)

        while True:
            states = self.np_random.uniform(low=-1.5, high=1.5, size=(4, 3))
            _ = self.np_random.uniform(0, np.pi / 2, size=4)
            states[:, 1] = - states[:, 0] - np.sign(states[:, 0]) * _
            states[:, 2] = np.sign(states[:, 0]) * _ + self.np_random.uniform(0, np.pi / 4, size=4)
            for state in states:
                if good_state(state):
                    return state

    def _get_goal_state(self, x=0, goal=None):
        qpos = np.zeros(3)
        if goal is None:
            goal = self.goal

        peg_x_y = [x, goal / 10]

        base = (0.03 + peg_x_y[0])
        hypo = np.linalg.norm([base, peg_x_y[1]], ord=2)
        a0 = np.arctan(peg_x_y[1] / base)
        a1 = np.sign(self.np_random.rand() - 0.5) * np.arccos(hypo / 0.04)
        qpos[0] = a0 + a1
        qpos[1] = - 2 * a1
        qpos[2] = 0 - qpos[0] - qpos[1]
        return qpos

    def reset_model(self, x=None, goal=None):
        self.reach_counts = 0
        if goal is None:
            goal = self._get_goal()
        self.goal = goal
        if x is None:
            x = self._get_state()

        # always move out of the way for goal_image
        qpos = np.concatenate([self._get_goal_state(), [1]])
        # qpos[-1:] = goal  # show the slot.

        self.set_state(qpos, self.sim.data.qvel)

        img = self.render(self.view_mode, width=self.width, height=self.height)
        if self.view_mode == "grey":
            img = img[..., None]
        self.goal_img = img.transpose(2, 0, 1) / 255

        # now reset.
        self.sim.data.qpos[:] = np.concatenate([x, [1] if self.free else goal])
        self.sim.data.qvel[:] = 0  # no velocity
        return self._get_obs()

    def _get_delta(self):
        *delta, _ = self.get_body_com("goal") - self.get_body_com("object")
        return delta

    def _get_obs(self):
        obs = {}
        qpos = self.sim.data.qpos.flat.copy()
        if 'x' in self.obs_keys:
            obs['x'] = qpos[:3]
        if 'img' in self.obs_keys:
            goal = qpos[3:].copy()
            qpos[3:] = [1].copy()  # move slot out of frame
            self.set_state(qpos, self.sim.data.qvel)

            img = self.render(self.view_mode, width=self.width, height=self.height)
            if self.view_mode == "grey":
                img = img[..., None]
            obs['img'] = img.transpose(2, 0, 1) / 255
            qpos[3:] = goal
            self.set_state(qpos, self.sim.data.qvel)
        if 'goal' in self.obs_keys:
            obs['goal'] = qpos[3:].copy()
        if 'goal_img' in self.obs_keys:
            obs['goal_img'] = self.goal_img

        return obs


if __name__ == "__main__":
    import gym
    from ml_logger import logger
    from time import sleep
    from tqdm import trange

    # env = Peg2DEnv(discrete=True, id_less=False, obs_keys=["x", 'img'])
    # env_id = "Peg2DImgDiscrete-v0"
    env_id = "Peg2DFreeImgDiscrete-v0"
    env = gym.make(env_id)
    seed = 100
    env.seed(seed)

    frames = []

    # env.render('human', width=200, height=200)

    for i in trange(100):
        env.reset()
        for step in range(4):
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
    logger.log_image(stack.min(0), key=f"../figures/{seed}/{env_id}_spread.png")

    print('done rendering!')

else:
    from gym.envs import register

    # note: kwargs are not passed in to the constructor when entry_point is a function.
    register(
        id="Peg2DImgDiscrete-v0",
        entry_point=Peg2DEnv,
        kwargs=dict(discrete=True, view_mode='rgb',
                    obs_keys=['x', 'goal', 'img', 'goal_img', 'a']),
        max_episode_steps=1000,
    )
    register(
        id="Peg2DFreeImgDiscrete-v0",
        entry_point=Peg2DEnv,
        kwargs=dict(discrete=True, view_mode='rgb', free=True,
                    obs_keys=['x', 'goal', 'img', 'goal_img', 'a']),
        max_episode_steps=1000,
    )
