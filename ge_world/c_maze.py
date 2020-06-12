import numpy as np
from gym import spaces

from ge_world.mujoco_env import MujocoEnv
from ge_world.base_envs import MazeCamEnv


def good_goal(goal):
    """
    filter for a good goal (state) in the maze.

    :param goal:
    :return: bool, True if goal position is good
    """
    return not (goal[0] < 0.13 and -0.11 < goal[1] and goal[1] < 0.11)


def good_state(state):
    """
    filter for a good goal (state) in the maze.

    :param state:
    :return: bool, True if goal position is good
    """
    return not (state[0] < 0.11 and -0.09 < state[1] and state[1] < 0.09)


class CMazeEnv(MujocoEnv, MazeCamEnv):
    """
    2D Point Mass Environment. Uses torque control.
    """
    achieved_key = 'x'
    desired_key = 'goal'
    is_good_goal = lambda self, _: good_goal(_)
    is_good_state = lambda self, _: good_state(_)

    def __init__(self, frame_skip=10, obs_keys=(achieved_key, desired_key),
                 obj_low=-0.24, obj_high=0.24, goal_low=-0.24, goal_high=0.24,
                 act_scale=0.5, discrete=False, id_less=False, done_on_goal=False,
                 cam_id=-1, **kwargs,
                 ):
        """
        :param frame_skip:
        :param discrete:
        :param id_less:
        :param done_on_goal: False, bool. flag for setting done to True when reaching the goal
        """
        # self.controls = Controls(k_goals=1)
        self.obs_keys = obs_keys
        self.discrete = discrete
        self.done_on_goal = done_on_goal

        if self.discrete:
            actions = [-act_scale, 0, act_scale]
            if id_less:
                self.a_dict = [(a, b) for a in actions for b in actions if not (a == 0 and b == 0)]
                self.action_space = spaces.Discrete(8)
            else:
                self.a_dict = [(a, b) for a in actions for b in actions]
                self.action_space = spaces.Discrete(9)

        # call super init after initializing the variables.
        import os
        xml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"assets/c-maze.xml")
        MujocoEnv.__init__(self, xml_path, frame_skip=frame_skip, set_action_space=True, set_observation_space=False,
                           **kwargs)
        MazeCamEnv.__init__(self, cam_id=cam_id, **kwargs)

        # dictionary observation space.
        _ = dict()
        if 'x' in obs_keys:
            _['x'] = spaces.Box(low=np.array([-0.3, -0.3]), high=np.array([0.3, 0.3]))
        if 'goal' in obs_keys:
            # todo: double check for agreement with actual goal distribution
            _['goal'] = spaces.Box(low=np.array([goal_low, goal_low]), high=np.array([goal_high, goal_high]))
        if 'img' in obs_keys:
            _['img'] = spaces.Box(
                low=np.zeros((1, self.width, self.height)), high=np.ones((1, self.width, self.height)))
        if 'goal_img' in obs_keys:
            _['goal_img'] = spaces.Box(
                low=np.zeros((1, self.width, self.height)), high=np.ones((1, self.width, self.height)))
        self.obj_low = obj_low
        self.obj_high = obj_high
        self.goal_low = goal_low
        self.goal_high = goal_high
        self.observation_space = spaces.Dict(_)

    def compute_reward(self, achieved, desired, *_):
        success = np.linalg.norm(achieved - desired, axis=-1) < 0.02
        return (success - 1).astype(float)

    reach_counts = 0

    def step(self, a):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        qvel[:2] = 0
        self.set_state(qpos, qvel)
        if self.discrete:
            a = self.a_dict[int(a)]
        vec = self._get_delta()
        dist = np.linalg.norm(vec)
        for i in range(self.frame_skip):
            self.do_simulation(a, 1)
            # note: return observation *after* simulation. This is how DeepMind Lab does it.
            ob = self._get_obs("x", "goal")
            reward = self.compute_reward(ob['x'], ob['goal'])
            if reward == 0:
                done = True
                return self._get_obs(), reward, done, dict(dist=dist, success=float(dist < 0.02))
            else:
                done = False

        # todo: I changed this to 0.4 b/c discrete action jitters around. Remember to fix this. --Ge
        return self._get_obs(), reward, done, dict(dist=dist, success=float(dist < 0.02))

    def _get_goal(self):
        # return self.np_random.uniform(low=self.goal_low, high=self.goal_high, size=2)
        while True:
            goals = self.np_random.uniform(low=self.goal_low, high=self.goal_high, size=(10, 2))
            for goal in goals:
                if good_goal(goal):
                    return goal

    def _get_state(self):
        while True:
            states = self.np_random.uniform(low=self.obj_low, high=self.obj_high, size=(10, 2))
            for goal in states:
                if good_state(goal):
                    return goal

    def reset_model(self, x=None, goal=None):
        self.reach_counts = 0
        if x is None:
            x = self._get_state()
        if goal is None:
            goal = self._get_goal()

        self.update_goal(goal)

        pos = np.concatenate([x, goal])
        self.set_state(pos, np.zeros_like(pos))
        return self._get_obs()

    goal_img = None

    def update_goal(self, goal=None):
        qpos = self.sim.data.qpos.flat.copy()
        if goal is not None:
            qpos[:2] = goal.copy()
        self.set_state(qpos, np.zeros_like(self.sim.data.qvel.flat))
        with self.with_color(-1):
            self.goal_img = self.render('grey', width=self.width, height=self.height
                                        ).transpose(0, 1)[None, ...]
        return self.goal_img

    def _get_delta(self):
        *delta, _ = self.get_body_com("goal") - self.get_body_com("object")
        return delta

    def _get_obs(self, *keys):
        obs = {}
        qpos = self.sim.data.qpos.flat.copy()
        keys = keys or self.obs_keys
        if 'x' in keys:
            obs['x'] = qpos[:2].copy()
        if 'img' in keys:
            with self.with_color(-1):
                obs['img'] = self.render('grey', width=self.width, height=self.height).transpose(0, 1)[None, ...]
        if 'goal' in keys:
            obs['goal'] = qpos[2:].copy()
        if 'goal_img' in keys:
            curr_qpos = qpos.copy()
            qpos[:2] = qpos[2:].copy()
            qpos[2:] = [.3, .3]  # move goal out of frame
            self.set_state(qpos, self.sim.data.qvel)
            with self.with_color(-1):
                obs['goal_img'] = self.render('grey', width=self.width, height=self.height).transpose(0, 1)[
                    None, ...]
            self.set_state(curr_qpos, self.sim.data.qvel)
        return obs

    # def sample_task(self, index=None):
    #     return self.controls.sample_task(index=index)

    # def get_goal_index(self):
    #     return self.controls.index

    # def get_true_goal(self):
    #     return self.controls.true_goal


from gym.envs import register

if __name__ == "__main__":
    import gym

    env = gym.make('CMazeDiscrete-v0')
    env.reset()
    frame = env.render('rgb', width=200, height=200)
    from os.path import basename
    from ml_logger import logger

    logger.log_image(frame, f"../figures/{basename(__file__)}:{env.spec.id}.png")

    # to show thy human.
    from PIL import Image

    im = Image.fromarray(frame)
    im.show()
else:
    # note: kwargs are not passed in to the constructor when entry_point is a function.
    register(
        id="CMaze-v0",
        entry_point=CMazeEnv,
        kwargs=dict(obs_keys=('x', 'img', 'goal', 'goal_img')),
        # max_episode_steps=1000,
    )
    # register(
    #     id="CMazeDiscrete-v0",
    #     entry_point=CMazeEnv,
    #     kwargs=dict(discrete=True),
    #     # max_episode_steps=1000,
    # )
    # register(
    #     id="CMazeDiscreteIdLess-v0",
    #     entry_point=CMazeEnv,
    #     kwargs=dict(discrete=True, id_less=True, act_scale=0.5),
    #     # max_episode_steps=1000,
    # )
    # register(
    #     id="CMazeDiscreteImgIdLess-v0",
    #     entry_point=CMazeEnv,
    #     kwargs=dict(discrete=True, obs_keys=('x', 'img', 'goal', 'goal_img'), id_less=True),
    #     # max_episode_steps=1000,
    # )
    # register(
    #     id="CMazeDiscreteFixGImgIdLess-v0",
    #     entry_point=CMazeEnv,
    #     kwargs=dict(discrete=True, obs_keys=('x', 'img', 'goal', 'goal_img'), goal_low=0.18, goal_high=0.181,
    #                 id_less=True),
    #     # max_episode_steps=1000,
    # )
    # register(
    #     id="CMazeDiscreteIdLessTerm-v0",
    #     entry_point=CMazeEnv,
    #     kwargs=dict(discrete=True, id_less=True, done_on_goal=True),
    #     # max_episode_steps=1000,
    # )
