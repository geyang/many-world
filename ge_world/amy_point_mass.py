import numpy as np
from gym import spaces

from . import mujoco_env


# from gym.envs.mujoco import mujoco_env


class Controls:
    def __init__(self, k_goals, seed=None):
        """
        The task index is always initialized to be 0. Which means that unless you sample task, this
        is the same as the single-task baseline. So if you want to use this in a multi-task setting,
        make sure that you resample goals each time after resetting the environment.

        :param k_goals:
        :param seed:
        """
        self.k = k_goals
        # deterministically generate the goals so that we don't have to pass in the positions.
        self.rng = np.random.RandomState(seed)
        self.sample_goal()
        self.index = 0  # this is always initialized to be 0.

    def seed(self, seed):
        self.rng.seed(seed)

    @property
    def _goals(self):
        while True:
            # chances decrease exponentially. Better to generate by pair.
            goals = self.rng.uniform(low=-.2, high=.2, size=(self.k, 2))
            if (np.linalg.norm(goals, axis=-1) < .2).all():
                return goals

    def __repr__(self):
        return f"Reacher Control: index({self.index}) true goal({self.true_goal}) all goals({self.goals})"

    @property
    def true_goal(self):
        return self.goals[self.index]

    def sample_goal(self, goals=None):
        if goals is None:
            self.goals = self._goals
        else:
            self.goals = goals
        return self.goals

    def sample_task(self, index=None):
        if index is None:
            self.index = np.random.randint(0, self.k)
        else:
            self.index = index
            assert index < self.k, f"index need to be less than the number of tasks {self.k}."
        return self.index


class PointMassEnv(mujoco_env.MujocoEnv):
    """
    2D Point Mass Environment. Uses torque control.
    """

    def __init__(self, frame_skip=10, discrete=False):
        """
        The multi-task Reacher environment for our experiment.

        :type virtual_distractors: bool
                                    If true, uses built-in xml definition with only 1 goal. Else use
                                    our custom xml definition, which also limites the shoulder joint angle.
        :param k_goals: int need to be in [2, 3, 4]
        :param obs_mode: oneOf("_get_delta", "pos")
                            "_get_delta" returns the distance vector between the fingertip and the goal.
                            "pos" returns the fingertip location (x, y) and the goal side-by-side. This tend to
                                have slightly worse performance.
        """
        self.controls = Controls(k_goals=1)
        self.discrete = discrete
        if self.discrete:
            actions = [-1, 0, 1]
            self.a_dict = [ (a, b) for a in actions for b in actions]

        # call super init after initializing the variables.
        import os
        xml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"assets/point-mass.xml")

        mujoco_env.MujocoEnv.__init__(self, xml_path, frame_skip=frame_skip, set_spaces=True, discrete=discrete)
        # utils.EzPickle.__init__(self)

        # note: Experimental, hard-coded
        self.observation_space = spaces.Box(low=np.array([-0.3, -0.3]),
                                            high=np.array([0.3, 0.3]))

    @property
    def k(self):
        return self.controls.k

    def step(self, a):
        if self.discrete:
            a = self.a_dict[int(a[0])]
        vec = self._get_delta()
        dist = np.linalg.norm(vec)
        ctrl = np.square(a).sum()
        reward = - dist - ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        return ob, reward, False, dict(dist=dist, success=float(dist < 0.02))

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[1] = 0
        # side-view
        # self.viewer.cam.lookat[2] = -0.1
        # self.viewer.cam.distance = .7
        # self.viewer.cam.elevation = -55
        self.viewer.cam.azimuth = 90

        # ortho-view
        self.viewer.cam.lookat[2] = 0
        self.viewer.cam.distance = .67
        self.viewer.cam.elevation = -90

    def reset_model(self):
        # todo: double check this.
        self.controls.sample_goal()
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        # this sets the target body position.
        qpos[-2:] = self.controls.true_goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_delta(self):
        *delta, _ = self.get_body_com("goal") - self.get_body_com("object")
        return delta

    def _get_obs(self):
        x = self.sim.data.qpos.flat[:2]
        # goal_pos = self.get_body_com("goal")[:2]
        # x_dot = self.sim.data.qvel.flat[:2]
        # base = [x, x_dot, ]
        # return np.concatenate([*base, goal_pos, self._get_delta()])
        return x

    def sample_task(self, index=None):
        return self.controls.sample_task(index=index)

    def get_goal_index(self):
        return self.controls.index

    def get_true_goal(self):
        return self.controls.true_goal


from gym.envs import register

# note: kwargs are not passed in to the constructor when entry_point is a function.
register(
    id="PointMassDiscrete-v0",
    # entry_point="envs.point_mass:PointMassEnv",
    entry_point=PointMassEnv,
    kwargs={'discrete': True},
    max_episode_steps=10,
    reward_threshold=-3.75,
)
