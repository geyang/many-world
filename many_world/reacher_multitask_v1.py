import numpy as np
from gym.envs import register
from . import mujoco_env


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


class ReacherMultiTaskEnv(mujoco_env.MujocoEnv):
    """
    Reacher with Multiple Targets, only one of them is real. The goal and the index
    are both shuffled when sample_goals is called.

    Note: We are not modifying the Mujoco xml, so only one goal (real) is shown
    in the rendering.
    """

    def __init__(self, k_goals=4, reward_type=None):
        """
        The multi-task Reacher environment for our experiment.

        :param virtual_distractors: bool
                                    If true, uses built-in xml definition with only 1 goal. Else use
                                    our custom xml definition, which also limites the shoulder joint angle.
        :param k_goals: int need to be in [2, 3, 4]
        :param obs_mode: oneOf("_get_delta", "pos")
                            "_get_delta" returns the distance vector between the fingertip and the goal.
                            "pos" returns the fingertip location (x, y) and the goal side-by-side. This tend to
                                have slightly worse performance.
        """
        self.reward_type = reward_type
        self.controls = Controls(k_goals=k_goals)
        assert k_goals in (1, 2, 3, 4), f"1 - 4 goals are allowed. {k_goals} is not in range."
        import os
        xml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "assets/reacher-single-task.xml" if k_goals == 1 else
                                f"assets/reacher-multitask-{k_goals}.xml")

        mujoco_env.MujocoEnv.__init__(self, xml_path, 2, set_spaces=True)
        # utils.EzPickle.__init__(self)

    @property
    def k(self):
        return self.controls.k

    def step(self, a):
        vec = self._get_delta()
        dist = np.linalg.norm(vec)
        ctrl = np.square(a).sum()
        if self.reward_type == "sparse":
            reward = float(dist < 0.02)
        else:
            reward = - dist - ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        return ob, reward, False, dict(dist=dist, success=float(dist < 0.02))

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[1] = 0
        self.viewer.cam.lookat[2] = -0.05
        self.viewer.cam.distance = .6
        self.viewer.cam.elevation = -55
        self.viewer.cam.azimuth = 90

    def reset_model(self):
        k = self.controls.k
        self.controls.sample_goal()
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        # this sets the target body position.
        qpos[-2 * self.controls.k: None if k == 1 else -2 * (k - 1)] = self.controls.true_goal
        if k > 1:
            qpos[-2 * (k - 1):] = np.concatenate(
                [g for i, g in enumerate(self.controls.goals) if self.controls.index != i])

        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2 * k:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_delta(self):
        *delta, _ = self.get_body_com("fingertip") - self.get_body_com("target")
        return delta

    def _get_obs(self):
        """
        The gym reacher environment returns the following.
            cos(theta), sin(theta), goal, theta_dot, delta
        Here instead, we return
            cos(theta), sin(theta), theta_dot, goal_posts (target + distractor), delta (with fingertip)
        We don't ever return the position of the finger tip.

        Note that the goal index is shuffled depending on the task index.

        :return: cos(theta), sin(theta), theta_dot, goal_posts (target,  *distractors), delta (with - fingertip)
        """
        theta = self.sim.data.qpos.flat[:2]
        theta_dot = self.sim.data.qvel.flat[:2]
        base = [np.cos(theta), np.sin(theta), theta_dot, ]
        t, *_ = [self.get_body_com("target" if k == 0 else f"distractor_{k}")[:2]
                 for k in range(0, self.k)]
        tip = self.get_body_com("fingertip")[:2]
        _.insert(self.controls.index, t)
        poses = np.array(_)
        deltas = poses - [tip] * self.k
        return np.concatenate([*base, *poses, *deltas])

    def sample_task(self, index=None):
        return self.controls.sample_task(index=index)

    def get_goal_index(self):
        return self.controls.index

    def get_true_goal(self):
        return self.controls.true_goal


# note: kwargs are not passed in to the constructor when entry_point is a function.
"""Multitask reacher environment with no distractors, return 1 goal vectors plus 1 goal positions."""
register(
    id="ReacherSingleTask-v0",
    entry_point=ReacherMultiTaskEnv,
    kwargs=dict(k_goals=1),
    max_episode_steps=50,
    reward_threshold=-3.75,
)
"""Multitask reacher environment with no distractors, return 1 goal vectors plus 1 goal positions."""
register(
    id="ReacherSingleTaskSparse-v0",
    entry_point=ReacherMultiTaskEnv,
    kwargs=dict(k_goals=1, reward_type="sparse"),
    max_episode_steps=50,
    reward_threshold=-3.75,
)
"""Multitask reacher environment with 1 distractor, return 2 goal vectors plus 2 goal positions."""
register(
    id="ReacherMultitaskSimple-v0",
    entry_point=ReacherMultiTaskEnv,
    kwargs=dict(k_goals=2),
    max_episode_steps=50,
    reward_threshold=-3.75,
)
"""Multitask reacher environment with 1 distractor, return 2 goal vectors plus 2 goal positions."""
register(
    id="ReacherMultitaskSimpleSparse-v0",
    entry_point=ReacherMultiTaskEnv,
    kwargs=dict(k_goals=2, reward_type="sparse"),
    max_episode_steps=50,
    reward_threshold=-3.75,
)
"""Multitask reacher environment with distractors, return 4 goal vectors plus 4 goal positions."""
register(
    id="ReacherMultitask-v0",
    entry_point=ReacherMultiTaskEnv,
    kwargs=dict(k_goals=4),
    max_episode_steps=50,
    reward_threshold=-3.75,
)
"""Multitask reacher environment with distractors, return 4 goal vectors plus 4 goal positions."""
register(
    id="ReacherMultitaskSparse-v0",
    entry_point=ReacherMultiTaskEnv,
    kwargs=dict(k_goals=4, reward_type="sparse"),
    max_episode_steps=50,
    reward_threshold=-3.75,
)
