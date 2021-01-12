import numpy as np
from gym.envs import register
from many_world import mujoco_env


class PendulumEnv(mujoco_env.MujocoEnv):
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
        assert k_goals in (1, 2, 3, 4), f"1 - 4 goals are allowed. {k_goals} is not in range."
        import os
        xml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/pendulum.xml")

        mujoco_env.MujocoEnv.__init__(self, xml_path, 2, set_spaces=True)
        # utils.EzPickle.__init__(self)

    def step(self, a):
        """
        :param a:
        :return: {
            ob: Observation Space: <theta, theta_dot>,
            reward: ctrl + max(theta - pi)
        """
        theta = self.sim.data.qpos.flat.copy()
        delta = (theta - np.pi / 2) if theta > - 0.5 * np.pi else 1.5 * np.pi + theta
        ctrl = np.square(a).sum()

        if self.reward_type == "sparse":
            reward = float(np.abs(delta) < 0.01 * np.pi)
        else:
            reward = - np.abs(delta) - ctrl

        self.do_simulation(a, self.frame_skip)
        # Note: DeepMind control suite handles this as such: return the x_dot **after** the simulation
        # ref: https://arxiv.org/abs/1801.00690
        theta_dot = self.sim.data.qvel.flat.copy()
        obs = np.concatenate([np.cos(theta), np.sin(theta), theta_dot])
        return obs, reward, False, dict(delta=delta, success=float(np.abs(delta) < 0.01 * np.pi))

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[1] = 0
        self.viewer.cam.lookat[2] = 0
        self.viewer.cam.distance = .3
        self.viewer.cam.elevation = 0
        self.viewer.cam.azimuth = 90

    def reset_model(self):
        qpos = self.np_random.uniform(low=-np.pi, high=np.pi, size=(1,))
        qvel = self.np_random.uniform(low=-1, high=1, size=(1,))
        self.set_state(qpos, qvel)

        theta = self.sim.data.qpos.flat
        theta_dot = self.sim.data.qvel.flat.copy()
        # Note: DeepMind control suite handles this as such: return the x_dot **after** the simulation
        return np.concatenate([np.cos(theta), np.sin(theta), theta_dot])

    def set_gravity(self, gravity):
        self.sim.o

if __name__ == "__main__":
    import gym

    env = gym.make('InvertedPendulum-v0')
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
    register(
        id="InvertedPendulum-v0",
        entry_point=PendulumEnv,
        kwargs=dict(),
        max_episode_steps=50,
        reward_threshold=-3.75,
    )

if __name__ == "__main__":
    # scratch space for the trigonometry calculation:
    # make sure that the delta_theta is calculated correctly.
    rs = [np.pi / 2, 0, -np.pi / 2, -np.pi, np.pi, np.pi * 3 / 4]
    ds = [0, -np.pi / 2, -np.pi, np.pi / 2, np.pi / 2, np.pi / 4]
    for r, d in zip(rs, ds):
        theta = np.arccos(np.cos(r)) * np.sign(np.arcsin(np.sin(r)))
        _theta = (theta - np.pi / 2) if theta > - 0.5 * np.pi else 1.5 * np.pi + theta
        print(r, theta, _theta, d)
