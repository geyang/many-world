import pytest


@pytest.fixture
def setup():
    import gym
    gym.logger.set_level(40)


# def test_h_maze(setup):
def main():
    import gym

    env = gym.make('many_world:HMaze-v0')
    env.reset()
    obs, reward, done, info = env.step(env.action_space.sample())
    print(obs.keys())


if __name__ == '__main__':
    main()
