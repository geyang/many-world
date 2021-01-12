import pytest


@pytest.fixture
def setup():
    from urllib.parse import quote
    from ml_logger import logger
    from many_world import IS_PATCHED
    assert IS_PATCHED, "need to be patched"
    import gym
    gym.logger.set_level(40)

    logger.configure(log_directory="/", prefix="tmp/env-debug")
    print(f'to see experiment, go to: /{quote(logger.prefix)}')


def test_render(setup):
    import gym
    from termcolor import cprint

    env = gym.make('PointMass-v0')

    f = env.render('rgb', width=10, height=10)
    assert f is not None, "frame need to be an image"
    cprint('RGB mode works', 'green')

    f = env.render('depth', width=10, height=10)
    assert f is not None, "frame need to be an image"
    cprint('depth mode works', 'green')

    f = env.render('rgbd', width=10, height=10)
    assert f[0] is not None, "frame need to be an image"
    assert f[1] is not None, "depth need to be an image"
    cprint('RGBD mode works', 'green')


def test_task_spec(setup):
    import gym
    env = gym.make('PointMass-v0')


def test_point_mass(setup):
    from time import sleep
    import gym
    env = gym.make('PointMass-v0')
    env.reset()

    import matplotlib.pyplot as plt
    for i in range(10):
        env.step([10, 10])
        # env.render()
        image = env.render('rgb_array', width=28, height=28)
        plt.imshow(image)
        plt.show()
        sleep(0.01)
