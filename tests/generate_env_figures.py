"""
Generate figures for each environment, using the default camera view
"""
envs = [
    "PointMass-v0",
    "ReacherSingleTask-v1",
]


def render(env_name):
    from ge_world import IS_PATCHED
    assert IS_PATCHED, "need the patched gym"
    import gym, numpy as np
    from ml_logger import logger
    env = gym.make(env_name)
    env.reset()

    from tqdm import tqdm

    images = []
    for x in tqdm(np.linspace(-0.25, 0.25, 11)):
        for y in np.linspace(-0.25, 0.25, 11):
            env.unwrapped.set_state(np.array([x, y, 0, 0]), np.array([0, 0, 0, 0]))
            # env.do_simulation([0, 0], 1)
            image = env.render('rgb', width=28, height=28)
            logger.log_image(image, f"{env_name}/images/{x:.2f},{y:.2f}.png")
            images.append(image)
    logger.log_video(images, f'{env_name}/samples.mp4')


if __name__ == "__main__":
    import os
    from ml_logger import logger

    # logger.configure(log_directory="/tmp/learning-to-imitate", prefix="envs")
    logger.configure(log_directory=os.path.abspath("../figures"), prefix="")
    # logger.configure(log_directory="http://54.71.92.65:8081", prefix="debug/ge_world/")

    [render(env) for env in envs]
