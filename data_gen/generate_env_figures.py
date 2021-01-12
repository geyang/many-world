"""
Generate figures for each environment, using the default camera view
"""
envs = [
    "PointMass-v0",
    # "Reacher-v2",
]


def render(env_name):
    from many_world import IS_PATCHED
    assert IS_PATCHED, "need the patched gym"
    import gym, numpy as np
    from tqdm import tqdm
    from ml_logger import logger

    logger.prefix = env_name

    env = gym.make(env_name)
    env.reset()

    frames = []

    y = 0
    args = [dict(x=x, y=y, filename=f"images/{x:0.3f},{y:0.3f}.png") for x in np.linspace(-0.25, 0.25, 128)]
    logger.log_data(args, 'index.pkl')

    for p in tqdm(args):
        x, y, filename = p['x'], p['y'], p['filename']
        env.set_state(np.array([x, y, 0, 0]), np.array([0, 0, 0, 0]))
        # env.do_simulation([0, 0], 1) # PointMass does not need this.
        image = env.render('grey', width=20, height=20)
        frames.append(image)
        logger.log_image(image, filename)

    print('saving video')
    logger.log_video(frames, f"{env_name}.mp4")
    print('done')


if __name__ == "__main__":
    import os
    from ml_logger import logger
    logger.log_params(some_namespace=dict(layer=10, learning_rate=0.0001))
    exit()

    # logger.configure(log_directory="/tmp/learning-to-imitate", prefix="envs")
    logger.configure(log_directory=os.path.abspath("../datasets"), prefix="")
    # logger.configure(log_directory="http://54.71.92.65:8081", prefix="debug/many_world/")

    [render(env) for env in envs]
