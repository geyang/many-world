import abc


class MazeCamEnv(metaclass=abc.ABCMeta):
    def __init__(self, *args, cam_id=-1, width=100, height=100, **kwargs):
        self.cam_id = cam_id
        self.width = width
        self.height = height

    def viewer_setup(self):
        """The camera id here is not real."""
        cam_id = self.cam_id
        camera = self.viewer.cam

        camera.trackbodyid = 0
        camera.lookat[0] = 0
        camera.lookat[1] = 0

        if cam_id == -1:
            # ortho-view
            camera.lookat[2] = 0
            camera.distance = .74
            camera.elevation = -90
            camera.azimuth = 90
        else:
            # side-view
            camera.lookat[2] = -0.1
            camera.distance = .7
            camera.elevation = -55
            camera.azimuth = 90
