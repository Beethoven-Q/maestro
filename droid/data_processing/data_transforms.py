import cv2


class ImageTransformer:
    """Drops the alpha channel and/or converts BGR camera images to RGB, in place."""

    def __init__(self, remove_alpha=False, bgr_to_rgb=False, image_path="observation/camera/image"):
        self.image_path = image_path.split("/")
        self.remove_alpha = remove_alpha
        self.bgr_to_rgb = bgr_to_rgb

    def transform(self, data):
        if self.remove_alpha:
            data = data[:, :, :3]
        if self.bgr_to_rgb:
            data = cv2.cvtColor(data, cv2.COLOR_BGRA2RGBA if data.shape[-1] == 4 else cv2.COLOR_BGR2RGB)
        return data

    def forward(self, timestep):
        if not (self.remove_alpha or self.bgr_to_rgb):
            return timestep
        obs = timestep
        for key in self.image_path:
            obs = obs[key]
        for cam_type in obs:
            obs[cam_type] = [self.transform(image) for image in obs[cam_type]]
        return timestep
