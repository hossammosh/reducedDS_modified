from lib.train.admin.environment import env_settings


class Settings:
    """ Training settings, e.g. the paths to datasets and networks."""
    def __init__(self):
        self.set_default()

    def set_default(self):
        self.env = env_settings()
        self.use_gpu = True
        self.selected_sampling = False
        self.sample_per_epoch = 0
        self.top_sample_ratio = 0.0  # Default value
        self.top_selected_samples = 0  # Default value

