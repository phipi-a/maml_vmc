from maml_vmc.lijax.utils import SelfStateClass


class DataSampler(SelfStateClass):
    def sample_data(self, *args, **kwargs):
        raise NotImplementedError

    def sample_system(self, *args, **kwargs):
        raise NotImplementedError

    def sample_validation_system_batch(self, num_systems: int):
        raise NotImplementedError

    def potential_fn(self, *args, **kwargs):
        raise NotImplementedError

    def get_fake_datapoint(self):
        raise NotImplementedError
