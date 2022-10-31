from hydra.core.config_store import ConfigStore


def register(*args, **kwargs):
    """
        Args: name=str, node=class
    """
    cs = ConfigStore.instance()
    cs.store(*args, **kwargs)

