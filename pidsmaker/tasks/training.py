from pidsmaker.detection.training_methods import (
    ml_training,
    training_loop,
)


def main(cfg):
    method = cfg.training.used_method.strip()
    if method == "default":
        return training_loop.main(cfg)
    elif method == "ml":
        return ml_training.main(cfg)
    else:
        raise ValueError(f"Invalid training method {method}")
