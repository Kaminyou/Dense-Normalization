from models.cut import ContrastiveModel
from models.cyclegan import CycleGanModel


def get_model(
    config,
    model_name="CUT",
    normalization="in",
    isTrain=True,
    parallelism=False,
    interpolate_mode='bilinear',
):
    if model_name == "CUT":
        model = ContrastiveModel(
            config,
            normalization=normalization,
            parallelism=parallelism,
            interpolate_mode=interpolate_mode,
        )
    elif model_name == "cycleGAN":
        model = CycleGanModel(
            config,
            normalization=normalization,
            parallelism=parallelism,
            interpolate_mode=interpolate_mode,
        )
    elif model_name == "LSeSim":
        print("Please use the scripts prepared in the F-LSeSim folder")
        raise NotImplementedError
    else:
        raise NotImplementedError
    return model
