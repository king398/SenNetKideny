import segmentation_models_pytorch as smp


def return_model(model_name: str, in_channels: int, classes: int):
    model = smp.Unet(
        encoder_name=model_name,
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=classes,



    )
    return model



