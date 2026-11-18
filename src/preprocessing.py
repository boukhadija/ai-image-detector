
"""shared image preprocessing for model training and inference."""

from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_MEAN = (0.5, 0.5, 0.5)
DEFAULT_STD = (0.5, 0.5, 0.5)


def build_transform(
    # determines the resolution used by the model
    image_size: int = 224,
    # select the normalization statistics
    imagenet_normalization: bool = True,
    # control whether random horizontal flipping is enabled
    training: bool = False,
):
    """build the image transform expected by a model."""

    if image_size <= 0:
        raise ValueError("image_size must be positive")

    if imagenet_normalization:
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        mean, std = DEFAULT_MEAN, DEFAULT_STD

    operations = [
        transforms.Resize((image_size, image_size)),
    ]

    if training:
        operations.append(transforms.RandomHorizontalFlip())

    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return transforms.Compose(operations)
