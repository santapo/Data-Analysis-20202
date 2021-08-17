from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization

from torchvision import transforms


def get_datamodule(data_dir: str = './data',
                    batch_size: int = 32,
                    num_workers: int = 4):
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        cifar10_normalization(),
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        cifar10_normalization(),
    ])

    cifar10_dm = CIFAR10DataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        val_transforms=test_transforms,
    )

    return cifar10_dm