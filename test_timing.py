import time

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from hdd_imagenet import ImageNetLoader

if __name__ == "__main__":
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    seq_loader = ImageNetLoader(
        "./database",
        mode="val",
        transform=train_transform,
        batch_size=256,
        num_workers=8,
        shuffle=True,
        drop_last=False,
    )

    t0 = time.perf_counter()
    for x, y in tqdm(seq_loader):
        print(x.shape, y.shape)
    diff = time.perf_counter() - t0
    print(f"Database Run Time: {diff} Seconds.")

    dataset = ImageFolder("~/datasets/imagenet/val", train_transform)
    loader = DataLoader(dataset, batch_size=256, num_workers=8, shuffle=True)

    t0 = time.perf_counter()
    for x, y in tqdm(loader):
        print(x.shape, y.shape)
    diff = time.perf_counter() - t0
    print(f"ImageFolder Run Time: {diff} Seconds.")
