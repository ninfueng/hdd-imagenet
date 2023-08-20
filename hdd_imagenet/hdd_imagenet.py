import os
from io import BytesIO
from typing import Callable, Self, Tuple

import dataflow as df
import torch
from PIL import Image


def _map_fn(
    x: df.DataFlow, transform: Callable[..., torch.nn.Module]
) -> Tuple[torch.Tensor, torch.Tensor]:
    img, label = df.LMDBSerializer._deserialize_lmdb(x)
    img = Image.open(BytesIO(img.tobytes())).convert("RGB")
    img = transform(img)
    return img, label


class ImageNetLoader:
    """Dataloader. Combines a dataset and a sampler, and provides single- or
    multi-process iterators over the dataset.

    Arguments:
        mode (str, required): mode of dataset to operate in, one of ['train', 'val']
        batch_size (int): how many samples per batch to load
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 4)
        cache (int, optional): cache size to use when loading data,
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
    """

    def __init__(
        self,
        imagenet_dir: str,
        mode: str,
        transform: Callable[..., torch.nn.Module],
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 4,
        cache: int = 50_000,
        drop_last: bool = False,
    ) -> None:
        assert mode in ["train", "val"], f"`mode` must be in (train, val), Your: {mode}"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.i = 0

        lmdb_loc = os.path.join(imagenet_dir, f"ILSVRC-{mode}.lmdb")
        dataset = df.LMDBData(lmdb_loc, shuffle=False)
        if shuffle:
            dataset = df.LocallyShuffleData(dataset, cache)

        map_fn = lambda x: _map_fn(x, transform)
        dataset = df.MultiThreadMapData(
            dataset, num_thread=num_workers, map_func=map_fn
        )
        self.dataset = df.BatchData(dataset, batch_size, use_list=True, remainder=False)
        self.dataset.reset_state()

    def __iter__(self) -> Self:
        self.i = 0
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.i == self.dataset.size():
            raise StopIteration("`ImagenetLoader` iteration is DONE.")

        x, y = next(iter(self.dataset))
        x, y = torch.stack(x), torch.tensor(y)
        self.i += 1
        return x, y

    def __len__(self) -> int:
        return self.dataset.size()
