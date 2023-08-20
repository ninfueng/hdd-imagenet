import argparse
import os

import numpy as np
from dataflow.dataflow import LMDBSerializer, MultiProcessRunnerZMQ, dataset


class BinaryILSVRC12(dataset.ILSVRC12Files):
    def get_data(self):
        sup = super().__iter__()
        for fname, label in sup:
            with open(fname, "rb") as f:
                jpeg = f.read()
            jpeg = np.asarray(bytearray(jpeg), dtype=np.uint8)
            yield (jpeg, label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Preprocess ImageNet into an LMDB file.")
    parser.add_argument("--imagenet-dir", type=str, default="~/datasets/imagenet")
    parser.add_argument("--database-dir", type=str, default="./database")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    args = parser.parse_args()

    lmdb_path = args.database_dir
    os.makedirs(lmdb_path, exist_ok=True)
    db_filename = f"ILSVRC-{args.split}.lmdb"
    db_loc = os.path.join(lmdb_path, db_filename)

    os.environ["TENSORPACK_DATASET"] = os.path.join(lmdb_path, "tensorpack_data")
    os.makedirs(os.environ["TENSORPACK_DATASET"], exist_ok=True)

    print(f"Processing from {args.imagenet_dir} {args.split} -> {db_loc}.")
    ds0 = BinaryILSVRC12(args.imagenet_dir, args.split)
    ds1 = MultiProcessRunnerZMQ(ds0, num_proc=1)
    LMDBSerializer.save(ds1, db_loc)
