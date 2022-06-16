import random
import argparse

from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField

import datasets.imagelistdataset


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--input", type=str, default="./datasets/ego4d_tiny.txt")
    args.add_argument("--output", type=str, default="/checkpoint/yixinlin/ego4d_ssl/ego4d_tiny.beton")
    args.add_argument("--num-workers", type=int, default=-1)
    cfg = args.parse_args()

    fname = cfg.input
    write_path = args.output
    num_workers = args.num_workers
    fields = {"image": RGBImageField(), "meta": IntField()}

    print(f"Loading dataset from {fname}")
    ds = datasets.imagelistdataset.BasicImageDataset(fname)

    print(f"Testing by attempting to load a random image from dataset of len {len(ds)}")
    print(random.choice(ds))

    print(f"Creating dataset writer with num_workers={num_workers}")
    writer = DatasetWriter(
        write_path, fields, num_workers=num_workers
    )
    print(f"Writing ffcv dataset to {write_path}")
    writer.from_indexed_dataset(ds)
