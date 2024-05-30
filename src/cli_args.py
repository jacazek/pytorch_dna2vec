from dataclasses import dataclass
from typing import List
import argparse
import subprocess
import os


@dataclass
class TrainArguments:
    epochs: int
    number_train_workers: int
    number_validate_workers: int
    number_devices: int
    batch_size: int
    kmer_size: int
    stride: int
    window_size: int
    embedding_dimensions: int
    learning_rate: float
    lr_gamma: float
    number_train_files_per_epoch: int
    number_validate_files_per_epoch: int
    tags: List[str]

    # keep this key last
    command: str


def get_arguments() -> TrainArguments:
    parser = argparse.ArgumentParser(description="Train dna2vec model.")

    parser.add_argument("--epochs", type=int, default=1, help="The number of epochs to train")
    parser.add_argument("--number_train_workers", type=int, default=1,
                        help="The number of worker processes to provide training data")
    parser.add_argument("--number_validate_workers", type=int, default=1,
                        help="The number of worker processes to provide validation data")
    parser.add_argument("--number_devices", type=int, default=1,
                        help="The number of devices on which to train the model")
    parser.add_argument("--batch_size", type=int, default=20480,
                        help="The size of each batch for training and validation")
    parser.add_argument("--kmer_size", type=int, default=7,
                        help="The size of each kmer")
    parser.add_argument("--stride", type=int, default=3,
                        help="The stride between kmers")
    parser.add_argument("--window_size", type=int, default=7,
                        help="The size of the window for training embedding")
    parser.add_argument("--embedding_dimensions", type=int, default=512,
                        help="The number of dimensions for each embedding")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="The learning rate for optimizer")
    parser.add_argument("--lr_gamma", type=float, default=0.5,
                        help="The learning rate gamma for the scheduler")
    parser.add_argument("--number_train_files_per_epoch", type=int, default=1,
                        help="The number of fasta files to train per device per epoch")
    parser.add_argument("--number_validate_files_per_epoch", type=int, default=1,
                        help="The number of fasta files to validate per device per epoch")
    parser.add_argument("--tags", action="append", help="Additional key:value tags to capture with the training run")

    args = parser.parse_args()
    train_arguments = TrainArguments(**vars(args), command=str(subprocess.run(["ps", "-p", f"{os.getpid()}", "-o", "args", "--no-headers"], capture_output=True, text=True).stdout))
    return train_arguments
