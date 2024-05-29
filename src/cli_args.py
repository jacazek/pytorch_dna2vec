from dataclasses import dataclass
import argparse


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

    args = parser.parse_args()
    return TrainArguments(**vars(args))
