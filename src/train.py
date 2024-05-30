import torch.nn
import os
import torch
from torchinfo import summary
from trainer import DDPTrainer, DDPRunner
from experiment import DDPExperiment
from multiprocessing import Queue, Process
import mlflow
import random

import torch_utils
from fasta_utils.tokenizers import KmerTokenizer
from fasta_utils.vocab import Vocab
from models.dna2vec import Dna2Vec
from fasta_pytorch_utils.data import FastaFileQueueDataset, StreamingEmbedding, PreprocessEmbedding
from fasta_pytorch_utils.data.FastaWindowQueuer import FastaWindowQueuer, FastaWindowQueueDataset, StreamingWindows, \
    FastFileIndexPair
from torch.utils.data import DataLoader
from cli_args import get_arguments, TrainArguments

script_directory = os.path.dirname(os.path.realpath(__file__))
root_directory = os.path.abspath(os.path.join(script_directory, "../"))

# mlflow.autolog()
mlflow.set_tracking_uri("http://localhost:8080")
device = torch_utils.get_device()


def save_model_summary(model, input_size, artifact_path):
    file_name = os.path.join(artifact_path, "model_summary.txt")
    with open(file_name, "w") as file:
        summary_string = str(summary(model, input_size, dtypes=[torch.long]))
        file.write(summary_string)
    mlflow.log_artifact(file_name, artifact_path=artifact_path)


def collate_fn(batch):
    """
    Collect then pad the sequences into a tensor
    :param batch: the batch
    :return: padded sequences, the original targets, and corresponding original lengths of each sequence
    """
    # sort the batch by length of sequence ascending

    # unzip the sequences from the corresponding targets
    [contexts, targets] = zip(*batch)
    return torch.tensor(contexts, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def experiment(rank, train_arguments: TrainArguments):
    artifact_name = os.path.basename(train_arguments.vocab_artifact_uri)
    artifact_path = os.path.join(train_arguments.artifact_directory, artifact_name)

    fasta_file_directory = os.path.join(root_directory, "data")
    fasta_file_extension = ".fa.gz"
    fasta_files = [f"{fasta_file_directory}/{file}" for file in os.listdir(fasta_file_directory) if
                   file.endswith(fasta_file_extension)]
    # validate_fasta_files = [f"{fasta_file_directory}/{file}" for file in os.listdir(fasta_file_directory) if
    #                file.endswith(fasta_file_extension)][2:4]
    # print(train_fasta_files)
    files_per_gpu = int(len(fasta_files) / train_arguments.number_devices)
    start = rank * files_per_gpu
    end = start + files_per_gpu
    fasta_files = fasta_files[start:end]

    train_sequence_queue = Queue()
    validate_sequence_queue = Queue()

    # loading artifacts needs to happen once before worker threads are launched
    # otherwise there may be corruption
    vocabulary = Vocab.load(artifact_path)
    # the model should not care about stuff like learning rate.  that is a training parameter
    # how do we communicate recommended optimizer settings then if the learning rate is not part of the model

    model = Dna2Vec(vocabulary, embedding_dimension=train_arguments.embedding_dimensions, device=device,
                    learning_rate=train_arguments.learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_arguments.learning_rate, fused=True)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=train_arguments.lr_gamma)
    model.lr_scheduler = lr_scheduler
    model.optimizer = optimizer

    # pull these from vocabulary metadata later
    tokenizer = KmerTokenizer(train_arguments.kmer_size, train_arguments.stride)
    # streaming_windows = StreamingWindows(vocabulary, tokenizer, train_arguments.window_size)

    streaming_embedding = StreamingEmbedding(vocabulary, tokenizer, train_arguments.window_size)
    # preprocessing_embedding = PreprocessEmbedding(vocabulary, tokenizer, window_size)
    dataset = FastaFileQueueDataset(train_sequence_queue, embedding_strategy=streaming_embedding, device=device)
    dataloader = DataLoader(dataset, train_arguments.batch_size, num_workers=train_arguments.number_train_workers,
                            prefetch_factor=10,
                            pin_memory=True, collate_fn=collate_fn)

    validate_dataset = FastaFileQueueDataset(validate_sequence_queue, embedding_strategy=streaming_embedding,
                                             device=device)
    validate_dataloader = DataLoader(validate_dataset, train_arguments.batch_size,
                                     num_workers=train_arguments.number_validate_workers,
                                     prefetch_factor=10, pin_memory=True, collate_fn=collate_fn)

    # torch.set_float32_matmul_precision('medium')
    # TODO: make this data loading more robust
    for epoch in range(train_arguments.epochs):
        train_fasta_files = fasta_files[:train_arguments.number_train_files_per_epoch]
        fasta_files = fasta_files[train_arguments.number_train_files_per_epoch:]
        print(f"rank: {rank}; queueing for training {epoch}: {train_fasta_files}")
        for fasta_file in train_fasta_files:
            # print(f"device: {rank}, train files: {fasta_files[start:end]}")
            # with FastaFileReader(fasta_file) as fasta_file_reader:
            for i in range(10):
                item = (fasta_file, i, random.random() < 0.5)
                train_sequence_queue.put(item)
                print(f"Epoch {epoch}: queued {item} for train")
                # train_sequence_queue.put((fasta_file, i, True))
            # train_sequence_queue.put((fasta_file, 10))
            # train_sequence_queue.put((fasta_file, 11))
        for i in range(train_arguments.number_train_workers):
            train_sequence_queue.put(None)

        validate_fasta_files = fasta_files[:train_arguments.number_validate_files_per_epoch]
        fasta_files = fasta_files[train_arguments.number_validate_files_per_epoch:]
        print(f"rank: {rank}; queueing for validation {epoch}: {validate_fasta_files}")
        for fasta_file in validate_fasta_files:
            # print(f"device: {rank}, train files: {fasta_files}")
            # with FastaFileReader(fasta_file) as fasta_file_reader:
            for i in range(10):
                item = (fasta_file, i, random.random() < 0.5)
                validate_sequence_queue.put(item)
                print(f"Epoch {epoch}: queued {item} for validation")
            # validate_sequence_queue.put((fasta_file, 10))
            # validate_sequence_queue.put((fasta_file, 11))
        for i in range(train_arguments.number_validate_workers):
            validate_sequence_queue.put(None)

    dataset.set_rank(rank)

    with DDPExperiment(model, "Pytorch DNA2VEC", rank=rank) as exp:
        parameters = {
            "epochs": train_arguments.epochs,
            "number_devices": train_arguments.number_devices,
            "number_train_workers": train_arguments.number_train_workers,
            "number_validate_workers": train_arguments.number_validate_workers,
            "vocabulary": train_arguments.vocab_artifact_uri,
            "number_of_train_files_per_epoch": train_arguments.number_train_files_per_epoch,
            "number_of_validation_files_per_epoch": train_arguments.number_validate_files_per_epoch,
            "optimizer": type(model.optimizer).__name__,
            "lr_initial": model.default_learning_rate,
            "optimizer_detailed": str(model.optimizer),
            "lr_scheduler": type(model.lr_scheduler).__name__,
            "lr_gamma": model.lr_scheduler.gamma,
            "loss_function": type(model.loss_function).__name__,
            "window_size": train_arguments.window_size,
            "kmer_size": train_arguments.kmer_size,
            "stride": train_arguments.stride,
            "batch_size": train_arguments.batch_size,
            "embedding_dimensions": train_arguments.embedding_dimensions,
        }

        mlflow.log_params(params=parameters)
        additional_tags = {}
        if train_arguments.tags is not None and len(train_arguments.tags) > 0:
            for tag in train_arguments.tags:
                keyValue = tag.split(":")
                if len(keyValue) > 1:
                    additional_tags[keyValue[0]] = keyValue[1]
        mlflow.set_tags({
                            "command": train_arguments.command
                        } | additional_tags)
        save_model_summary(model, (train_arguments.batch_size, train_arguments.window_size - 1), train_arguments.artifact_directory)
        with DDPTrainer(exp, epochs=train_arguments.epochs, device=device) as trainer:
            trainer.fit(dataloader, validate_dataloader)


def main():
    train_arguments = get_arguments()
    mlflow.artifacts.download_artifacts(artifact_uri=train_arguments.vocab_artifact_uri,
                                        dst_path=train_arguments.artifact_directory)
    ddprunner = DDPRunner(experiment, train_arguments)
    ddprunner.run()


if __name__ == "__main__":
    main()
