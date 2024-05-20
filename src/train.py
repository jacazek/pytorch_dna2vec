import torch.nn
import os
import torch
from torchinfo import summary
from trainer import DDPTrainer, DDPRunner
from experiment import DDPExperiment
from multiprocessing import Queue, Process
import mlflow

import torch_utils
from fasta_utils.tokenizers import KmerTokenizer
from fasta_utils.vocab import Vocab
from models.dna2vec import Dna2Vec
from fasta_pytorch_utils.data import FastaKmerDataset
from torch.utils.data import DataLoader

script_directory = os.path.dirname(os.path.realpath(__file__))
root_directory = os.path.abspath(os.path.join(script_directory, "../"))


# mlflow.autolog()
mlflow.set_tracking_uri("http://localhost:8080")
num_devices = 2
num_workers = 10
device = torch_utils.get_device()


vocab_artifact_uri="mlflow-artifacts:/2/bd867003e0ed4696ac17251250c25564/artifacts/7mer-s3-202405182143.pickle"
artifact_name = os.path.basename(vocab_artifact_uri)
artifact_directory = "./artifacts"
artifact_path = os.path.join(artifact_directory, artifact_name)


def save_model_summary(model, input_size):
    file_name = "./artifacts/model_summary.txt"
    with open(file_name, "w") as file:
        summary_string = str(summary(model, input_size, dtypes=[torch.long]))
        file.write(summary_string)
    mlflow.log_artifact(file_name, artifact_path="artifacts")


def experiment(rank, world_size):
    fasta_file_directory = os.path.join(root_directory, "data")
    fasta_file_extension = ".fa.gz"
    fasta_files = [f"{fasta_file_directory}/{file}" for file in os.listdir(fasta_file_directory) if
                   file.endswith(fasta_file_extension)]
    # validate_fasta_files = [f"{fasta_file_directory}/{file}" for file in os.listdir(fasta_file_directory) if
    #                file.endswith(fasta_file_extension)][2:4]
    # print(train_fasta_files)
    files_per_gpu = int(len(fasta_files) / num_devices)
    start = rank * files_per_gpu
    end = start + files_per_gpu
    fasta_files = fasta_files[start:end]

    train_sequence_queue = Queue()
    validate_sequence_queue = Queue()


    # loading artifacts needs to happen once before worker threads are launched
    # otherwise there may be corruption
    vocabulary = Vocab.load(artifact_path)
    embedding_dimension = 512
    model = Dna2Vec(vocabulary, embedding_dimension=embedding_dimension, device=device)

    # pull these from vocabulary metadata later
    kmer_size = 7
    stride = 3
    tokenizer = KmerTokenizer(kmer_size, stride)

    window_size = 7
    batch_size = 20480
    dataset = FastaKmerDataset(train_sequence_queue, tokenizer=tokenizer, vocabulary=vocabulary, window_size=window_size,
                           device_count=num_devices)
    dataloader = DataLoader(dataset, batch_size, num_workers=num_workers, prefetch_factor=10, pin_memory=True)
    validate_dataset = FastaKmerDataset(validate_sequence_queue, tokenizer=tokenizer, vocabulary=vocabulary, window_size=window_size,
                               device_count=num_devices)
    validate_dataloader = DataLoader(validate_dataset, batch_size, num_workers=num_workers, prefetch_factor=10, pin_memory=True)


    # torch.set_float32_matmul_precision('medium')
    number_of_train_fasta_files_per_epoch = 1
    number_of_validation_fasta_files_per_epoch = 1

    epochs = 3
    for epoch in range(epochs):
        train_fasta_files = fasta_files[:number_of_train_fasta_files_per_epoch]
        fasta_files = fasta_files[number_of_train_fasta_files_per_epoch:]
        print(f"rank: {rank}; queueing for training {epoch}: {train_fasta_files}")
        for fasta_file in train_fasta_files:
            # print(f"device: {rank}, train files: {fasta_files[start:end]}")
            # with FastaFileReader(fasta_file) as fasta_file_reader:
            for i in range(10):
                train_sequence_queue.put((fasta_file, i))
            # train_sequence_queue.put((fasta_file, 10))
            # train_sequence_queue.put((fasta_file, 11))
        for i in range(num_workers):
            train_sequence_queue.put(None)

        validate_fasta_files = fasta_files[:number_of_validation_fasta_files_per_epoch]
        fasta_files = fasta_files[number_of_validation_fasta_files_per_epoch:]
        print(f"rank: {rank}; queueing for validation {epoch}: {validate_fasta_files}")
        for fasta_file in validate_fasta_files:
            # print(f"device: {rank}, train files: {fasta_files}")
            # with FastaFileReader(fasta_file) as fasta_file_reader:
            for i in range(10):
                validate_sequence_queue.put((fasta_file, i))
            # validate_sequence_queue.put((fasta_file, 10))
            # validate_sequence_queue.put((fasta_file, 11))
        for i in range(num_workers):
            validate_sequence_queue.put(None)

    dataset.set_rank(rank)

    with DDPExperiment(model,"Pytorch DNA2VEC", rank=rank) as exp:
        parameters = {
            "epochs": epochs,
            "number_of_devices": num_devices,
            "number_of_workers": num_workers,
            "vocabulary": vocab_artifact_uri,
            "number_of_train_files_per_epoch": number_of_train_fasta_files_per_epoch,
            "number_of_validation_files_per_epoch": number_of_validation_fasta_files_per_epoch,
            "optimizer": type(model.optimizer).__name__,
            "lr_initial": model.default_learning_rate,
            "optimizer_detailed": str(model.optimizer),
            "lr_scheduler": type(model.lr_scheduler).__name__,
            "lr_gamma": model.lr_scheduler.gamma,
            "loss_function": type(model.loss_function).__name__,
            "window_size": window_size,
            "kmer_size": kmer_size,
            "stride": stride,
            "batch_size": batch_size
        }
        mlflow.log_params(params=parameters)
        save_model_summary(model,  ( batch_size, window_size-1))
        with DDPTrainer(exp, epochs=epochs, device=device) as trainer:
            trainer.fit(dataloader, validate_dataloader)


def main():
    mlflow.artifacts.download_artifacts(artifact_uri=vocab_artifact_uri, dst_path=artifact_directory)
    ddprunner = DDPRunner(experiment, num_devices)
    ddprunner.run()

if __name__ == "__main__":
    main()
