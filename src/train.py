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
    train_fasta_files = [f"{fasta_file_directory}/{file}" for file in os.listdir(fasta_file_directory) if
                   file.endswith(fasta_file_extension)][0:2]
    validate_fasta_files = [f"{fasta_file_directory}/{file}" for file in os.listdir(fasta_file_directory) if
                   file.endswith(fasta_file_extension)][2:4]
    # print(train_fasta_files)

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

    files_per_gpu = int(len(train_fasta_files) / num_devices)
    start = rank * files_per_gpu
    end = start + files_per_gpu

    epochs = 2
    for epoch in range(epochs):
        for fasta_file in train_fasta_files[start:end]:
            # print(f"device: {rank}, train files: {fasta_files[start:end]}")
            # with FastaFileReader(fasta_file) as fasta_file_reader:
            for i in range(10):
                train_sequence_queue.put((fasta_file, i))
            # train_sequence_queue.put((fasta_file, 10))
            # train_sequence_queue.put((fasta_file, 11))
        for i in range(num_workers):
            train_sequence_queue.put(None)

        # if rank  == 0:
        for fasta_file in validate_fasta_files[start:end]:
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
        save_model_summary(model,  ( batch_size, window_size-1))
        with DDPTrainer(exp, epochs=epochs, device=device) as trainer:
            trainer.fit(dataloader, validate_dataloader)


def main():
    mlflow.artifacts.download_artifacts(artifact_uri=vocab_artifact_uri, dst_path=artifact_directory)
    ddprunner = DDPRunner(experiment, num_devices)
    ddprunner.run()

if __name__ == "__main__":
    main()
