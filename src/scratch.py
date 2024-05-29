import os
import torch_utils
from fasta_utils import FastaFileReader
from fasta_utils.vocab import Vocab
from models.dna2vec import Dna2Vec
from fasta_pytorch_utils.data import FastaKmerDataset
from torch.utils.data import DataLoader

script_directory = os.path.dirname(os.path.realpath(__file__))
root_directory = os.path.abspath(os.path.join(script_directory, "../"))


fasta_file_directory = os.path.join(root_directory, "data")
fasta_file_extension = ".fa.gz"
fasta_files = [f"{fasta_file_directory}/{file}" for file in os.listdir(fasta_file_directory) if
               file.endswith(fasta_file_extension)]


for fasta_file_path in fasta_files:
    with FastaFileReader(fasta_file_path) as fasta_file:
        print(fasta_file.index_table[0])
        print((fasta_file_path, sum([row[1] for row in fasta_file.index_table[:10]])))
