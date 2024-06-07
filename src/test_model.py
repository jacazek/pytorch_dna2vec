import mlflow.pyfunc
from mlflow.pytorch import load_model
import mlflow.pytorch
import torch
from fasta_pytorch_utils.data.FastaWindowQueuer import StreamingWindows
from fasta_utils.tokenizers import KmerTokenizer
from fasta_utils import FastaFileReader
from src.models.dna2vec import Dna2Vec

model_uri = "mlflow-artifacts:/3/214f94223ccb43928219b4b98761a9cf/artifacts/model"
vocabulary_uri = "mlflow-artifacts:/2/5b1b448e36b74d74a5f043c6605ab538/artifacts/6mer-s1-202406010300.pickle"
kmer_size = 6
stride = 3
mlflow.set_tracking_uri("http://localhost:8080")

vocabulary = mlflow.artifacts.download_artifacts(artifact_uri=vocabulary_uri,
                                        dst_path="./artifacts")
tokenizer = KmerTokenizer(kmer_size, stride)
window_streamer = StreamingWindows(vocabulary, window_size=7, tokenizer=tokenizer)
model = load_model(model_uri, dst_path="./artifacts")


with FastaFileReader("../data/Zm-CML103-REFERENCE-NAM-1.0.fa.gz") as fasta_reader:
    for header, sequence in fasta_reader.read_at_index(0):
        for context, target in window_streamer.get_windows(sequence):
            prediction = model(context)
            print(target)
            print(prediction)
            break
        break