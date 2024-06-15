import mlflow.pyfunc
from mlflow.pytorch import load_model
import mlflow.pytorch
import torch
from torch.utils.data import DataLoader
from fasta_pytorch_utils.data.FastaWindowQueuer import StreamingWindows
from fasta_pytorch_utils.data.FastaKmerDataset import FastaSequenceDataset
# from fasta_pytorch_utils.data.FastaDataset import F, create_dna_sequence_collate_function
from fasta_utils.tokenizers import KmerTokenizer
from fasta_utils import FastaFileReader
from fasta_utils.vocab import Vocab
# from models.dna2vec import Dna2Vec

device = 1

def load_scripted_model(model_uri):
    model_path = mlflow.artifacts.download_artifacts(model_uri, dst_path="./artifacts")
    model = torch.jit.load(model_path, map_location=torch.device(device))
    return model

model_uri = "mlflow-artifacts:/3/d9ce66e53ca643db801ef5ce3e4fccef/artifacts/scripted_model/data/model.pth"
vocabulary_uri = "mlflow-artifacts:/2/5b1b448e36b74d74a5f043c6605ab538/artifacts/6mer-s1-202406010300.pickle"
kmer_size = 6
stride = 3
mlflow.set_tracking_uri("http://localhost:8080")

vocabulary_file = mlflow.artifacts.download_artifacts(artifact_uri=vocabulary_uri,
                                        dst_path="./artifacts")
vocabulary: Vocab = Vocab.load(vocabulary_file)
tokenizer = KmerTokenizer(kmer_size, stride)
window_streamer = StreamingWindows(vocabulary, window_size=7, tokenizer=tokenizer,return_tensors=True)
model = load_scripted_model(model_uri)
# model.to("cpu")

results = []
incorrect = []

with (FastaFileReader("/home/jacob/DEV/data-science-example/data/train_data/genes.fa") as fasta_reader):
    for header, sequence in fasta_reader.read_all():
        total = 0
        total_correct = 0
        contexts = []
        targets = []
        for context, target in window_streamer.get_windows(sequence):
            contexts.append(context)
            targets.append(target)

        contexts = torch.tensor(contexts, device=device)
        targets = torch.tensor(targets, device=device)
        output = model(contexts)
        value, predicted = torch.max(output, 1)

        batch_correct = int((predicted == targets).float().sum().item())
        total += targets.size(0)
        if total != batch_correct:
            print(f"{header}, {batch_correct} of {total} correct")
            incorrect = predicted != targets
            incorrect_predicted = predicted[incorrect].tolist()
            incorrect_targets = targets[incorrect].tolist()
            incorrect_contexts = contexts[incorrect].tolist()
            for actual, expected, context in zip(incorrect_predicted, incorrect_targets, incorrect_contexts):
                print(f"\texpected: {vocabulary.get_token_for_index(expected)}, target: {vocabulary.get_token_for_index(actual)}, context: {[vocabulary.get_token_for_index(index) for index in (context[0:2] + [actual] + context[3:])]}")



