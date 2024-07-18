import mlflow.pyfunc
import mlflow.pytorch as mlflow_pytorch
import mlflow.pytorch
import pickle
import torch
from torch.utils.data import DataLoader
from fasta_pytorch_utils.data.FastaWindowQueuer import StreamingWindows
from fasta_pytorch_utils.data import FastaDataset
from fasta_pytorch_utils.data.FastaKmerDataset import FastaSequenceDataset
# from fasta_pytorch_utils.data.FastaDataset import F, create_dna_sequence_collate_function
from fasta_utils.tokenizers import KmerTokenizer
from fasta_utils import FastaFileReader
# from fasta_utils.vocab import Vocab
# from models.dna2vec import Dna2Vec

class CreateDnaSequenceCollateFunction:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def __call__(self, batch):
        """
        Collect then pad the sequences into a tensor
        :param batch: the batch
        :return: padded sequences, the original targets, and corresponding original lengths of each sequence
        """
        # sort the batch by length of sequence ascending
        batch = sorted(batch, key=lambda item: len(item[0]), reverse=True)
        # unzip the sequences from the corresponding targets
        [sequences, targets] = zip(*batch)

        # make the targets a 2-dimensional batch of size 1, so we can easily support multiple targets later
        # by easily refactoring the dataset and dataloader
        # targets = torch.stack([torch.tensor([target]) for target in targets], dim=0)
        targets = torch.unsqueeze(torch.tensor(targets, dtype=torch.float32), dim=1)
        # gather the original lengths of the sequence before padding
        lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)

        """
        The sequences should have already been loaded for cpu manipulation, so we should pad them before
        moving them to the gpu because it is more efficient to pad on the cpu
        """
        sequences = torch.nn.utils.rnn.pad_sequence([torch.tensor(sequence, dtype=torch.long) for sequence in sequences], batch_first=True, padding_value=self.vocabulary["pad"])
        return sequences, targets, lengths


device = 1

def load_scripted_model(model_uri):
    model_path = mlflow.artifacts.download_artifacts(model_uri, dst_path="./artifacts")
    model = torch.jit.load(model_path, map_location=torch.device(device))
    return model

def load_model(model_uri):
    model = mlflow_pytorch.load_model(model_uri, dst_path="./artifacts")
    model.to(device)
    return model

def load_vocabulary(vocabulary_uri):
    vocabulary_file = mlflow.artifacts.download_artifacts(artifact_uri=vocabulary_uri,
                                                          dst_path="./artifacts")
    with open(vocabulary_file, "rb") as file:
        return pickle.load(file)


model_uri = "mlflow-artifacts:/3/d9ce66e53ca643db801ef5ce3e4fccef/artifacts/scripted_model"
vocabulary_uri = "mlflow-artifacts:/2/5b1b448e36b74d74a5f043c6605ab538/artifacts/6mer-s1-202406010300.pickle"
kmer_size = 6
stride = 3
mlflow.set_tracking_uri("http://localhost:8080")

vocabulary = load_vocabulary(vocabulary_uri)
tokenizer = KmerTokenizer(kmer_size, stride)
window_streamer = StreamingWindows(vocabulary, window_size=7, tokenizer=tokenizer,return_tensors=True)
model = load_model(model_uri)
model.eval()
# model.to("cpu")

results = []
incorrect = []

train_dataset = FastaDataset("/home/jacob/PycharmProjects/pytorch_gene_recognition/data/train_data/genes.fa", tokenizer=tokenizer, vocabulary=vocabulary,
                                 dtype=torch.float32)
train_dataloader = DataLoader(train_dataset, batch_size=1,
                                  collate_fn=CreateDnaSequenceCollateFunction(vocabulary))
for item in train_dataloader:
    print(item)
    # print([vocabulary.get_token_for_index(token) for token in item[0]])
    break
#
# with (FastaFileReader("/home/jacob/PycharmProjects/pytorch_gene_recognition/data/train_data/genes.fa") as fasta_reader):
#     for header, sequence in fasta_reader.read_all():
#         total = 0
#         total_correct = 0
#         contexts = []
#         targets = []
#         for context, target in window_streamer.get_windows(sequence):
#             contexts.append(context)
#             targets.append(target)
#
#         contexts = torch.tensor(contexts, device=device)
#         targets = torch.tensor(targets, device=device)
#         output = model(contexts)
#         value, predicted = torch.max(output, 1)
#
#         batch_correct = int((predicted == targets).float().sum().item())
#         total += targets.size(0)
#         for actual, expected, context in zip(predicted.tolist(), targets.tolist(), contexts.tolist()):
#             print(value)
#             print(
#                 f"\texpected: {vocabulary.get_token_for_index(expected)}, actual: {vocabulary.get_token_for_index(actual)}, context: {[vocabulary.get_token_for_index(index) for index in (context[0:3] + [expected] + context[3:])]}")
#
#         # if total != batch_correct:
#         #     print(f"{header}, {batch_correct} of {total} correct")
#         #     incorrect = predicted != targets
#         #     incorrect_predicted = predicted[incorrect].tolist()
#         #     incorrect_targets = targets[incorrect].tolist()
#         #     incorrect_contexts = contexts[incorrect].tolist()
#         #     for actual, expected, context in zip(incorrect_predicted, incorrect_targets, incorrect_contexts):
#         #         print(value)
#         #         print(f"\texpected: {vocabulary.get_token_for_index(expected)}, actual: {vocabulary.get_token_for_index(actual)}, context: {[vocabulary.get_token_for_index(index) for index in (context[0:3] + [expected] + context[3:])]}")
#
#
#
