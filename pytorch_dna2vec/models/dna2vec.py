import torch


class Dna2Vec(torch.nn.Module):
    def __init__(self, vocabulary, embedding_dimension=32, device="cpu"):
        super(Dna2Vec, self).__init__()
        self.vocabulary = vocabulary
        self.vocabulary_size = self.vocabulary.__len__()
        self.embedding_dimension = embedding_dimension
        self.device = device
        self.embedding = torch.nn.Embedding(self.vocabulary_size, embedding_dimension,
                                      padding_idx=self.vocabulary["pad"],  # implicit dependency on vocabulary padding
                                      device=self.device)
        self.linear = torch.nn.Linear(embedding_dimension, self.vocabulary_size, device=self.device)

    def forward(self, context):
        embeds = self.embedding(context).mean(dim=1)
        output = self.linear(embeds)
        return output

