import torch


class Dna2Vec(torch.nn.Module):
    def __init__(self, vocabulary, embedding_dimension=32, device="cpu", optimizer=None, loss_function=None,
                 lr_scheduler=None, learning_rate=0.0001):
        super(Dna2Vec, self).__init__()
        self.vocabulary = vocabulary
        self.vocabulary_size = self.vocabulary.__len__()
        self.embedding_dimension = embedding_dimension
        self.embedding = torch.nn.Embedding(self.vocabulary_size, embedding_dimension,
                                            padding_idx=self.vocabulary["pad"])
        self.lstm = torch.nn.LSTM(embedding_dimension, embedding_dimension, batch_first=True)
        self.linear = torch.nn.Linear(embedding_dimension, self.vocabulary_size)
        self.default_learning_rate = learning_rate
        self.optimizer = optimizer or torch.optim.Adam(self.parameters(), lr=self.default_learning_rate, fused=True)
        self.loss_function = loss_function or torch.nn.CrossEntropyLoss()
        self.lr_scheduler = lr_scheduler or torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.6)

    def forward(self, context):
        # print(context.shape())
        embeds = self.embedding(context)
        # output = torch.relu(lstm_output.mean(dim=1))

        lstm_output, _ = self.lstm(embeds)
        output = self.linear(lstm_output[:, -1, :])  # Use the output of the last time step

        return output

    def get_optimizer(self):
        return self.optimizer

    def get_loss_function(self):
        return self.loss_function

    def get_learning_rate_scheduler(self):
        return self.lr_scheduler
