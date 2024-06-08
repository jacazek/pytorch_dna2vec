import torch


class Dna2Vec(torch.nn.Module):
    def __init__(self, vocabulary, embedding_dimension=32, device="cpu", optimizer=None, loss_function=None,
                 lr_scheduler=None, learning_rate=0.0001):
        super(Dna2Vec, self).__init__()
        self.vocabulary = vocabulary
        self.vocabulary_size = self.vocabulary.__len__()
        self.embedding_dimension = embedding_dimension
        self.device = device
        self.embedding = torch.nn.Embedding(self.vocabulary_size, embedding_dimension,
                                            padding_idx=self.vocabulary["pad"],
                                            # implicit dependency on vocabulary padding
                                            device=self.device)
        # self.linear1 = torch.nn.Linear(embedding_dimension, embedding_dimension, device=self.device)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=embedding_dimension, nhead=2, dim_feedforward=16, batch_first=True, device=device)
        self.position_embedding = torch.nn.Embedding(100, embedding_dimension, device=device)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.layer_norm = torch.nn.LayerNorm(embedding_dimension, device=device)

        self.linear = torch.nn.Linear(embedding_dimension, self.vocabulary_size, device=self.device)
        self.default_learning_rate = learning_rate
        self.optimizer = optimizer or torch.optim.Adam(self.parameters(), lr=self.default_learning_rate, fused=True)
        self.loss_function = loss_function or torch.nn.CrossEntropyLoss()
        self.lr_scheduler = lr_scheduler or torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.6)

    def forward(self, context):
        # print(context.shape())
        sequence_length = context.size(1)
        position_ids = torch.arange(sequence_length, dtype=torch.long, device=self.device).unsqueeze(0)
        embeds = self.embedding(context)
        position_embeds = self.position_embedding(position_ids)
        embeds = self.layer_norm(embeds + position_embeds)

        output = self.transformer_encoder(embeds)  # Transformer expects (S, N, E) format
        output = self.linear(output.mean(dim=1))  # Use the output of the last time step
        return output

    def get_optimizer(self):
        return self.optimizer

    def get_loss_function(self):
        return self.loss_function

    def get_learning_rate_scheduler(self):
        return self.lr_scheduler
