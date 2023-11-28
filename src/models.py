import torch
from torch import nn
import torch.nn.functional as F

from src.constants import CHAR2ID, CHAR2ID_WITH_PAD


class CharRNN(nn.Module):
    """
    A simple RNN model for character-level predictions.

    Attributes:
        embedding (Embedding): Embedding layer.
        rnn (RNN): Recurrent Neural Network layer.
        fc (Linear): Fully connected layer to produce output.
    """

    def __init__(self, input_size=len(CHAR2ID), hidden_size=128, output_size=len(CHAR2ID), num_hidden_layers=1,
                 embedding_dim=100):
        super(CharRNN, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers=num_hidden_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)

        rnn_out, _ = self.rnn(embedded)

        return self.fc(rnn_out[:, -1, :])


class CharLSTM(nn.Module):
    """
    A simple LSTM model for character-level predictions.

    Attributes:
        embedding (Embedding): Embedding layer.
        lstm (LSTM): Long Short-Term Memory layer.
        fc (Linear): Fully connected layer to produce output.
    """

    def __init__(self, input_size=len(CHAR2ID), hidden_size=128, output_size=len(CHAR2ID), num_hidden_layers=1,
                 embedding_dim=100, dropout_rate=0.5):
        super(CharLSTM, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_size, num_layers=num_hidden_layers, batch_first=True,
            dropout=dropout_rate if num_hidden_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)

        lstm_out, _ = self.lstm(embedded)

        return self.fc(lstm_out[:, -1, :])


class NextCharLSTM(nn.Module):
    """
    LSTM model for next character prediction in a sequence.

    Attributes:
        embedding (Embedding): Embedding layer.
        lstm (LSTM): Long Short-Term Memory layer.
        fc (Linear): Fully connected layer to produce output.
    """

    def __init__(self, vocab=CHAR2ID_WITH_PAD, hidden_size=128, num_hidden_layers=2, embedding_dim=100,
                 dropout_rate=0.5):
        super(NextCharLSTM, self).__init__()
        self.vocab_size = len(vocab)
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_size, num_layers=num_hidden_layers, batch_first=True,
            dropout=dropout_rate if num_hidden_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, self.vocab_size)

    def forward(self, x):
        x = self.embedding(x)

        lstm_out, _ = self.lstm(x)

        return self.fc(lstm_out[:, -1, :])


class CharCNN(nn.Module):
    """
    Convolutional Neural Network model for character-level predictions.

    Attributes:
        embedding_layer (Embedding): Embedding layer.
        conv_layer (Conv1d): Convolutional layer.
        dropout (Dropout): Dropout layer.
        fc_layer (Linear): Fully connected layer to produce output.
    """

    def __init__(self, max_seq_length=8, vocab_size=len(CHAR2ID), num_output_classes=len(CHAR2ID), embedding_dim=100,
                 num_conv_filters=50,
                 conv_kernel_size=3, dropout_rate=0.5):
        super(CharCNN, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.conv_layer = nn.Conv1d(embedding_dim, num_conv_filters, conv_kernel_size)
        self.dropout = nn.Dropout(dropout_rate)

        conv_output_length = max_seq_length - conv_kernel_size + 1
        self.fc_layer = nn.Linear(num_conv_filters * conv_output_length, num_output_classes)

    def forward(self, x):
        embedded_seq = self.embedding_layer(x).permute(0, 2, 1)

        conv_output = F.relu(self.conv_layer(embedded_seq))

        conv_output = self.dropout(conv_output)

        return self.fc_layer(torch.flatten(conv_output, start_dim=1))


class CharCNNYLecun(nn.Module):
    """
    Convolutional Neural Network model for character-level text classification.
    Simplest implementation of Xhang et al. (https://arxiv.org/abs/1509.01626)

    Attributes:
        embedding (Embedding): Embedding layer to convert character indices to embeddings.
        conv1 (Conv1d): Convolutional layer for feature extraction.
        pool (MaxPool1d): Max pooling layer to reduce the dimensionality.
        fc (Linear): Fully connected layer for classification.
        dropout (Dropout): Dropout layer for regularization.
    """

    def __init__(self, max_seq_length=9, conv_kernel_size=3, embedding_dim=100, num_conv_filters=50,
                 vocab_size=len(CHAR2ID), num_output_classes=len(CHAR2ID), dropout_rate=0.5):
        super(CharCNNYLecun, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, num_conv_filters, kernel_size=conv_kernel_size,
                               padding=conv_kernel_size // 2)
        self.pool = nn.MaxPool1d(max_seq_length - conv_kernel_size + 1)
        self.fc = nn.Linear(num_conv_filters, num_output_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Defines the forward pass of the CharCNN model.

        Args:
            x (tensor): Input tensor containing character indices.

        Returns:
            tensor: Output tensor representing class probabilities or scores.
        """
        x = self.embedding(x)  # Embedding shape: [batch_size, max_seq_length, embedding_dim]
        # Transpose to fit Conv1d input requirements
        x = x.transpose(1, 2)  # Shape: [batch_size, embedding_dim, max_seq_length]

        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)  # Reshape to [batch_size, -1]

        x = self.dropout(x)

        return self.fc(x)


class BiTriGramLSTM(nn.Module):
    def __init__(self, input_size=len(CHAR2ID), hidden_size=128, output_size=len(CHAR2ID), num_hidden_layers=1,
                 embedding_dim=100, dropout_rate=0.5):
        super(BiTriGramLSTM, self).__init__()

        self.embedding = nn.Embedding(input_size, embedding_dim=embedding_dim)

        # Convolutional layers for bigrams and trigrams
        self.conv2 = nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_size, kernel_size=2, padding=1)
        self.conv3 = nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_size, kernel_size=3, padding=1)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_size * 2, hidden_size=hidden_size, num_layers=num_hidden_layers, batch_first=True,
            dropout=dropout_rate if num_hidden_layers > 1 else 0
        )

        # Fully connected layer for output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Embedding input
        embedded = self.embedding(x)  # Shape: (batch_size, seq_length, embedding_dim)
        embedded = embedded.permute(0, 2, 1)  # Shape: (batch_size, embedding_dim, seq_length)

        # Apply convolutional layers
        bigram_features = nn.functional.relu(
            self.conv2(embedded))[:, :, :-1]  # Shape: (batch_size, hidden_size, seq_length)
        trigram_features = nn.functional.relu(self.conv3(embedded))  # Shape: (batch_size, hidden_size, seq_length)

        # Concatenate bigram and trigram features
        combined = torch.cat((bigram_features, trigram_features),
                             dim=1)  # Shape: (batch_size, hidden_size * 2, seq_length)

        # LSTM layer
        combined = combined.permute(0, 2, 1)  # Shape: (batch_size, seq_length, hidden_size * 2)
        lstm_out, _ = self.lstm(combined)

        # Final output
        out = self.fc(lstm_out[:, -1, :])
        return out
