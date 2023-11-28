import unittest
import torch
import torch.nn as nn

from src.constants import CHAR2ID
from src.models import CharRNN, CharLSTM


class TestCharRNN(unittest.TestCase):
    def test_output_shape(self):
        # Define model parameters
        input_size = len(CHAR2ID)
        hidden_size = 128
        output_size = len(CHAR2ID)
        num_hidden_layers = 1
        embedding_dim = 100

        # Create the model instance
        model = CharRNN(input_size, hidden_size, output_size, num_hidden_layers, embedding_dim)

        # Create a dummy input tensor [batch_size, sequence_length]
        # Assuming CHAR2ID maps characters to indices
        batch_size = 4
        sequence_length = 10  # example sequence length
        dummy_input = torch.randint(0, input_size, (batch_size, sequence_length))

        # Get the model output
        output = model(dummy_input)

        # Check the output shape
        expected_shape = (batch_size, output_size)
        self.assertEqual(output.shape, expected_shape)

class TestCharLSTM(unittest.TestCase):
    def test_output_shape(self):
        # Define model parameters
        input_size = len(CHAR2ID)
        hidden_size = 128
        output_size = len(CHAR2ID)
        num_hidden_layers = 1
        embedding_dim = 100
        dropout_rate = 0.5

        # Create the model instance
        model = CharLSTM(input_size, hidden_size, output_size, num_hidden_layers, embedding_dim, dropout_rate)

        # Create a dummy input tensor [batch_size, sequence_length]
        batch_size = 4
        sequence_length = 10  # Example sequence length
        dummy_input = torch.randint(0, input_size, (batch_size, sequence_length))

        # Get the model output
        output = model(dummy_input)

        # Check the output shape
        expected_shape = (batch_size, output_size)
        self.assertEqual(output.shape, expected_shape)


if __name__ == '__main__':
    unittest.main()
