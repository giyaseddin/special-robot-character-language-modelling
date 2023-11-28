import unittest
import torch
from src.constants import CHAR2ID, CHAR2ID_WITH_PAD
from src.dataset import CharSequenceDataset, CharSubsequenceDataset


class TestCharSequenceDataset(unittest.TestCase):

    def setUp(self):
        self.sequences = ['ABC', 'DEF', 'GHI']
        self.labels = ['J', 'K', 'L']
        self.dataset = CharSequenceDataset(self.sequences, self.labels)

    def test_len(self):
        self.assertEqual(len(self.dataset), len(self.sequences))

    def test_getitem(self):
        sequence, label = self.dataset[0]
        expected_sequence = torch.tensor([CHAR2ID[char] for char in self.sequences[0]], dtype=torch.long)
        expected_label = torch.tensor(CHAR2ID[self.labels[0]], dtype=torch.long)
        self.assertTrue(torch.equal(sequence, expected_sequence))
        self.assertTrue(torch.equal(label, expected_label))


class TestCharSubsequenceDataset(unittest.TestCase):

    def setUp(self):
        self.sequences = ['ABC', 'DEF']
        self.dataset = CharSubsequenceDataset(self.sequences)

    def test_len(self):
        expected_length = sum(len(seq) - 1 for seq in self.sequences)
        self.assertEqual(len(self.dataset), expected_length)

    def test_getitem(self):
        first_subseq, first_next_char = self.dataset[0]
        expected_subseq = torch.tensor([CHAR2ID_WITH_PAD[char] for char in self.sequences[0][0]], dtype=torch.long)
        expected_next_char = torch.tensor(CHAR2ID_WITH_PAD[self.sequences[0][1]], dtype=torch.long)

        self.assertTrue(torch.equal(first_subseq, expected_subseq))
        self.assertTrue(torch.equal(first_next_char, expected_next_char))


if __name__ == '__main__':
    unittest.main()
