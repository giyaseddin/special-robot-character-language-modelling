import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import os

from src.constants import PROJECT_ROOT_PATH
from src.data_loader import separate_and_concat_xy, save_split, load_split, load_raw_dataset, \
    get_processed_train_valid_test
from src.dataset import get_full_train_test_loaders, get_train_valid_test_loaders


class TestDataLoader(unittest.TestCase):

    def setUp(self):
        # Create temporary files for train and test data
        self.sample_data = pd.DataFrame({'text': ['hello', 'world', 'test', 'data']})

    def test_load_raw_dataset(self):
        # Test
        train_df, test_df = load_raw_dataset(data_path=PROJECT_ROOT_PATH.joinpath("data"))

        assert len(train_df)
        assert len(test_df)

    def test_get_processed_train_valid_test(self):
        train_X, train_y, val_X, val_y, test_X, test_y = get_processed_train_valid_test()
        self.assertIsInstance(train_X, pd.Series)
        self.assertIsInstance(train_y, pd.Series)
        self.assertIsInstance(val_X, pd.Series)
        self.assertIsInstance(val_y, pd.Series)
        self.assertIsInstance(test_X, pd.Series)
        self.assertIsInstance(test_y, pd.Series)

    def test_separate_and_concat_xy(self):
        # Test the separate_and_concat_xy function
        result_df = separate_and_concat_xy(self.sample_data.copy())
        self.assertIn('train_X', result_df.columns)
        self.assertIn('train_y', result_df.columns)
        self.assertEqual(result_df['train_X'].iloc[0], 'hell')
        self.assertEqual(result_df['train_y'].iloc[0], 'o')

    def test_save_and_load_data(self):
        # Test both save_data and load_data functions
        temp_file = 'temp_test_file.csv'
        save_split(self.sample_data, temp_file)
        loaded_data = load_split(temp_file)
        pd.testing.assert_frame_equal(loaded_data, self.sample_data)
        os.remove(temp_file)  # Clean up the temporary file


class TestLoaderFunctions(unittest.TestCase):

    @patch('src.dataset.get_processed_train_valid_test')
    @patch('src.dataset.DataLoader')
    def test_get_train_valid_test_loaders(self, mock_dataloader, mock_get_processed):
        # Mock the return value of get_processed_train_valid_test
        mock_get_processed.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())

        # Call the function
        train_loader, val_loader, test_loader = get_train_valid_test_loaders(batch_size=32, target_as_sequence=False)

        # Assertions to ensure DataLoader is called correctly
        self.assertEqual(mock_dataloader.call_count, 3)  # Called three times for train, val, and test

    @patch('src.dataset.get_processed_train_test')
    @patch('src.dataset.DataLoader')
    def test_get_full_train_test_loaders(self, mock_dataloader, mock_get_processed):
        # Mock the return value of get_processed_train_test
        mock_get_processed.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())

        # Call the function
        full_train_loader, test_loader = get_full_train_test_loaders(batch_size=32, target_as_sequence=False)

        # Assertions to ensure DataLoader is called correctly
        self.assertEqual(mock_dataloader.call_count, 2)  # Called twice for full train and test


if __name__ == '__main__':
    unittest.main()
