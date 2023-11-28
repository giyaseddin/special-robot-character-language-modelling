from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.constants import PROJECT_ROOT_PATH, RANDOM_STATE


def separate_and_concat_xy(df: pd.DataFrame, prefix: str = "train") -> pd.DataFrame:
    """
    Separates and concatenates the input and output data.

    Parameters:
    df (pd.DataFrame): Dataframe containing the text data.
    prefix (str, optional): Prefix for naming the columns. Defaults to "train".

    Returns:
    pd.DataFrame: Updated dataframe with separated input and output columns.
    """
    df[prefix + "_X"] = df["text"].apply(lambda x: x[:-1])
    df[prefix + "_y"] = df["text"].apply(lambda x: x[-1])

    return df


def load_raw_dataset(data_path: Path = PROJECT_ROOT_PATH.joinpath("data")) -> Tuple[
    pd.DataFrame, pd.DataFrame]:
    """
    Loads the raw dataset from the specified path.

    Parameters:
    data_path (str, optional): Path to the dataset directory. Defaults to "../data".

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing training and test dataframes.
    """
    train_df = pd.read_csv(data_path.joinpath("train.csv"), header=None, names=["text"])
    test_df = pd.read_csv(data_path.joinpath("answers.csv"), header=None, names=["text"])

    return train_df, test_df


def get_processed_train_valid_test(valid_size=0.1, target_as_sequence=False) -> Tuple[pd.Series, ...]:
    """
    Processes the raw data and splits it into training, validation, and test sets.

    Parameters:
    valid_size (float, optional): Proportion of the dataset to include in the validation split. Defaults to 0.1.
    target_as_sequence (bool, optional): Whether the target should be treated as a sequence. Defaults to False.

    Returns:
    Tuple[pd.Series, ...]: Tuple containing train inputs, train targets, validation inputs, validation targets,
                           test inputs, and test targets.
    """
    train_df, test_df = load_raw_dataset()
    train_df = separate_and_concat_xy(train_df, prefix="train")
    test_df = separate_and_concat_xy(test_df, prefix="test")

    train_X, val_X, train_y, val_y = train_test_split(
        train_df["train_X"], (train_df["text"] if target_as_sequence else train_df["train_y"]),
        test_size=valid_size, random_state=RANDOM_STATE
    )

    return train_X, train_y, val_X, val_y, test_df["test_X"], (
        test_df["text"] if target_as_sequence else test_df["test_y"]
    )


def get_processed_train_test(target_as_sequence=False) -> Tuple[pd.Series, ...]:
    """
    Processes the raw data and splits it into training, and test sets only.

    Parameters:
    target_as_sequence (bool, optional): Whether the target should be treated as a sequence. Defaults to False.

    Returns:
    Tuple[pd.Series, ...]: Tuple containing train inputs, train targets, test inputs, and test targets.
    """
    train_df, test_df = load_raw_dataset()
    train_df = separate_and_concat_xy(train_df, prefix="train")
    test_df = separate_and_concat_xy(test_df, prefix="test")

    return train_df["train_X"], train_df["train_y"], test_df["test_X"], (
        test_df["text"] if target_as_sequence else test_df["test_y"]
    )


def save_split(data, file_path):
    """
    Saves the provided data to a CSV file.

    Parameters:
    data (pd.Series or pd.DataFrame): Data to be saved.
    file_path (str): File path where the data should be saved.
    """
    data.to_csv(file_path, index=False)


def load_split(file_path):
    """
    Loads data from a CSV file.

    Parameters:
    file_path (str): File path from where the data should be loaded.

    Returns:
    pd.Series or pd.DataFrame: Loaded data.
    """
    return pd.read_csv(file_path)


if __name__ == '__main__':
    destin_path = PROJECT_ROOT_PATH.joinpath("data", "processed")
    # Saving and loading for character target
    train_X, train_y, val_X, val_y, test_X, test_y = get_processed_train_valid_test()
    save_split(train_X, destin_path.joinpath("train_X.csv"))
    save_split(train_y, destin_path.joinpath("train_y.csv"))
    save_split(val_X, destin_path.joinpath("val_X.csv"))
    save_split(val_y, destin_path.joinpath("val_y.csv"))
    save_split(test_X, destin_path.joinpath("test_X.csv"))
    save_split(test_y, destin_path.joinpath("test_y.csv"))

    # Saving and loading for sequence target only
    train_X_seq, train_y_seq, val_X_seq, val_y_seq, test_X_seq, test_y_seq = get_processed_train_valid_test(
        target_as_sequence=True
    )
    save_split(train_y_seq, destin_path.joinpath("train_y_seq.csv"))
    save_split(val_y_seq, destin_path.joinpath("val_y_seq.csv"))
    save_split(test_y_seq, destin_path.joinpath("test_y_seq.csv"))
