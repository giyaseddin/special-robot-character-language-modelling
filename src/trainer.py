import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import seaborn as sns
import importlib

from src.constants import CHAR2ID, ID2CHAR


class DLTrainer:
    def __init__(self, model_name, model_args, train_loader, val_loader, test_loader, device, learning_rate,
                 num_epochs=20,
                 early_stopping_patience=7, weight_decay=0.05, checkpoint_path=None, char2id=CHAR2ID, id2char=ID2CHAR):
        """
        Initializes the Trainer class with model, dataloaders, and training parameters.

        Args:
            model_name (str): Name of the model class to be used.
            model_args (dict): Arguments to be passed to the model.
            train_loader, val_loader (DataLoader): Training and validation data loaders.
            device (torch.device): Device to train the model on.
            learning_rate (float): Learning rate for the optimizer.
            num_epochs (int): Number of epochs for training.
            early_stopping_patience (int): Patience for early stopping.
            checkpoint_path (str): Path to save the best model.
        """
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.model_name = model_name
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else Path(
            "trained_models", self.model_name + "_best.pth"
        )

        self.train_losses = []
        self.val_losses = []
        self.char2id = char2id
        self.id2char = id2char

        # Dynamically load the model class from src.models
        model_module = importlib.import_module(f"src.models")
        model_class = getattr(model_module, self.model_name)
        self.model = model_class(**model_args).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def train_epoch(self):
        """
        Runs one training epoch.
        """
        self.model.train()
        total_train_loss = 0
        for input_seq, next_char in self.train_loader:
            input_seq, next_char = input_seq.to(self.device), next_char.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(input_seq)
            loss = self.criterion(output, next_char)
            loss.backward()
            self.optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(self.train_loader)
        self.train_losses.append(avg_train_loss)
        return avg_train_loss

    def validate(self):
        """
        Runs one validation epoch.
        """
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for input_seq, next_char in self.val_loader:
                input_seq, next_char = input_seq.to(self.device), next_char.to(self.device)
                output = self.model(input_seq)
                loss = self.criterion(output, next_char)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(self.val_loader)
        self.val_losses.append(avg_val_loss)
        return avg_val_loss

    def train(self):
        """
        Runs the entire training process with early stopping.
        """
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(self.num_epochs):
            avg_train_loss = self.train_epoch()
            avg_val_loss = self.validate()

            print(
                f"Epoch {epoch + 1}/{self.num_epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                self.save_model()
                print(f"Model improved and saved to {self.checkpoint_path}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break

        return self.model, self.train_losses, self.val_losses

    def save_model(self):
        """
        Saves the model state to the specified path.
        """
        torch.save(self.model.state_dict(), self.checkpoint_path)

    def evaluate(self):
        """
        Evaluates the model, visualizes results with character labels,
        and saves results and visualizations to the 'results' folder.
        """
        self.model.eval()

        # Function to get predictions and convert IDs to characters
        def get_predictions(loader):
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for input_seq, next_char in loader:
                    input_seq, next_char = input_seq.to(self.device), next_char.to(self.device)
                    output = self.model(input_seq)
                    preds = torch.argmax(output, dim=1)
                    all_preds.extend([self.id2char[id] for id in preds.cpu().numpy()])
                    all_targets.extend([self.id2char[id] for id in next_char.cpu().numpy()])
            return all_targets, all_preds

        # Get predictions for validation and test sets
        val_targets, val_preds = get_predictions(self.val_loader)
        test_targets, test_preds = get_predictions(self.test_loader)

        # Compute accuracy
        val_accuracy = accuracy_score(val_targets, val_preds)
        test_accuracy = accuracy_score(test_targets, test_preds)

        result_path = Path('results', 'outputs', self.model_name)
        result_path.mkdir(exist_ok=True, parents=True)

        # Compute and plot confusion matrix
        def plot_confusion_matrix(targets, preds, set_name):
            cm = confusion_matrix(targets, preds, labels=list(CHAR2ID.keys()))
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CHAR2ID.keys(), yticklabels=CHAR2ID.keys())
            plt.title(f'Confusion Matrix - {set_name}')
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.savefig(result_path.joinpath(f'confusion_matrix_{set_name}.png'))
            plt.savefig(result_path)
            plt.close()

        plot_confusion_matrix(val_targets, val_preds, 'Validation')
        plot_confusion_matrix(test_targets, test_preds, 'Test')

        # Save classification reports
        val_report = classification_report(val_targets, val_preds, output_dict=True)
        test_report = classification_report(test_targets, test_preds, output_dict=True)
        pd.DataFrame(val_report).transpose().to_csv(result_path.joinpath('val_classification_report.csv'))
        pd.DataFrame(test_report).transpose().to_csv(result_path.joinpath('test_classification_report.csv'))

        # Save losses and accuracies
        results_df = pd.DataFrame({
            'epoch': range(1, len(self.train_losses) + 1),
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'val_accuracy': [val_accuracy] * len(self.train_losses),
            'test_accuracy': [test_accuracy] * len(self.train_losses)
        })
        results_df.to_csv(result_path.joinpath('training_results.csv'), index=False)

        print(f"Validation Accuracy: {val_accuracy}")
        print(f"Test Accuracy: {test_accuracy}")

        # Save predictions along with actual labels
        def save_predictions(targets, preds, set_name):
            predictions_df = pd.DataFrame({'Actual': targets, 'Predicted': preds})
            predictions_df.to_csv(result_path.joinpath(f'{set_name}_predictions.csv'), index=False)

        save_predictions(val_targets, val_preds, 'Validation')
        save_predictions(test_targets, test_preds, 'Test')


class SKLearnTrainer:
    def __init__(
            self, train_X, train_y, test_X, test_y, model_name, param_grid, ngram_range=(1, 2), cv=5, model_args=None,
            save_path=None,
    ):
        """
        Initializes the SKLearnTrainer class with data and training parameters.

        Args:
            train_X (Series): Training data features.
            train_y (Series): Training data labels.
            test_X (Series): Test data features.
            test_y (Series): Test data labels.
            model_name (str): Name of the model class to be used ('SVC' or 'MultinomialNB').
            param_grid (dict): Grid of parameters to search over.
            ngram_range (tuple): The ngram range for TF-IDF vectorization.
            cv (int): Number of folds for cross-validation.
        """
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.param_grid = param_grid
        self.ngram_range = ngram_range
        self.cv = cv
        self.model_args = model_args or {}
        self.save_path = save_path or Path("trained_models", model_name + "_best.pickle")

        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=self.ngram_range)

        # Dynamically load the model class
        if model_name == 'SVM':
            self.model_class = SVC
        elif model_name == 'MultinomialNB':
            self.model_class = MultinomialNB
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def vectorize_data(self):
        """
        Vectorizes the training and test data using TF-IDF.
        """
        self.train_X_tfidf = self.vectorizer.fit_transform(self.train_X)
        self.test_X_tfidf = self.vectorizer.transform(self.test_X)

    def train_and_tune(self):
        """
        Trains the model and tunes hyperparameters using GridSearchCV.
        """
        grid_search = GridSearchCV(
            self.model_class(**self.model_args, probability=True), self.param_grid, refit=True, verbose=3, cv=self.cv
        )
        grid_search.fit(self.train_X_tfidf, self.train_y)
        print("Best parameters found: ", grid_search.best_params_)
        self.best_model = grid_search.best_estimator_

    def train(self):
        model_args = self.model_args
        if hasattr(self.model_class, 'probability'):
            model_args["probability"] = True
        svm_best = self.model_class(**model_args)
        svm_best.fit(self.train_X_tfidf, self.train_y)
        self.best_model = svm_best

    def evaluate(self):
        """
        Evaluates the best model on the test set.
        """
        predictions = self.best_model.predict(self.test_X_tfidf)
        accuracy = sum(np.array(self.test_y) == predictions) / len(self.test_y)
        print("Accuracy:", accuracy)
        print(classification_report(self.test_y, predictions))

    def retrain_best_model(self):
        """
        Retrains the model with the best parameters found.
        """
        self.best_model.fit(self.train_X_tfidf, self.train_y)
        predictions = self.best_model.predict(self.test_X_tfidf)
        accuracy = sum(np.array(self.test_y) == predictions) / len(self.test_y)
        print("Retrained Model Accuracy:", accuracy)

    def save_model(self):
        import pickle
        # save
        with open(self.save_path, 'wb') as f:
            pickle.dump(self.best_model, f)
