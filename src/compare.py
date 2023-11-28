import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from src.constants import PROJECT_ROOT_PATH


def load_model_results(model_name):
    base_path = PROJECT_ROOT_PATH.joinpath("results", "outputs", model_name)
    accuracy_df = pd.read_csv(base_path.joinpath('training_results.csv'))
    val_report_df = pd.read_csv(base_path.joinpath('val_classification_report.csv'))
    test_report_df = pd.read_csv(base_path.joinpath('test_classification_report.csv'))
    return accuracy_df, val_report_df, test_report_df


# Plot Accuracy and Loss Over Epochs
def plot_accuracy_loss(model_results):
    fig, axes = plt.subplots(nrows=len(model_results), ncols=2, figsize=(15, len(model_results) * 5))
    for i, (model_name, (accuracy_df, _, _)) in enumerate(model_results.items()):
        sns.lineplot(data=accuracy_df, x='epoch', y='train_loss', ax=axes[i][0], label='Train Loss')
        sns.lineplot(data=accuracy_df, x='epoch', y='val_loss', ax=axes[i][0], label='Validation Loss')
        axes[i][0].set_title(f'{model_name} - Loss')

        sns.lineplot(data=accuracy_df, x='epoch', y='val_accuracy', ax=axes[i][1], label='Validation Accuracy')
        axes[i][1].set_title(f'{model_name} - Accuracy')

    plt.tight_layout()
    plt.savefig(PROJECT_ROOT_PATH.joinpath("results", "analysis", 'comparative_loss_accuracy.png'))
    plt.show()


# Plot Final Accuracy Comparison
def plot_final_accuracy(model_results, split_name="val"):
    accuracies = {model: df[f'{split_name}_accuracy'].iloc[-1] for model, (df, _, _) in model_results.items()}
    pd.Series(accuracies).plot(kind='bar')
    print(accuracies)
    plt.title('Final Validation Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.savefig(PROJECT_ROOT_PATH.joinpath("results", "analysis", f'final_accuracy_comparison_{split_name}.png'))
    plt.show()


def load_classification_report(model_name, set_name='val'):
    report_path = PROJECT_ROOT_PATH.joinpath('results', 'outputs', model_name, f'{set_name}_classification_report.csv')
    report_df = pd.read_csv(report_path)

    # Transpose the DataFrame to make 'precision', 'recall', 'f1-score' as rows
    report_df = report_df[['precision', 'recall', 'f1-score']].transpose()

    return report_df


def plot_classification_reports(classification_reports, set_name='val'):
    num_models = len(classification_reports)
    fig, axes = plt.subplots(nrows=num_models, figsize=(10, num_models * 4))

    for i, (model_name, reports) in enumerate(classification_reports.items()):
        sns.heatmap(reports[set_name], annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[i])
        axes[i].set_title(f'{model_name} - {set_name.capitalize()} Set Classification Report')

    plt.tight_layout()
    plt.savefig(PROJECT_ROOT_PATH.joinpath("results", "analysis", f'{set_name}_set_classification_reports_heatmap.png'))

    plt.show()


if __name__ == '__main__':
    model_names = ['CharRNN', 'CharLSTM', 'NextCharLSTM', 'CharCNN', 'CharCNNYLecun']
    model_results = {model: load_model_results(model) for model in model_names}
    plot_accuracy_loss(model_results)
    plot_final_accuracy(model_results)
    plot_final_accuracy(model_results, split_name="test")

    classification_reports = {
        model: {
            'val': load_classification_report(model, 'val'),
            'test': load_classification_report(model, 'test')
        } for model in model_names
    }

    plot_classification_reports(classification_reports, 'val')
    plot_classification_reports(classification_reports, 'test')
