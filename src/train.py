import argparse
import json

import torch

from src.constants import PROJECT_ROOT_PATH, CHAR2ID, CHAR2ID_WITH_PAD, ID2CHAR, ID2CHAR_WITH_PAD
from src.data_loader import get_processed_train_test
from src.dataset import get_train_valid_test_loaders
from src.trainer import DLTrainer, SKLearnTrainer


def validate_params(model_name, provided_args, required_params):
    # Filter provided arguments to include only those that are required for the model
    filtered_args = {param: getattr(provided_args, param) for param in required_params}

    # Check for missing parameters
    missing_params = [param for param, value in filtered_args.items() if value is None]
    if missing_params:
        print(f"Missing parameters for {model_name}: {', '.join(missing_params)}")
        print(f"Required parameters for {model_name}: {', '.join(required_params)}")
        exit(1)

    return filtered_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model name')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.0003)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--early_stopping_patience', type=int, default=7)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    # Arguments for Deep Learning models
    # parser.add_argument('--input_size', type=int)
    parser.add_argument('--hidden_size', type=int)
    # parser.add_argument('--output_size', type=int)
    parser.add_argument('--num_hidden_layers', type=int)
    parser.add_argument('--embedding_dim', type=int)
    parser.add_argument('--dropout_rate', type=float)
    # parser.add_argument('--vocab', type=int)
    parser.add_argument('--max_seq_length', type=int)
    # parser.add_argument('--vocab_size', type=int)
    # parser.add_argument('--num_output_classes', type=int)
    parser.add_argument('--num_conv_filters', type=int)
    parser.add_argument('--conv_kernel_size', type=int)

    # Arguments for sklearn models
    parser.add_argument('--cv', type=float, default=5)
    parser.add_argument('--C', type=float, default=0.8)
    parser.add_argument('--ngram_range', type=float, default=(1, 2))
    parser.add_argument('--kernel', type=str, default='rbf')
    parser.add_argument('--alpha', type=float, default=1.0)

    args = parser.parse_args()

    # Define required parameters for each model
    model_params = json.load(open(PROJECT_ROOT_PATH.joinpath("results", "model_hyperparams.json")))

    # Validation and Trainer selection
    if args.model in model_params["dl"]:
        id2char = ID2CHAR
        char2id = CHAR2ID

        if args.model in ["NextCharLSTM"]:
            target_as_sequence = True
            model_args = model_params["dl"][args.model]

        else:
            target_as_sequence = False
            model_args = model_params["dl"][args.model]
            id2char = ID2CHAR_WITH_PAD
            char2id = CHAR2ID_WITH_PAD
            # TODO: add conditions as needed

        validated_model_args = validate_params(args.model, args, model_args)

        # Assuming get_train_valid_test_loaders returns appropriate DataLoader objects
        train_loader, val_loader, test_loader = get_train_valid_test_loaders(args.batch_size, target_as_sequence)

        trainer = DLTrainer(
            model_name=args.model,
            model_args=validated_model_args,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            early_stopping_patience=args.early_stopping_patience,
            weight_decay=args.weight_decay,
            checkpoint_path=args.checkpoint_path if hasattr(args, 'checkpoint_path') else None,
            id2char=id2char,
            char2id=char2id,
        )
        trainer.train()
        trainer.evaluate()
        trainer.save_model()


    elif args.model in model_params["sklearn"]:
        model_args = validate_params(args.model, args, model_params["sklearn"][args.model])

        # Load your training and testing data here
        train_X, train_y, test_X, test_y = get_processed_train_test()

        trainer = SKLearnTrainer(
            train_X=train_X,
            train_y=train_y,
            test_X=test_X,
            test_y=test_y,
            model_name=args.model,
            param_grid=model_args,
            ngram_range=args.ngram_range,
            cv=args.cv,
            model_args=model_args
        )
        trainer.vectorize_data()
        trainer.train()
        trainer.evaluate()
        trainer.retrain_best_model()
    else:
        available_models = list(model_params["dl"].keys()) + list(model_params["sklearn"].keys())
        print(f"Model {args.model} not recognized. Available models are: {available_models}")
        exit(1)

    # Start training
    trainer.save_model()
