.PHONY: help setup test clean

help:
	@echo "make setup - Create a Conda environment and install dependencies"
	@echo "make test - Run unit tests"
	@echo "make clean - Remove all temporary files and delete the Conda environment"

setup:
	@echo "Creating the Conda environment..."
	conda create --name botNextCharPred python=3.11 --yes
	@echo "Activating the Conda environment..."
	. $${HOME}/miniconda3/etc/profile.d/conda.sh && conda activate botNextCharPred
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	export PYTHONPATH=${PWD}:${PYTHONPATH}

test:
	@echo "Running tests..."
	python -m unittest discover -s tests

train_all:
	@echo "Running tests..."
	python src/train.py --model "SVM" --num_epochs 1
	python src/train.py --model "MultinomialNB" --num_epochs 1
	python src/train.py --model "CharRNN" --num_epochs 1
	python src/train.py --model "CharLSTM" --num_epochs 1
	python src/train.py --model "CharCNN" --num_epochs 1
	python src/train.py --model "CharCNNYLecun" --num_epochs 1
	python src/train.py --model "NextCharLSTM" --num_epochs 1

clean:
	@echo "Cleaning up..."
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	. $${HOME}/miniconda3/etc/profile.d/conda.sh && conda env remove --name botNextCharPred
