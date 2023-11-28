import random
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT_PATH: Path = Path(__file__).parents[1]
RANDOM_STATE = 42

VOCAB = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567'

ID2CHAR = {idx: char for idx, char in enumerate(VOCAB)}
CHAR2ID = {char: idx for idx, char in ID2CHAR.items()}

CHAR2ID_WITH_PAD = CHAR2ID.copy()
CHAR2ID_WITH_PAD['<PAD>'] = len(CHAR2ID)

ID2CHAR_WITH_PAD = {idx: char for char, idx in CHAR2ID_WITH_PAD.items()}

torch.manual_seed(RANDOM_STATE)
torch.cuda.manual_seed_all(RANDOM_STATE)  # For multi-GPU.
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
