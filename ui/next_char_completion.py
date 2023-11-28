import streamlit as st
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

vocab = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567'

id2char = {idx: char for idx, char in enumerate(vocab)}

char2id = {char: idx for idx, char in id2char.items()}

char2id['<PAD>'] = len(char2id)

# Assuming CharacterLSTM and other necessary classes and functions are defined above or imported
vocab_size = len(char2id)
hidden_size = 128
embedding_dim = 100
num_layers = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CharacterLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim, num_layers):
        super(CharacterLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        final_output = self.fc(lstm_out[:, -1, :])
        return final_output


# Initialize your model (and other necessary components)
# Make sure the model is loaded with the trained weights and set to eval mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CharacterLSTM(vocab_size, hidden_size, embedding_dim, num_layers)
model.load_state_dict(torch.load("./trained_models/next_char_completion_lstm.pth"))

model = model.to(device)
model.eval()


def generate_next_char(model, start_seq):
    model.eval()
    input_seq = torch.tensor([char2id[char] for char in start_seq], dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_seq)
        predicted_char_id = output.argmax(dim=1).item()
    return id2char[predicted_char_id]



# Streamlit UI
st.title("Character Prediction App")

# Text input for the user sequence
user_input = st.text_input("Enter a sequence of characters:", "")

# Displaying the next predicted character
if user_input:
    next_char = generate_next_char(model, user_input,)
    st.write(f"Suggested next character: {user_input}->{next_char}")
else:
    st.write("Start typing to see the next character suggestion.")
