import torch
import torch.nn as nn

class ASLBiLSTM(nn.Module):
    def __init__(self, input_size=84, hidden_size=128, num_classes=10, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            batch_first=True,
            bidirectional=True
        )
        self.layernorm = nn.LayerNorm(hidden_size * 2)  # ✅ add this
        self.dropout = nn.Dropout(dropout)               # ✅ add this
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.layernorm(out[:, -1, :])              # ✅ add this
        out = self.dropout(out)                          # ✅ add this
        return self.fc(out)


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu")
    labels = checkpoint["labels"]

    model = ASLBiLSTM(num_classes=len(labels))  # ✅ don't pass input_size/hidden_size as positional
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, labels