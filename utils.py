import random, os, torch
import numpy as np
from torch import nn

def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seeding everything to seed {seed}")
    return


# custom early stopping, based on chosen metric
class EarlyStopping:
    def __init__(self, patience=5, delta=1e-6):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        elif current_score > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = current_score
            self.counter = 0

class ClassificationModel(nn.Module):
  def __init__(self, input_dim, num_classes):
    super(ClassificationModel, self).__init__()
    # Define layers of your model architecture
    self.fc1 = nn.Linear(input_dim, 128)  # Adjust input size and hidden layer size based on your data complexity
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(p=0.2)  # Optional dropout for regularization
    self.fc2 = nn.Linear(128, num_classes)

  def forward(self, x):
    x = self.fc1(x)
    x = self.relu(x)
    x = self.dropout(x)  # Apply dropout if enabled
    x = self.fc2(x)
    return x


def save_model(model, encoder, epoch, num_classes, lr, lr_factor, save_path, fname):
    save_dict = {"model_state_dict": model.state_dict(),
                 "encoder": encoder,
                 "num_classes": num_classes,
                 "epoch": epoch,
                 "lr": lr,
                 "lr_scheduler_factor": lr_factor}
    torch.save(save_dict, save_path/f"{fname}.pth")