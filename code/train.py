# Ignore future warnings
import warnings
from copy import deepcopy

import numpy as np
import torch
from matplotlib import pyplot as plt

import dpp
from visualize import visualize

warnings.simplefilter(action='ignore', category=FutureWarning)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

# Config
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
dataset_name = 'BPI_Challenge_2017_1k_1.pkl'  # run dpp.data.list_datasets() to see the list of available datasets

# Model config
context_size = 64  # Size of the RNN hidden vector
mark_embedding_size = 32  # Size of the mark embedding (used as RNN input)
num_mix_components = 64  # Number of components for a mixture model
rnn_type = "GRU"  # What RNN to use as an encoder {"RNN", "GRU", "LSTM"}

# Training config
batch_size = 64  # Number of sequences in a batch
regularization = 1e-5  # L2 regularization parameter
learning_rate = 1e-3  # Learning rate for Adam optimizer
max_epochs = 1000  # For how many epochs to train
display_step = 50  # Display training statistics after every display_step
patience = 50  # After how many consecutive epochs without improvement of val loss to stop training

# Load the data
dataset = dpp.data.load_dataset(dataset_name)
d_train, d_val, d_test = dataset.train_val_test_split(seed=None, shuffle=False)
visualize(d_train, d_val, d_test)
# Create the model
dl_train = d_train.get_dataloader(batch_size=batch_size, shuffle=True)
dl_val = d_val.get_dataloader(batch_size=batch_size, shuffle=False)
dl_test = d_test.get_dataloader(batch_size=batch_size, shuffle=False)

# Define the model
print('Building model...')
mean_log_inter_time, std_log_inter_time = d_train.get_inter_time_statistics()

model = dpp.models.LogNormMix(
        num_marks=d_train.num_marks,
        mean_log_inter_time=mean_log_inter_time,
        std_log_inter_time=std_log_inter_time,
        context_size=context_size,
        mark_embedding_size=mark_embedding_size,
        rnn_type=rnn_type,
        num_mix_components=num_mix_components,
)
opt = torch.optim.Adam(model.parameters(), weight_decay=regularization, lr=learning_rate)

# Training
print('Starting training...')


def aggregate_loss_over_dataloader(dl):
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in dl:
            total_loss += -model.log_prob(batch).sum()
            total_count += batch.size
    return total_loss / total_count


def evaluate_mark_over_dataloader(dl, metric):
    total_accuracy = 0.0
    total_batch = 0
    with torch.no_grad():
        for batch in dl:
            total_accuracy += model.evaluate_mark(batch, metric)
            total_batch += 1
    error = total_accuracy / total_batch
    # Convert unix timestamp to datetime

    return error


def evaluate_timestamp_over_dataloader(dl):
    total_mae = 0.0
    total_batch = 0
    with torch.no_grad():
        for batch in dl:
            total_mae += model.evaluate_timestamp(batch)
            total_batch += 1
    return total_mae / total_batch


impatient = 0
best_loss = np.inf
best_model = deepcopy(model.state_dict())
training_val_losses = []

for epoch in range(max_epochs):
    model.train()
    for batch in dl_train:
        opt.zero_grad()
        loss = -model.log_prob(batch).mean()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        loss_val = aggregate_loss_over_dataloader(dl_val)
        training_val_losses.append(loss_val)

    if (best_loss - loss_val) < 1e-4:
        impatient += 1
        if loss_val < best_loss:
            best_loss = loss_val
            best_model = deepcopy(model.state_dict())
    else:
        best_loss = loss_val
        best_model = deepcopy(model.state_dict())
        impatient = 0

    if impatient >= patience:
        print(f'Breaking due to early stopping at epoch {epoch}')
        break

    if epoch % display_step == 0:
        print(f"Epoch {epoch:4d}: loss_train_last_batch = {loss.item():.1f}, loss_val = {loss_val:.1f}")

# Evaluation
model.load_state_dict(best_model)
model.eval()

# All training & testing sequences stacked into a single batch
with torch.no_grad():
    final_loss_train = aggregate_loss_over_dataloader(dl_train)
    final_loss_val = aggregate_loss_over_dataloader(dl_val)
    final_loss_test = aggregate_loss_over_dataloader(dl_test)
    final_acc_train = evaluate_mark_over_dataloader(dl_train, 'acc')
    final_acc_val = evaluate_mark_over_dataloader(dl_val, 'acc')
    final_acc_test = evaluate_mark_over_dataloader(dl_test, 'acc')
    final_f1_train = evaluate_mark_over_dataloader(dl_train, 'f1')
    final_f1_val = evaluate_mark_over_dataloader(dl_val, 'f1')
    final_f1_test = evaluate_mark_over_dataloader(dl_test, 'f1')
    final_mae_train = evaluate_timestamp_over_dataloader(dl_train)
    final_mae_val = evaluate_timestamp_over_dataloader(dl_val)
    final_mae_test = evaluate_timestamp_over_dataloader(dl_test)


print(f'Negative log-likelihood:\n'
      f' - Train: {final_loss_train:.1f}\n'
      f' - Val:   {final_loss_val:.1f}\n'
      f' - Test:  {final_loss_test:.1f}\n'
      f' - Train Acc: {final_acc_train * 100}\n'
      f' - Val Acc:   {final_acc_val * 100}\n'
      f' - Test Acc:  {final_acc_test * 100}\n'
      f' - Train F1: {final_f1_train * 100}\n'
      f' - Val F1:   {final_f1_val * 100}\n'
      f' - Test F1:  {final_f1_test * 100}\n'
      f' - Train MAE: {final_mae_train}\n'
      f' - Val MAE:   {final_mae_val}\n'
      f' - Test MAE:  {final_mae_test}\n')

sampled_batch = model.sample(t_end=100000, batch_size=100)
real_batch = dpp.data.Batch.from_list([s for s in dataset])
plt.hist(sampled_batch.mask.sum(-1).cpu().numpy(), 50, label="Sampled", density=True, range=(0, 300))
plt.hist(real_batch.mask.sum(-1).cpu().numpy(), 50, alpha=0.3, label="Real data", density=True, range=(0, 300))
plt.xlabel("Sequence length")
plt.ylabel("Frequency")
plt.legend()
plt.show()
