import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_loader import ReportsDataset
from models import ESGClassifier, sentence_selection

reportdata = ReportsDataset(pd.read_pickle("full_data.pkl"))
train_loader = DataLoader(reportdata, batch_size=1, shuffle=True)
print('padding_lenth', reportdata[0][0].shape[0])
model = ESGClassifier(384, reportdata[0][0].shape[0], 9)


# model.load_state_dict(torch.load('classifier.pt'))


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)


model.apply(init_weights)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 200

epoch_losses = []
accuracy = []
for epoch in range(num_epochs):
    num_correct = 0
    training_loss = 0
    for data in train_loader:
        inputs, labels = data
        labels = labels.long()
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, labels)
        if torch.argmax(outputs) == labels:
            num_correct += 1
        training_loss += loss.item()
        loss.backward()
        optimizer.step()

    epoch_loss = training_loss / len(train_loader)
    epoch_losses.append(epoch_loss)
    accuracy.append(num_correct / len(train_loader))

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_losses[-1]}, Accuracy: {accuracy[-1]}')

    if epoch_loss <= min(epoch_losses):
        torch.save(model.state_dict(), 'classifier.pt')
        print(f"model saved with loss {epoch_losses[-1]} and accuracy {accuracy[-1]}")

# draw the training loss and accuracy
fig, ax1 = plt.subplots(figsize=(10, 5))

color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color=color)
ax1.plot(epoch_losses, label="Loss", color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Accuracy', color=color)
ax2.plot(accuracy, label="Accuracy", color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title("Training Loss and Accuracy")
fig.tight_layout()
plt.show()


# print(reportdata.df.iloc[0])
# r1 = sentence_selection(model, reportdata.df.iloc[0])
# print(r1)

test = model(reportdata[0][0].unsqueeze(0))
print(test)
