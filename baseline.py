import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import Dataset

class SNLIDataset(Dataset):

    def __init__(self, filename, max_size=None):
        super().__init__()
        self.xs = []
        self.ys = []
        with open(filename) as source:
            for i, line in enumerate(source):
                if max_size and i >= max_size:
                    break
                sentence1, sentence2, gold_label = line.rstrip().split('\t')
                self.xs.append((sentence1, sentence2))
                self.ys.append(['contradiction', 'entailment', 'neutral'].index(gold_label))

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]

    def __len__(self):
        return len(self.xs)

def BERT(device, train_dataset, test_dataset):
    # Instantiate the tokenizer and the model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3).to(device)

    # Define the optimizer and the loss function
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # Define the data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Train the model
    model.train()

    for epoch in range(1):
        for batch_idx, ((s1, s2), label) in enumerate(train_loader):
            # Convert the data to tensor form
            inputs = tokenizer(s1, s2, padding=True, truncation=True, return_tensors='pt').to(device)
            labels = torch.tensor(label).to(device)

            # Forward pass
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch index: {batch_idx}, Loss: {loss.item()}')

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for (s1, s2), label in test_loader:
            # Convert the data to tensor form
            inputs = tokenizer(s1, s2, padding=True, truncation=True, return_tensors='pt').to(device)
            labels = torch.tensor(label).to(device)
            # Compute the predicted labels
            outputs = model(**inputs)
            predicted_labels = torch.argmax(outputs.logits, axis=1)
            # Compute the accuracy
            total += labels.size(0)
            correct += (predicted_labels == labels).sum().item()
            print(100*correct/total)
    print(f'Accuracy on the test set: {100 * correct / total:.2f}%')
    #return 100*correct/total

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_dataset = SNLIDataset('snli_1.0_train_preprocessed.txt', max_size=100)
    test_dataset = SNLIDataset('snli_1.0_test_preprocessed.txt')
    BERT(device, train_dataset, test_dataset)