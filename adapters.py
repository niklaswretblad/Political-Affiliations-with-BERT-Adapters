import torch
from torch.utils.data import DataLoader
from transformers import AdapterType, AdapterConfig, AutoConfig, AutoModelWithHeads, AdamW

from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import Dataset
import json
#from transformers.adapters import AdapterType, AdapterConfig, AdapterHelper, AdapterTrainer

class SNLIDataset(Dataset):

    def __init__(self, filename, max_size=None):
        super().__init__()
        self.xs = []
        self.ys = []
        with open(filename) as source:
            for i, line in enumerate(source):
                if max_size and i >= max_size:
                    break
                data = json.loads(line)
                sentence = data['headline']
                label = data['category']

                self.xs.append(sentence)
                self.ys.append(['POLITICS', 'WELLNESS', 'ENTERTAINMENT', 'TRAVEL', 'STYLE & BEAUTY',
                'PARENTING', 'HEALTHY LIVING', 'QUEER VOICES', 'FOOD & DRINK', 'BUSINESS', 'CRIME',
                'COMEDY', 'SPORTS', 'BLACK VOICES', 'HOME & LIVING', 'PARENTS', 'WORLD NEWS', 'SCIENCE',
                'U.S. NEWS', 'CULTURE & ARTS', 'TECH', 'WEIRD NEWS', 'ENVIRONMENT', 'EDUCATION', 'MEDIA', 
                'WOMEN', 'MONEY', 'RELIGION', 'LATINO VOICES', 'IMPACT', 'WEDDINGS', 'COLLEGE',
                'ARTS & CULTURE', 'STYLE', 'GREEN', 'TASTE', 'THE WORLDPOST', 'GOOD NEWS',
                'WORLDPOST', 'FIFTY', 'ARTS'].index(label))
                # print('xs: ', self.xs[i])
                # print('ys: ', self.ys[i])

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]

    def __len__(self):
        return len(self.xs)

# def add_adapter(model):
#     config = AdapterConfig.load("pfeiffer")
#     model.add_adapter("classification", AdapterType.text_task, config=config)
#     model.train_adapter(["classification"])
#     return model

def BERT(device, train_dataset, test_dataset, freeze_bert=False):
    # Instantiate the tokenizer and the model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=41).to(device)

    # Load the adapter configuration named "pfeiffer" using the 
    # AdapterConfig.load() method and assign it to a variable called adapter_config.
    adapter_config = AdapterConfig.load("pfeiffer")
    # Set the model type to "bert" for the adapter_config 
    # using the model_type attribute.
    adapter_config.model_type = "bert"
    # Set the task name to "classification" for the 
    # adapter_config using the task_name attribute.
    adapter_config.task_name = "classification"
    adapter_config.requires_grad = True
    # Add a new adapter to the model with the name "classification", using
    #  the model.add_adapter() method, and configure it with the adapter_config object.
    model.bert.add_adapter("classification", config=adapter_config)
    # Train the newly added adapter with the name "classification" 
    # using the model.train_adapter() method.
    # This also freezes the model.
    model.bert.train_adapter("classification")
    
    # Unfreeze the classifier
    model.classifier.requires_grad = True

    model.to(device)

    # optimizer = AdamW([
    #     {'params': model.get_adapter("classification").parameters(), 'lr': 1e-3},
    #     {'params': model.classifier.parameters(), 'lr': 1e-3},
    # ]) 
    # Define the optimizer and the loss function
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-5)
    adapter_optimizer = torch.optim.AdamW(model.bert.encoder.parameters(), lr=1e-4)

    # Define the data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Train the model
    #model.train()

    for epoch in range(1):
        for batch_idx, (sent, label) in enumerate(train_loader):
            # Convert the data to tensor form
            inputs = tokenizer(sent, padding=True, truncation=True, return_tensors='pt').to(device)
            labels = torch.tensor(label).to(device)

            # Forward pass
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            # Backward pass
            optimizer.zero_grad()
            adapter_optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            adapter_optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch index: {batch_idx}, Loss: {loss.item()}')

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for sent, label in test_loader:
            # Convert the data to tensor form
            inputs = tokenizer(sent, padding=True, truncation=True, return_tensors='pt').to(device)
            labels = torch.tensor(label).to(device)
            # Compute the predicted labels
            outputs = model(**inputs)
            predicted_labels = torch.argmax(outputs.logits, axis=1)
            # Compute the accuracy
            total += labels.size(0)
            correct += (predicted_labels == labels).sum().item()
            print(100*correct/total)
    print(f'Accuracy on the test set: {100 * correct / total:.2f}%')

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_dataset = SNLIDataset('News_Category_Dataset_v3.json', max_size=50000)
    test_dataset = SNLIDataset('News_Category_Dataset_v3.json', max_size=2500)

    BERT(device, train_dataset, test_dataset, freeze_bert=True)

if __name__ == '__main__':
    main()

"""
|Training samples | Test samples | Time | Accuracy | Freeze |
|-----------------------------------------------------------| 
|    2000         |  500         |   ?  |  43.80%  | False  |
|    10 000       |  2000        |   ?  |  66.50%  | False  |
|    10 000       |  2000        |   ?  |  25.50%  | True   |
|    50 000       |  5000        |   ?  |  38.04%  | True   |
"""