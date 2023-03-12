import torch
from torch.utils.data import DataLoader
from transformers import AdapterConfig, BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset
import json
from sklearn.model_selection import train_test_split
import copy
from datetime import datetime

DATASET = 'RIKSDAGEN'
#DATASET = 'SNLI'

class RiksdagenDataset(Dataset):
    
    def __init__(self, filename, max_size=None):
        super().__init__()
        self.vocab_labels = {'KD': 0, 'SD': 1, 'S': 2, 'M': 3, 'V': 4, 'MP': 5, 'L': 6, 'C': 7}
        self.xs = []
        self.ys = []
        
        with open(filename, encoding="utf-8") as source_file:
            data = json.load(source_file)
            
            for i, idx in enumerate(data): 
                if max_size and i >= max_size:
                    break
                
                label = data[idx]['label']
                text = data[idx]['text']
                if label in self.vocab_labels:
                    self.xs.append(text)
                    self.ys.append(self.vocab_labels[label])            


    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]

    def __len__(self):
        return len(self.xs)

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

def BERT(device, train_dataset, test_dataset, swedish_bert, task):
    # Instantiate the tokenizer and the model
    if swedish_bert:        
        tokenizer = BertTokenizer.from_pretrained('KB/bert-base-swedish-cased', model_max_length=512)
        model = BertForSequenceClassification.from_pretrained('KB/bert-base-swedish-cased', num_labels=len(train_dataset.vocab_labels)).to(device)
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(train_dataset.vocab_labels)).to(device)

    #model.classifier.requires_grad = True # always, kanske är automatiskt men safear

    # task == 1: freeze bert model weights
    if task == 1:
        # Freeze all the parameters of the BERT model
        for param in model.bert.parameters():
            param.requires_grad = False
        model.classifier.requires_grad = True
        model.to(device)
        # Define the data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32)

    # task == 2: add adapters and freeze bert model weights
    elif task == 2:
        adapter_config = AdapterConfig.load("pfeiffer")
        adapter_config.model_type = "bert"
        adapter_config.task_name = "classification"
        # Nessesary for updating parameters in adapter
        #adapter_config.requires_grad = True
        model.bert.add_adapter("classification", config=adapter_config)
        # Freeze all the parameters of the BERT model
        for param in model.bert.parameters():
            param.requires_grad = False
        model.bert.train_adapter("classification")
        model.classifier.requires_grad = True
        model.to(device)
        # Define the data loaders
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=8)
    else:
        # Define the data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32)
        pass
    
    # Define the optimizer and the loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Train the model
    #model.train()

    train_t1 = datetime.now()
    
    for epoch in range(1):
        t1 = datetime.now()

        for batch_idx, (sent, label) in enumerate(train_loader):
            # Convert the data to tensor form    
            inputs = tokenizer(sent, padding='max_length', truncation=True, max_length = 512, return_tensors='pt').to(device)
           
            labels = torch.tensor(label).to(device)
            # Forward pass
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0 and batch_idx > 0:
                t2 = datetime.now()
                print(f'Epoch: {epoch}, Batch index: {batch_idx}, Loss: {loss.item()}, Deltatime: {t2-t1}')
                t1 = datetime.now()

                # Code for continously evaluating a statement during training
                #with torch.no_grad():
                #    sent = "<string>"
                #    inputs = tokenizer(sent, padding=True, truncation=True, return_tensors='pt').to(device)
                #    outputs = model(**inputs)
                #    predicted_labels = torch.argmax(outputs.logits, axis=1)
                #    # Print list of logits
                #    print(outputs.logits.tolist())
                    

    print(f"Total trainingtime: {datetime.now() - train_t1}")
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        correct = 0
        correct_top2 = 0
        correct_top3 = 0
        total = 0
        for sent, label in test_loader:
            # Convert the data to tensor form
            inputs = tokenizer(sent, padding=True, truncation=True, return_tensors='pt').to(device)
            labels = torch.tensor(label).to(device)
            # Compute the predicted labels
            outputs = model(**inputs)
            predicted_labels = torch.argsort(outputs.logits, descending=True)
            # Compute the accuracy
            total += labels.size(0)
            correct += (predicted_labels[:, 0] == labels).sum().item()
            correct_top2 += ((predicted_labels[:, 0] == labels) | (predicted_labels[:, 1] == labels)).sum().item()
            correct_top3 += ((predicted_labels[:, 0] == labels) | (predicted_labels[:, 1] == labels) | (predicted_labels[:, 2] == labels)).sum().item()
            #print(100*correct/total)
            print(f"Top-1 Accuracy: {100*correct/total:.2f}%, Top-2 Accuracy: {100*correct_top2/total:.2f}%, Top-3 Accuracy: {100*correct_top3/total:.2f}%")
    #print(f'Accuracy on the test set: {100 * correct / total:.2f}%')
    print(f'Accuracy on the test set: Top-1: {100 * correct / total:.2f}%, Top-2: {100 * correct_top2 / total:.2f}%, Top-3: {100 * correct_top3 / total:.2f}%')


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #device = torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu')         

    if DATASET == 'RIKSDAGEN':
        swedish_bert = True
        dataset = RiksdagenDataset('preprocessed_speeches.json', max_size=20000)
        
        X_train, X_test, Y_train, Y_test = train_test_split(dataset.xs, dataset.ys, test_size = 0.3)
        
        train_dataset = copy.deepcopy(dataset)
        test_dataset = copy.deepcopy(dataset)

        train_dataset.xs = X_train
        train_dataset.ys = Y_train
        test_dataset.xs = X_test
        test_dataset.ys = Y_test
    else:         
        swedish_bert = False
        train_dataset = SNLIDataset('News_Category_Dataset_v3.json', max_size=1000)
        test_dataset = SNLIDataset('News_Category_Dataset_v3.json', max_size=200)
    
    BERT(device, train_dataset, test_dataset, swedish_bert, task=2)

if __name__ == '__main__':
    main()

# task = 0: finetuning bertmodel and training classification layer
# task = 1: only training classification layer (freeze bert)
# task = 2: implement and train adapter weights, freeze bert weights and train classification layer

"""
|Training samples | Test samples | Time | Accuracy | Freeze |
|-----------------------------------------------------------| 
|    2000         |  500         |   ?  |  43.80%  | False  |
|    10 000       |  2000        |   ?  |  66.50%  | False  |
|    10 000       |  2000        |   ?  |  25.50%  | True   |
|    50 000       |  5000        |   ?  |  38.04%  | True   |
"""