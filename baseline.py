import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import Dataset
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
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
        
        with open(filename, encoding='utf-8') as source_file:
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
        self.vocab_labels = { 'POLITICS': 0, 'WELLNESS': 1, 'ENTERTAINMENT': 2, 'TRAVEL': 3, 'STYLE & BEAUTY': 4,
                'PARENTING': 5, 'HEALTHY LIVING': 6, 'QUEER VOICES': 7, 'FOOD & DRINK': 8, 'BUSINESS': 9, 'CRIME': 10,
                'U.S. NEWS': 11, 'CULTURE & ARTS': 12, 'TECH': 13, 'WEIRD NEWS': 14, 'ENVIRONMENT': 15, 'EDUCATION': 16, 'MEDIA': 17, 
                'WOMEN': 18, 'MONEY': 19, 'RELIGION': 20, 'LATINO VOICES': 21, 'IMPACT': 22, 'WEDDINGS': 23, 'COLLEGE': 24,
                'ARTS & CULTURE': 25, 'STYLE': 26, 'GREEN': 27, 'TASTE': 28, 'THE WORLDPOST': 29, 'GOOD NEWS': 30,
                'WORLDPOST': 31, 'FIFTY': 32, 'ARTS': 33, 'COMEDY': 34, 'SPORTS': 35, 'BLACK VOICES': 36, 'HOME & LIVING': 37, 
                'PARENTS': 38, 'WORLD NEWS': 39, 'SCIENCE': 40 }
        
        with open(filename) as source:
            for i, line in enumerate(source):
                if max_size and i >= max_size:
                    break
                data = json.loads(line)
                sentence = data['headline']
                label = data['category']

                self.xs.append(sentence)                
                self.ys.append(self.vocab_labels[label])
                # print('xs: ', self.xs[i])
                # print('ys: ', self.ys[i])

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]

    def __len__(self):
        return len(self.xs)

def BERT(device, train_dataset, test_dataset, swedish_bert, freeze_bert=False):
    if swedish_bert:        
        tokenizer = BertTokenizer.from_pretrained('KB/bert-base-swedish-cased', model_max_length=512)
        model = BertForSequenceClassification.from_pretrained('KB/bert-base-swedish-cased', num_labels=len(train_dataset.vocab_labels)).to(device)
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(train_dataset.vocab_labels)).to(device)

    if freeze_bert:
        for param in model.base_model.parameters():
            param.requires_grad = False

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-5)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model.train()
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

    print(f"Total trainingtime: {datetime.now() - train_t1}")
    
    # Evaluate the model
    true_labels = []
    predicted_labels_list = []
    
    model.eval()
    with torch.no_grad():
        for sent, label in test_loader:
            # Convert the data to tensor form
            inputs = tokenizer(sent, padding=True, truncation=True, return_tensors='pt').to(device)
            labels = torch.tensor(label).to(device)
            # Compute the predicted labels
            outputs = model(**inputs)
            predicted_labels = torch.argmax(outputs.logits, axis=1)
            # Append to the lists
            true_labels.extend(label)
            predicted_labels_list.extend(predicted_labels.cpu().numpy())

    print(f'Accuracy on the test set: {accuracy_score(true_labels, predicted_labels_list):.2f}%')
    print(classification_report(true_labels, predicted_labels_list, target_names=list(train_dataset.vocab_labels.keys())))


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #device = torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu')         

    if DATASET == 'RIKSDAGEN':
        swedish_bert = True
        dataset = RiksdagenDataset('preprocessed_speeches.json', max_size=5000)
        
        X_train, X_test, Y_train, Y_test = train_test_split(dataset.xs, dataset.ys, test_size = 0.3)
        
        train_dataset = copy.deepcopy(dataset)
        test_dataset = copy.deepcopy(dataset)

        train_dataset.xs = X_train
        train_dataset.ys = Y_train
        test_dataset.xs = X_test
        test_dataset.ys = Y_test
    else:         
        swedish_bert = False
        train_dataset = SNLIDataset('News_Category_Dataset_v3.json', max_size=50000)
        test_dataset = SNLIDataset('News_Category_Dataset_v3.json', max_size=5000)
    
    BERT(device, train_dataset, test_dataset, swedish_bert, freeze_bert=True)

if __name__ == '__main__':
    main()
