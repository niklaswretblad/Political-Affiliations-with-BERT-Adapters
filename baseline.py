import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, AutoConfig, Trainer, TrainingArguments, default_data_collator
from transformers.adapters import PfeifferConfig, BertAdapterModel, setup_adapter_training, AdapterArguments, AdapterTrainer
from torch.utils.data import Dataset
#import evaluate
import json


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
                'WORLDPOST', 'FIFTY', 'ARTS', 'DIVORCE'].index(label))
                # print('xs: ', self.xs[i])
                # print('ys: ', self.ys[i])

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]

    def __len__(self):
        return len(self.xs)

def BERT(device, train_dataset, test_dataset, freeze_bert=False):
    # Instantiate the tokenizer and the model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=42).to(device)

    config = AutoConfig.from_pretrained(
        "bert-base-uncased",
        num_labels=42,
    )

    model = BertAdapterModel.from_pretrained(
        'bert-base-uncased',
        config = config,
    )
    model.add_classification_head('classification', num_labels=42)
    
    # Freeze the BERT model if required
    if freeze_bert:
        # Freeze all the parameters of the BERT model
        for param in model.base_model.parameters():
            param.requires_grad = False

    # Define the optimizer and the loss function
    #optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-4)

    #metric = evaluate.load("accuracy")

    # Define the data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    adapter_args = AdapterArguments()
    adapter_args.train_adapter = True
    adapter_args.adapter_config = "pfeiffer"

    setup_adapter_training(model, adapter_args, adapter_name = "classification")

    #task_name = "adapter"
    # resolve the adapter config
    #adapter_config = PfeifferConfig()
    # add a new adapter
    #model.add_adapter(task_name, config=adapter_config)
    # Enable adapter training
    #model.train_adapter(task_name)

    training_args = TrainingArguments(output_dir = "/tmp/bert",
    #training_args = training_args(
        do_train = True,
        do_eval = False,
        #max_seq_length = 512,
        learning_rate = 1e-4,
        num_train_epochs = 1,
    )

    trainer_class = AdapterTrainer if adapter_args.train_adapter else Trainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=test_dataset if training_args.do_eval else None,
        #compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    #model.set_active_adapters(task_name)

    model.to(device)

    # Train the model

    if training_args.do_train:
        #checkpoint = None
        #if training_args.resume_from_checkpoint is not None:
        #    checkpoint = training_args.resume_from_checkpoint
        #elif last_checkpoint is not None:
        #    checkpoint = last_checkpoint
        train_result = trainer.train()
        #metrics = train_result.metrics
        #max_train_samples = (
        #    data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        #)
        #metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        #trainer.save_model()  # Saves the tokenizer too for easy upload

        #trainer.log_metrics("train", metrics)
        #trainer.save_metrics("train", metrics)
        trainer.save_state()

    #model.train()

    #for epoch in range(1):
    #    for batch_idx, (sent, label) in enumerate(train_loader):
    #        # Convert the data to tensor form
    #        inputs = tokenizer(sent, padding=True, truncation=True, return_tensors='pt').to(device)
    #        labels = torch.tensor(label).to(device)
    #
    #        # Forward pass
    #        outputs = model(**inputs, labels=labels)
    #        loss = outputs.loss
    #        # Backward pass
    #        optimizer.zero_grad()
    #        loss.backward()
    #        optimizer.step()
    #        if batch_idx % 100 == 0:
    #            print(f'Epoch: {epoch}, Batch index: {batch_idx}, Loss: {loss.item()}')

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
    train_dataset = SNLIDataset('News_Category_Dataset_v3.json', max_size=20000)
    test_dataset = SNLIDataset('News_Category_Dataset_v3_test.json', max_size=2000)
    BERT(device, train_dataset, test_dataset, freeze_bert=False)

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