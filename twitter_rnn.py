import numpy as np
import pandas as pd
import spacy
import csv
import pickle, json
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.distributions as tdist
from vocab import Vocab
from IPython.display import clear_output
import matplotlib.pyplot as plt
import time
from nltk import (sent_tokenize as splitter, wordpunct_tokenize as tokenizer)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from embeddings import Embeddings


class RNN(nn.Module):
        def __init__(self, vocab_size, embedding_dim,
                     hidden_dim, output_dim, pad_idx):

            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)        
            self.rnn = nn.LSTM(embedding_dim, 
                               hidden_dim,
                               batch_first=True)

            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, text, text_lengths):
            embedded = self.embedding(text)
            #print('embedding dim', embedded.size())
            if text_lengths == 0:
                return
            #packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True)
            packed_output, (hidden, cell) = self.rnn(embedded) # size (4, 1, 256)
#             hidden = hidden.squeeze(1) # size (4, 256)
#             print('hidden dims1', hidden.size())
#             hidden = hidden.view(1, -1) 
#             print('hidden dims2', hidden.size()) # size (1, 1024)
            #print('hidden dims1 before cat', hidden.size())
            
            #print('hidden[-2,:,:].size()', hidden[-2,:,:].size())
            #print('hidden[-1,:,:].size()', hidden[-1,:,:].size())
            #hidden = torch.cat((hidden[-4,:,:], hidden[-3, :,:], hidden[-2,:,:], hidden[-1, :,:]), dim = 1)

            #print('hidden dims1 after cat', hidden.size())
            #print('hidden.squeeze(0)', hidden.squeeze(0).size())
            return self.fc(hidden.squeeze(0))
        
        
class SentimentClassifier(object): 
    def __init__(self):
        self.dataset = self.read_json('twitter_prep_data.json') 
        self.word_list = []
        self.d = self.read_json('embeddings_dict.json') 
        
    def read_pickle(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            dataset = pd.DataFrame.from_dict(data) 
        return dataset
    
    def read_json(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            dataset = pd.DataFrame.from_dict(data) 
        return dataset

# dump to pickle
    def create_vocab(self):
        for s in tqdm(self.dataset['text'].values):
            self.word_list += s
        word_counter = Counter(self.word_list)
        vocab = Vocab(word_counter, min_freq=10)
        return vocab


    class TwitterDataset(Dataset):
        def __init__(self, data, vocab):
            self.vocab = vocab
            self.data = data
            self.text = self.data['text'].values
            self.label = self.data['label'].values

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            text = self.text[index]
            label = self.label[index]
            text = self.vocab.sent2idx(text)
            sample = {'label': label, 'text': text}
            return sample

# torch pad read

        def collate_fn(self, dicts): 
            pad_token = 0
            sents_padded = []
            corpus_size = len(dicts)
            len_text_list = [len(d['text']) for d in dicts]
            text_list = [d['text'] for d in dicts]
            labels = [i['label'] for i in dicts]

            sorted_len_text, sorted_text, sorted_labels = list(zip(*sorted(zip(len_text_list, text_list, labels), key=lambda x: x[0] ,reverse=True))) #sorts sentences in the reverse hierarchical order        
            max_lens = sorted_len_text[0]

            text_padded = [sorted_text[i] + [pad_token] * (max_lens - sorted_len_text[i]) for i in range(corpus_size)]
            text_padded = torch.LongTensor(text_padded)
            labels = torch.FloatTensor(sorted_labels)

            return text_padded, labels, sorted_len_text


    def create_train_dataset(self):
        X_train, X_test = train_test_split(self.dataset, test_size=0.33, random_state=42)
        vocab = self.create_vocab()
        train_dataset = SentimentClassifier.TwitterDataset(X_train, vocab)
        test_dataset = SentimentClassifier.TwitterDataset(X_test, vocab)
        return train_dataset, test_dataset, vocab


    def create_dataloaders(self, train_dataset, test_dataset):
        train_dataloader = DataLoader(train_dataset, batch_size=32,
                            shuffle=True, collate_fn=train_dataset.collate_fn)
        val_dataloader = DataLoader(test_dataset, batch_size=32,
                               shuffle=False, collate_fn=test_dataset.collate_fn)
        return train_dataloader, val_dataloader

# np save ()
    def create_pretrained_embeddings(self, vocab):
        matrix_len = len(vocab._token2idx)
        weights_matrix = np.zeros((matrix_len, 100))
        words_found = 0
        for i, word in enumerate(vocab._token2idx):
            try: 
                weights_matrix[i] = self.d[word]
                words_found += 1
            except KeyError:
                weights_matrix[i] = np.random.normal(scale=0.6, size=(100, ))
        pretrained_embeddings = weights_matrix
        pretrained_embeddings = torch.FloatTensor(pretrained_embeddings)
        return pretrained_embeddings


    def binary_accuracy(self, preds, y):
        rounded_preds = torch.round(torch.sigmoid(preds))
        correct = (rounded_preds == y).float() 
        acc = correct.sum() / len(correct)
        return acc


    def train(self, model, iterator, optimizer, criterion, epoch, train_loss_list):
        #train_losses = []
        epoch_loss = 0
        epoch_acc = 0
        model.train()

        for batch_idx, (text, label, len_text) in enumerate(iterator): 
            text = text.to(device)
            label = label.to(device)
            predictions = model((text), len_text).squeeze(1)
            loss = criterion(predictions, label.float())
            acc = self.binary_accuracy(predictions, label.float())   
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            train_loss_list.append(loss.item())

            if batch_idx % 50 == 0:
                self.plot(epoch, batch_idx, train_loss_list)
        return epoch_loss/len(iterator), epoch_acc / len(iterator), train_loss_list


    def evaluate(self, model, iterator, criterion):
        eval_losses = []
        epoch_loss = 0
        epoch_acc = 0
        model.eval()

        with torch.no_grad():
            for text, label, len_text in iterator:
                text = text.to(device)
                label = label.to(device)
                predictions = model((text), len_text).squeeze(1)
                loss = criterion(predictions, label.float())
                eval_losses.append(loss.item())
                acc = self.binary_accuracy(predictions, label.float())
                epoch_loss += loss.item()
                epoch_acc += acc.item()
        return epoch_loss/len(iterator), epoch_acc / len(iterator)


    def plot(self, epoch, step, train_losses):
        clear_output()
        plt.title(f'Epochs {epoch}, step {step}')
        plt.plot(train_losses)
        plt.show()


    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs


    def make_all(self): 
        train_dataset, test_dataset, vocab = self.create_train_dataset()
        train_dataloader, test_dataloader = self.create_dataloaders(train_dataset, test_dataset)
        pretrained_embeddings = self.create_pretrained_embeddings(vocab)
        
    
        INPUT_DIM = len(vocab)
        EMBEDDING_DIM = 100
        HIDDEN_DIM = 256
        OUTPUT_DIM = 1
        PAD_IDX = 0

        model = RNN(INPUT_DIM,
                    EMBEDDING_DIM,
                    HIDDEN_DIM, 
                    OUTPUT_DIM,
                    PAD_IDX).to(device)
        model.embedding.weight.data.copy_(pretrained_embeddings)
   
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()

        N_EPOCHS = 3
        train_loss_list = []
        for epoch in range(N_EPOCHS):
            start_time = time.time()

            train_loss, train_acc, train_loss_list = self.train(model, train_dataloader, optimizer, criterion, epoch, train_loss_list)
            valid_loss, valid_acc = self.evaluate(model, test_dataloader, criterion)
            
            end_time = time.time()

            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)
            torch.save(model.state_dict(), 'checkpoint.pth')

            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
        
        
    
class SentimentPredictor(object):
    def __init__(self):
        _, _, self.vocab = SentimentClassifier().create_train_dataset()

        INPUT_DIM = len(self.vocab)
        EMBEDDING_DIM = 100
        HIDDEN_DIM = 256
        OUTPUT_DIM = 1
        PAD_IDX = 0

        self.model = RNN(INPUT_DIM,
                    EMBEDDING_DIM,
                    HIDDEN_DIM, 
                    OUTPUT_DIM,
                    PAD_IDX).to(device)
        
    def predict_sentiment(self, sentence):
        self.model.load_state_dict(torch.load('checkpoint.pth'))
        self.model.eval()
        tokenized = [tokenizer(sentence) for sentence in splitter(sentence)]
        print('tokeinized', tokenized)
        indexed = [self.vocab.sent2idx(tokenized[0])]
        print('indexed', indexed)
        length = [len(indexed)]
        
        length_tensor = torch.LongTensor(length)
        
        tensor = torch.LongTensor(indexed).to(device)
        print('tensor size', tensor.size())
        print('len size', length_tensor)
        prediction = torch.sigmoid(self.model(tensor, length_tensor)).squeeze(1)
        print('prediction', prediction)
        print(prediction.shape)
        return prediction
