import json
import pickle
import tempfile
import time
from collections import defaultdict
from os import path
from typing import List, Optional, Tuple

import numpy as np
import gensim
import click
import torch
import torch.nn as nn
from flask import Flask, jsonify, request
from qanta import qlogging, util
from qanta.dataset import QuizBowlDataset
from qanta.torch import (BaseLogger, EarlyStopping, MaxEpochStopping,
                         ModelCheckpoint, TerminateOnNaN, TrainingManager)
from qanta.torch.torch_dataset import TorchQBData, OverfitDataset
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

''' CONSTANTS '''
MODEL_PATH = 'dan.pickle'
BUZZ_NUM_GUESSES = 10
BUZZ_THRESHOLD = 0.3
log = qlogging.get(__name__)
CUDA = torch.cuda.is_available()



def create_save_model(model):
    def save_model(path):
        torch.save(model.state_dict(), path)
    return save_model

def get_tmp_filename(dir='/tmp'):
    with tempfile.NamedTemporaryFile('w', delete=True, dir=dir) as f:
        file_name = f.name

    return file_name


''' Quizbowl dataset iterator '''

def get_QuizbowlIter(QuizBowlDataset, batch_size=5):
    '''
    Returns iterator over batches of QuizBowl qs
    '''

class DanEmbedding(nn.Module):
    def __init__(self, pretrained_fp):
        super(DanEmbedding, self).__init__()
        self.vocab, self.stoi_, self.embedding_weights = \
            self.load_pretrained_weights(pretrained_fp)
        self.embed = nn.Embedding.from_pretrained(self.embedding_weights)
        self.embed_dim = self.embedding_weights.shape[1]
        self.pad_index = 0

    def load_pretrained_weights(self, pretrained_fp):
        wiki2vec = gensim.models.KeyedVectors.load_word2vec_format(pretrained_fp)
        vectors = wiki2vec.vectors
        vocab_len = len(vectors)
        vocab = [wiki2vec.index2word[i] for i in range(vocab_len)]

        vocab.insert(0, '<UNK>')
        vocab.insert(0, '<PAD>')

        unk_vec = np.mean(vectors, axis=0)
        pad_vec = np.zeros_like(vectors[0])
        vectors = np.vstack((pad_vec, unk_vec, vectors))
        vectors = torch.FloatTensor(vectors)
        stoi = dict((s, i) for i,s in enumerate(vocab))
        return vocab, stoi, vectors
    
    def stoi(self, s):
        if s in self.stoi_:
            return self.stoi_[s]
        else:
            return self.stoi_['<UNK>']
    
    def forward(self, questions: List[List[str]]):
        return self.embed(questions)

''' DAN '''
class DanModel(nn.Module):
    def __init__(self, embedding, embed_dim,  n_classes, n_hidden_units=50, nn_dropout=.5, pretrained=True):
        super(DanModel, self).__init__()
        log.info('loading embeddings')

        self.n_classes = n_classes
        self.n_hidden_units = n_hidden_units
        self.nn_dropout = nn_dropout
        self.embedding = embedding
        self.embed_dim = embed_dim
        self.linear1 = nn.Linear(self.embed_dim, n_hidden_units)
        self.linear2 = nn.Linear(n_hidden_units, n_classes)
        self.softmax = nn.Softmax(dim=1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=self.nn_dropout),
            self.linear1,
            nn.ReLU(),
            nn.Dropout(p=self.nn_dropout),
            self.linear2,
        )
        # if pretrained:
        #     
        # else:
        #     self.embeddings = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=0)
        # pass
    
    def forward(self, input_text, text_len, is_prob=False):
        """
        Model forward pass
        
        Keyword arguments:
        input_text : vectorized question text 
        text_len : batch * 1, text length for each question
        in_prob: if True, output the softmax of last layer

        """
        #### write the forward funtion, the output is logits 
        x = self.embedding(input_text)
        x = x.sum(1)
        x /= text_len.view(x.size(0), -1)
        x = self.classifier(x)

        if is_prob:
            return self.softmax(x)
        else:
            return x

class DanGuesser(object):
    def __init__(self, pretrained_fp, quizbowl_dataset,
                 batch_size=5,
                 max_epochs=1,
                 grad_clip=5,
                 n_training_samples=None):
        # TODO: initialize with DAN model 
        # (Remove this once saving and dummy model is no longer needed)
        self.dan_embed = DanEmbedding(pretrained_fp)
        self.torch_qb_data = TorchQBData(quizbowl_dataset, 
                                         stoi = self.dan_embed.stoi, 
                                         pad_index=self.dan_embed.pad_index,
                                         n_samples=n_training_samples)
        self.model = DanModel(embedding=self.dan_embed, 
                    embed_dim=self.dan_embed.embed_dim, 
                    n_classes=self.torch_qb_data.n_answers)        # TODO:

        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.is_train = True
        self.gradient_clip = grad_clip

        self.optimizer = Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, verbose=True, mode='max')

    def train(self) -> None:
        log.info('Loading Quiz Bowl dataset')
        train_dataloader = DataLoader(self.torch_qb_data, 
                                      batch_size = self.batch_size, 
                                      shuffle=True, num_workers=1,
                                      collate_fn=TorchQBData.collate)
        log.info(f'N Train={len(train_dataloader)}')

        #log.info(f'N Test={len(val_iter.dataset.examples)}')

        if CUDA:
            self.model = self.model.cuda()

        # TODO log hyperparameters
        log.info(f'Model:\n{self.model}')

        temp_prefix = get_tmp_filename()
        self.model_file = f'{temp_prefix}.pt'
        manager = TrainingManager([
            BaseLogger(log_func=log.info), TerminateOnNaN(), EarlyStopping(monitor='test_acc', patience=10, verbose=1),
            MaxEpochStopping(self.max_epochs), ModelCheckpoint(create_save_model(self.model), self.model_file, monitor='test_acc')
        ])

        log.info('Starting training')

        epoch = 0
        while True:
            self.model.train()
            train_acc, train_loss, train_time = self.run_epoch(train_dataloader, is_train=True)

            # TODO: actually get validation set
            self.model.eval()
            test_acc, test_loss, test_time = self.run_epoch(train_dataloader, is_train=False)

            stop_training, reasons = manager.instruct(
                train_time, train_loss, train_acc,
                test_time, test_loss, test_acc
            )

            if stop_training:
                log.info(' '.join(reasons))
                break
            else:
                self.scheduler.step(test_acc)
            epoch += 1
            log.info('Epoch complete: %d', epoch)
    
    def run_epoch(self, dataloader, is_train=True) -> None:
        epoch_start = time.time()
        batch_accuracies = []
        batch_losses = []
        for i_batch, batch in enumerate(dataloader):
            question_text = batch['text']
            question_len = batch['len']
            question_labels = batch['labels']

            if is_train:
                self.model.zero_grad()

            out = self.model(question_text, question_len)
            _, preds = torch.max(out, 1)
            accuracy = torch.mean(torch.eq(preds, question_labels).float()).item()
            batch_loss = self.criterion(out, question_labels)
            
            #print(i_batch, batch_loss.item())

            if is_train:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm(self.model.parameters(), 
                                              self.gradient_clip)
                self.optimizer.step()

            batch_accuracies.append(accuracy)
            batch_losses.append(batch_loss.data[0])

        epoch_end = time.time()

        return np.mean(batch_accuracies), np.mean(batch_losses), epoch_end - epoch_start
            
    def guess(self, questions: List[str], max_n_guesses: Optional[int]):
        if len(questions) == 0:
            return []
        batch_size = 500
        if len(questions) < batch_size:
            return self._guess_batch(questions, max_n_guesses)
        else:
            all_guesses = []
            for i in range(0, len(questions), batch_size):
                batch_questions = questions[i:i + batch_size]
                guesses = self._guess_batch(batch_questions, max_n_guesses)
                all_guesses.extend(guesses)
            return all_guesses

    def _guess_batch(self, questions: List[str], max_n_guesses: Optional[int]):
        if len(questions) == 0:
            return []
        guesses = []
        out = self.model(questions)
        probs = F.softmax(out).data.cpu().numpy()
        n_examples = probs.shape[0]
        preds = np.argsort(-probs, axis=1)
        for i in range(n_examples):
            guesses.append([])
            for p in preds[i][:max_n_guesses]:
                guesses[-1].append((self.i_to_ans[p], probs[i][p]))
        return guesses

    def save(self):
        log.info('This does not save')
        pass

    @classmethod
    def load(cls):
        guesser = DanGuesser()
        return guesser


''' CODE FOR WEB APP '''
def guess_and_buzz(model, question_text) -> Tuple[str, bool]:
    guesses = model.guess([question_text], BUZZ_NUM_GUESSES)[0]
    scores = [guess[1] for guess in guesses]
    buzz = scores[0] / sum(scores) >= BUZZ_THRESHOLD
    return guesses[0][0], buzz


def batch_guess_and_buzz(model, questions) -> List[Tuple[str, bool]]:
    question_guesses = model.guess(questions, BUZZ_NUM_GUESSES)
    outputs = []
    for guesses in question_guesses:
        scores = [guess[1] for guess in guesses]
        buzz = scores[0] / sum(scores) >= BUZZ_THRESHOLD
        outputs.append((guesses[0][0], buzz))
    return outputs

def create_app(enable_batch=True):
    dan_guesser = DanGuesser.load()
    app = Flask(__name__)

    @app.route('/api/1.0/quizbowl/act', methods=['POST'])
    def act():
        question = request.json['text']
        guess, buzz = guess_and_buzz(dan_guesser, question)
        return jsonify({'guess': guess, 'buzz': True if buzz else False})

    @app.route('/api/1.0/quizbowl/status', methods=['GET'])
    def status():
        return jsonify({
            'batch': enable_batch,
            'batch_size': 200,
            'ready': True
        })

    @app.route('/api/1.0/quizbowl/batch_act', methods=['POST'])
    def batch_act():
        questions = [q['text'] for q in request.json['questions']]
        return jsonify([
            {'guess': guess, 'buzz': True if buzz else False}
            for guess, buzz in batch_guess_and_buzz(dan_guesser, questions)
        ])


    return app


@click.group()
def cli():
    pass


@cli.command()
@click.option('--host', default='0.0.0.0')
@click.option('--port', default=4861)
@click.option('--disable-batch', default=False, is_flag=True)
def web(host, port, disable_batch):
    """
    Start web server wrapping DAN model
    """
    app = create_app(enable_batch=not disable_batch)
    app.run(host=host, port=port, debug=False)


@cli.command()
def train():
    """
    Train the DAN model, requires downloaded data and saves to models/
    """
    dataset = QuizBowlDataset(guesser_train=True)
    dan_guesser = DanGuesser()
    dan_guesser.train(TorchQBData(dataset))
    dan_guesser.save()


@cli.command()
@click.option('--local-qanta-prefix', default='data/')
def download(local_qanta_prefix):
    """
    Run once to download qanta data to data/. Runs inside the docker container, but results save to host machine
    """
    util.download(local_qanta_prefix)


if __name__ == '__main__':
    cli()
