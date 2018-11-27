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
from qanta.torch.torch_dataset import TorchQBData
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

wiki2vec = gensim.models.KeyedVectors.load_word2vec_format('embeddings/enwiki_20180420_100d.txt')
wiki2vec_weights = torch.FloatTensor(model.vectors)

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


''' DAN '''

class DanModel(nn.Module):
    def __init__(self, n_classes, vocab_size, emb_dim=50,
                 n_hidden_units=50, nn_dropout=.5, pretrained=True):
        super(DanModel, self).__init__()
        self.vocab_size = vocab_size  # do we put this in...?
        self.emb_dim = emb_dim
        self.n_classes = n_classes
        self.n_hidden_units = n_hidden_units
        self.nn_dropout = nn_dropout

        self.clf = nn.Linear(10, 1)
        if pretrained:
            self.embeddings = nn.Embedding.from_pretrained(wiki2vec_weights)
        else:
            self.embeddings = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=0)
        pass

    def forward(self, questions: List[str]):
        log.info('Calling DanModel.forward with: %s ...' % questions[0])
        N_answers = 2
        batchsize = len(questions)

        a = torch.zeros((batchsize, N_answers))
        a[:,0] = 1.0
        return a

class DanGuesser(object):
    def __init__(self):
        # TODO: initialize with DAN model 
        # (Remove this once saving and dummy model is no longer needed)
        self.model = DanModel()
        # TODO:

        self.i_to_ans = {0: 'Egypt', 1: 'London'}
        self.batch_size = 5
        self.max_epochs = 1
    
    def train(self, torch_qb_data: TorchQBData) -> None:
        log.info('Loading Quiz Bowl dataset')
        train_dataloader = DataLoader(torch_qb_data, batch_size = self.batch_size, shuffle=True, num_workers=1)
        log.info(f'N Train={len(train_dataloader)}')
        #log.info(f'N Test={len(val_iter.dataset.examples)}')

        # TODO: SET THIS 
        self.n_classes = 2#len(self.ans_to_i)
        self.emb_dim = 50 # TODO: set this later
        log.info('Initializing Model')
        self.model = DanModel()
        if CUDA:
            self.model = self.model.cuda()

        # TODO log hyperparameters
        log.info(f'Model:\n{self.model}')

        self.optimizer = Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, verbose=True, mode='max')
        
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
            train_acc, train_loss, train_time = self.run_epoch(train_dataloader)

            # TODO: actually get validation set
            self.model.eval()
            test_acc, test_loss, test_time = self.run_epoch(train_dataloader)

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
    
    def run_epoch(self, dataloader) -> None:
        epoch_start = time.time()
        for i_batch, sample_batched in enumerate(dataloader):
            if i_batch % 10000 == 0:
                log.info('Example of answers: %s', sample_batched['page'])
        acc, loss = 0, 0
        epoch_end = time.time()
        return acc, loss, epoch_end - epoch_start
            
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
