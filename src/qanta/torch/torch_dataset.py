from torch.utils.data import Dataset
from qanta.dataset import QuizBowlDataset
from nltk.tokenize import word_tokenize
import torch
import random


class TorchQBData(Dataset):
    def __init__(self, dataset: QuizBowlDataset, stoi, pad_index, n_samples=None,
                split_sentences=False):

        self.split_sentences = split_sentences
        self.qs, self.pages, _ = dataset.training_data()
        if n_samples is not None:
            self.qs = self.qs[:n_samples]
            self.pages = self.pages[:n_samples]
        self.answers = list(set(self.pages))
        self.n_answers = len(self.answers)
        self.i_to_ans = dict(enumerate(self.answers))
        self.ans_to_i = dict((n, i) for i, n in  enumerate(self.answers))
        assert(len(self.qs) == len(self.pages))
        self.stoi = stoi
        assert(pad_index == 0)

        if split_sentences:
            q_by_sent = []
            p_by_sent = []
            for i in range(len(self.qs)):
                sentences = [s for s in self.qs[i]]
                pages = [self.pages[i]] * len(sentences)
                q_by_sent.extend(sentences)
                p_by_sent.extend(pages)
            self.qs = q_by_sent
            self.pages = p_by_sent

    
    def __len__(self):
        return(len(self.qs))
    
    def __getitem__(self, idx):
        if self.split_sentences:
            tokens = word_tokenize(self.qs[idx])
        else:
            sentences_concatenated = ' '.join(self.qs[idx])
            tokens = word_tokenize(sentences_concatenated)
        ex = (tokens, self.pages[idx])
        return self.vectorize(ex)

    def vectorize(self, ex):
        """
        vectorize a single example based on the word2ind dict. 

        Keyword arguments:
        exs: list of input questions-type pairs
        ex: tokenized question sentence (list)
        label: type of question sentence

        Output:  vectorized sentence(python list) and label(int)
        e.g. ['text', 'test', 'is', 'fun'] -> [0, 2, 3, 4]
        """

        question_text, question_label = ex
        vec_text = [self.stoi(w) for w in question_text]
        #### modify the code to vectorize the question text
        #### You should consider the out of vocab(OOV) cases
        #### question_text is already tokenized
        return vec_text, self.ans_to_i[question_label]

    @staticmethod
    def collate(batch):
        question_len = list()
        label_list = list()
        for ex in batch:
            question_len.append(len(ex[0]))
            label_list.append(ex[1])
        target_labels = torch.LongTensor(label_list)
        # NOTE: pad index is asserted to 0
        x1 = torch.LongTensor(len(question_len), max(question_len)).zero_()
        for i in range(len(question_len)):
            question_text = batch[i][0]
            vec = torch.LongTensor(question_text)
            x1[i, :len(question_text)].copy_(vec)
        q_batch = {'text': x1, 'len': torch.FloatTensor(question_len), 'labels': target_labels}
        return q_batch
        

class OverfitDataset(Dataset):
    def __init__(self, dataset: QuizBowlDataset, stoi, pad_index, n_samples=None):

        self.qs, self.pages, _ = dataset.training_data()
        self.qs = self.qs[:n_samples]
        self.pages = self.pages[:n_samples]
        self.answers = list(set(self.pages))
        self.n_answers = len(self.answers)
        self.i_to_ans = dict(enumerate(self.answers))
        self.ans_to_i = dict((n, i) for i, n in  enumerate(self.answers))
        assert(len(self.qs) == len(self.pages))
        self.stoi = stoi
        assert(pad_index == 0)
    
    def __len__(self):
        return(len(self.qs))
    
    def __getitem__(self, idx):
        sentences_concatenated = ' '.join(self.qs[idx])
        tokens = word_tokenize(sentences_concatenated)
        ex = (tokens, self.pages[idx])
        return self.vectorize(ex, idx)

    def vectorize(self, ex, idx):
        """
        vectorize a single example based on the word2ind dict. 

        Keyword arguments:
        exs: list of input questions-type pairs
        ex: tokenized question sentence (list)
        label: type of question sentence

        Output:  vectorized sentence(python list) and label(int)
        e.g. ['text', 'test', 'is', 'fun'] -> [0, 2, 3, 4]
        """

        question_text, question_label = ex
        vec_text = [idx for w in question_text]
        #### modify the code to vectorize the question text
        #### You should consider the out of vocab(OOV) cases
        #### question_text is already tokenized
        return vec_text, self.ans_to_i[question_label]

    @staticmethod
    def collate(batch):
        question_len = list()
        label_list = list()
        for ex in batch:
            question_len.append(len(ex[0]))
            label_list.append(ex[1])
        target_labels = torch.LongTensor(label_list)
        # NOTE: pad index is asserted to 0
        x1 = torch.LongTensor(len(question_len), max(question_len)).zero_()
        for i in range(len(question_len)):
            question_text = batch[i][0]
            vec = torch.LongTensor(question_text)
            x1[i, :len(question_text)].copy_(vec)
        q_batch = {'text': x1, 'len': torch.FloatTensor(question_len), 'labels': target_labels}
        return q_batch
        