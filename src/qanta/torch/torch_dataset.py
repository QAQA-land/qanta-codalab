from torch.utils.data import Dataset
from qanta.dataset import QuizBowlDataset
from nltk.tokenize import word_tokenize
import torch


class TorchQBData(Dataset):
    def __init__(self, 
                questions,
                pages,
                stoi_dict,
                pad_index,
                unk_index,
                n_samples=None):

        self.qs, self.pages = questions, pages
        assert(len(self.qs) == len(self.pages))
        if n_samples != None:
            self.qs = self.qs[:n_samples]
        self.answers = list(set(self.pages))
        self.n_answers = len(self.answers)
        self.i_to_ans = dict(enumerate(self.answers))
        self.ans_to_i = dict((n, i) for i, n in  enumerate(self.answers))
        self.stoi_ = stoi_dict
        assert(pad_index == 0)
        assert(stoi_dict['<PAD>'] == pad_index)
        assert(stoi_dict['<UNK>'] == unk_index)
    
    def __len__(self):
        return(len(self.qs))

    def stoi(self, s):
        if s in self.stoi_:
            return self.stoi_[s]
        else:
            return self.stoi_['<UNK>']
    

    def __getitem__(self, idx):
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
        