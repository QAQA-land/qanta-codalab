# -*- coding: utf-8 -*-
"""
@author: Craig Thorburn
"""
from nltk.tokenize import word_tokenize
from nltk import FreqDist
import nltk
nltk.download('punkt')
import re

class ThisEntity:
    
    def __init__(self, threshold = 2, exceptions = ['is']):
        # Threshold: Number of mentions required to be entity
        # Exceptions: Words which should never be entities
        self._entities = FreqDist()
        self._exceptions = exceptions
        self._threshold = threshold
        self._finalized = False
        self._this_regex = re.compile('(^|\s)(t|T)his \w*')
        self.num_entities = None
        
    def get_entity(self, sentence):
        # Returns entity of sentence
        regex_return = self._this_regex.search(sentence)
        if regex_return == None:
            return 'UNK'
        tokens = word_tokenize(regex_return.group())
        if len(tokens) != 2:
            return 'UNK'
        entity = word_tokenize(regex_return.group())[1]
        return entity
        
    def add_sentence(self, sentence):
        # Adds entity of sentence to vocab
        assert not self._finalized, 'Vocab is already finalized'
        entity = self.get_entity(sentence)
        if not entity == 'UNK':
            self._entities[entity]+=1
        
    def finalize_vocab(self):
        # Finalizes vocab for testing
        # Returns number of entitites (including UNK)
        self._final_entities = [ee for ee in self._entities if self._entities[ee] >= self._threshold]
        for xx in self._exceptions:
            if xx in self._final_entities:
                self._final_entities.remove(xx)   
  
        self._final_entities = ['UNK'] + self._final_entities               
        self._finalized = True
        self.num_entities = len(self._final_entities)
        return self.num_entities
    
    def get_entity_index(self, sentence):
        # Returns index of entity (0 if UNK)
        assert self._finalized, 'Vocab not finalized'
        sentence_entity = self.get_entity(sentence)
        if sentence_entity not in self._final_entities:
            return 0
        else:
            return self._final_entities.index(sentence_entity)
        
    def get_all_entities(self):
        # Returns list of all entities
        assert self._finalized, 'Vocab not finalized'
        return self._final_entities
