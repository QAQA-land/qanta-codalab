# -*- coding: utf-8 -*-
"""
@author: Craig Thorburn
"""
from nltk.tokenize import word_tokenize
from nltk import FreqDist
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
        
    def get_entity(self, sentence):
        # Returns entity of sentence
        regex_return = self._this_regex.finditer(sentence)
        entities = []
        for ii in regex_return:
            entities.append(word_tokenize(ii.group())[1])
        if entities == []:
            return 'UNK'
        else:
            return entities
        
    def add_sentence(self, sentence):
        # Adds entity of sentence to vocab
        assert not self._finalized, 'Vocab is already finalized'
        entities = self.get_entity(sentence)
        if not entities == 'UNK':
            for ee in entities:
                self._entities[ee]+=1
        
    def finalize_vocab(self):
        # Finalizes vocab for testing
        # Returns number of entitites (including UNK)
        self._final_entities = [ee for ee in self._entities if self._entities[ee] >= self._threshold]
        for xx in self._exceptions:
            if xx in self._final_entities:
                self._final_entities.remove(xx)   
  
        self._final_entities = ['UNK'] + self._final_entities               
        self._finalized = True
        return len(self._final_entities)
        
    
    def get_entity_index(self, sentence):
        # Returns index of entity (0 if UNK)
        assert self._finalized, 'Vocab not finalized'
        sentence_entities = self.get_entity(sentence)
        index_list=[]
        if sentence_entities == []:
            return index_list
        else:
            for ee in sentence_entities:
                if ee in self._final_entities:
                   index_list.append(self._final_entities.index(ee))
            return index_list
        
    def get_all_entities(self):
        # Returns list of all entities
        assert self._finalized, 'Vocab not finalized'
        return self._final_entities
