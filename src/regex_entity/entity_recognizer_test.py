# -*- coding: utf-8 -*-
"""
@author: Craig Thorburn
"""

import unittest

from entity_recognizer import ThisEntity


class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        self.base_test = ThisEntity(threshold = 0, exceptions = [])
        self.exception_test = ThisEntity(threshold = 0, exceptions = ['is'])
        self.threshold_test = ThisEntity(threshold = 2, exceptions = [])
        self.multple_test = ThisEntity(threshold = 0, exceptions = [])
        
        self.base_test.add_sentence('This man did something')
        self.base_test.add_sentence('this boy did something')
        self.base_test.add_sentence('I will tell you that this woman did something')
        self.base_test.add_sentence('I will tell you that This girl did something')
        self.base_test.add_sentence('alskdfjahsldkfj')
        
        self.multple_test.add_sentence('This man this man')
        self.multple_test.add_sentence('this boy did something')
        self.multple_test.add_sentence('this poet this girl')

        
        self.threshold_test.add_sentence('This man did something')
        self.threshold_test.add_sentence('this man did something')
        self.threshold_test.add_sentence('I will tell you that this woman did something')
        self.threshold_test.add_sentence('I will tell you that This girl did something')
        
        self.exception_test.add_sentence('This man did something')
        self.exception_test.add_sentence('this is doing something')
        self.exception_test.add_sentence('I will tell you that this woman did something')
        self.exception_test.add_sentence('I will tell you that This girl is doing something')


    def test_base(self):
        self.assertEqual(self.base_test.finalize_vocab(),4)
        self.assertEqual(self.base_test.get_entity_index('this man'),[0])
        self.assertEqual(self.base_test.get_entity_index('this boy'),[1])
        self.assertEqual(self.base_test.get_entity_index('this woman'),[2])
        self.assertEqual(self.base_test.get_entity_index('this girl'),[3])
        self.assertEqual(self.base_test.get_entity_index('this person'),[])
    
    def test_multiple(self):
        self.assertEqual(self.multple_test.finalize_vocab(),4)
        self.assertEqual(self.multple_test.get_entity_index('this man said something to this poet'),[0, 2])
        self.assertEqual(self.multple_test.get_entity_index('this boy said something to this poet'),[1, 2])
        self.assertEqual(self.multple_test.get_entity_index('this woman said something to this man'),[0])
        self.assertEqual(self.multple_test.get_entity_index('this girl'),[3])
        
    def test_threshold(self):
        self.assertEqual(self.threshold_test.finalize_vocab(),1)
        self.assertEqual(self.threshold_test.get_entity_index('this man'),[0])
        self.assertEqual(self.threshold_test.get_entity_index('this woman'),[])
        self.assertEqual(self.threshold_test.get_entity_index('this girl'),[])
        self.assertEqual(self.threshold_test.get_entity_index('this person'),[])
        
    def test_exceptions(self):
        self.assertEqual(self.exception_test.finalize_vocab(),3)
        self.assertEqual(self.exception_test.get_entity_index('this man'),[0])
        self.assertEqual(self.exception_test.get_entity_index('this is'),[])
        self.assertEqual(self.exception_test.get_entity_index('this woman'),[1])
        self.assertEqual(self.exception_test.get_entity_index('this girl'),[2])
        
if __name__ == '__main__':
    unittest.main()