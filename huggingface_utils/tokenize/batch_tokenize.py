from typing import Callable
from transformers.tokenization_utils_base import BatchEncoding

class TokenClassifyTokenizeFn:
    def __init__(self, 
                 tokenizer:Callable, 
                 label_aligner:Callable = None, 
                 input_column:str = "tokens",
                 label_column:str = 'labels'):
        
        self.tokenizer = tokenizer
        self.label_aligner = label_aligner
        self.label_column = label_column
        self.input_column = input_column

    def tokenize(self, texts:list) -> BatchEncoding:
        """tokenize batch
        
        You should refer to the tokenizer documentation on how to use it.
        Then call the function in right way.

        Args:
            texts (list): list of list of strings

        Returns:
            BatchEncoding: hugginface transformers BatchEncoding object
            
        """
        return self.tokenizer(
            texts, 
            truncation = True, 
            is_split_into_words = True)
        
    def align_labels(self, labels:list, tokenized_inputs:BatchEncoding) -> BatchEncoding:
        """use label aligner to deal with subword labels and iob format labels."""
        aligned_labels = [
            self.label_aligner(labels = l, word_ids = tokenized_inputs.word_ids(i)) 
            for i,l in enumerate(labels)
        ]
        return aligned_labels

    def __call__(self, batch:dict) -> BatchEncoding:
        texts = batch[self.input_column]
        tokenized_inputs = self.tokenize(texts)
        labels = batch[self.label_column]
        if self.label_aligner is None:
            aligned_labels = labels
        else:
            aligned_labels = self.align_labels(
                labels = labels, 
                tokenized_inputs = tokenized_inputs)
        tokenized_inputs['labels'] = aligned_labels
        return tokenized_inputs
    