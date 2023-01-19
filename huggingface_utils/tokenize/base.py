from typing import Protocol
from transformers.tokenization_utils_base import BatchEncoding

class BatchTokenizeFn(Protocol):

    def tokenize(self, texts:list) -> BatchEncoding:
        """tokenize list of lists of strings"""

    def align_labels(self, labels:list, tokenized_inputs:BatchEncoding) -> BatchEncoding:
        """align labels with tokenized_inputs"""

    def __call__(self, batch:dict) -> BatchEncoding:
        """tokenize batch data, align labels...."""