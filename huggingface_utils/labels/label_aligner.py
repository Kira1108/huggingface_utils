"""
Use LabelAligner doesn't change `Tokenizer` behavior.
You can safely pass the toeknizer into huggingface Trainer.
Align labels for token classification task
Simple Case
```
aligner = LabelAligner()
aligner(labels, word_ids)
```
IOB case
```
label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
aligner = LabelAligner(label_names = label_names, use_iob = True)
aligner(labels, word_ids)
```
"""


from __future__ import annotations
from dataclasses import dataclass
from typing import List
from huggingface_utils.labels.iob import IOB as iob

@dataclass
class LabelAligner:
    label_names:List[str] = None
    use_iob:bool = False

    def __post_init__(self) -> None:
        if self.use_iob:
            if self.label_names is None:
                raise ValueError("label_names must be provided when use_iob is set to True")
            self.label_mapper = iob.create_iob_label_mapper(self.label_names)
        else:
            self.label_mapper = None

    def __call__(self, labels, word_ids):
        if self.label_mapper is None:
            return align_labels(labels, word_ids)
        else:
            return align_labels_with_iob(labels, word_ids, self.label_mapper)


def align_labels(labels, word_ids):
    """Simple aligner(ignore IOB format)"""
    return [
        -100 if word_id is None
        else labels[word_id] 
        for word_id in word_ids
    ]

def align_labels_with_iob(labels, word_ids, label_mapper):
    """Aligner consider IOB format
    Parameters:
        `labels` is an array of interger (word-tokenized-labels, not subword)
        `word_ids` is a list of integers
        represents the original position of a subword in the word-tokenized-array
        `label_mapper`: a dictionary specify label replace rule.
    Returns:
        aligned_labels: consider special tokens, subwords and IOB format
        convert the labels array into subword tokenized array
    """

    # stores aligned labels
    aligned_labels = []

    # keep there last_word_id to compare with
    last_word_id = None
    for word_id in word_ids:
        # incase [CLS] [SEP] like tokens
        if word_id is None:
            label = -100

        # incase new word, find the label of the word
        # through word_id
        elif word_id != last_word_id:
            label = labels[word_id]

        # incase of an old word, find the label
        # and alter the label if necessary
        else:
            label = labels[word_id]
            if label in label_mapper.keys():
                label = label_mapper[label]

        aligned_labels.append(label)
        last_word_id = word_id
    return aligned_labels

if __name__ == "__main__":

    # Test Simple Case
    labels =     [       1,  0,      1,    1]
    word_ids =   [None,  0,  1,  1,  2,    3,  None]
    want_label = [-100,  1,  0,  0,  1,    1,  -100]

    aligner = LabelAligner()
    print("Simple Case Test:")
    print("Expected: ", want_label)
    print("Got     : ", aligner(labels, word_ids))


    # Test IOB case
    label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
    labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0]
    word_ids = [None,0,1,1,1,2,3,4,5,5,5,5,5,5,6,7,7,7,8,9,9,10,10,11,12,13,14,15,16,17,None]
    want_label = [-100,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7,8,0,0,0,0,0,0,0,-100]

    aligner = LabelAligner(label_names = label_names, use_iob = True)
    print("\nIOB Case Test:")
    print("Expected: ", want_label)
    print("Got     : ", aligner(labels, word_ids))