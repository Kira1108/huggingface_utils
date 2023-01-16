"""Token Classification Labelling
1. if your classification model input(after tokenization) matches you labels, you don't need this.
2. if your label dont follow IOB format(something like begin-tag, inside-tag, out-tag), you don't need this.
When using subword tokenization, sentences are further tokenized into subwords,
which causes incompatibility between the labels and the tokenized inputs.
This module provides a solution to this problem.
Note: 
    1. gegin label is formatted as B-something
    2. inside labels is formatted as I-something
    3. label_names should be ordered
        e.g. labels_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
        corresponse to integer label labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    4. the maper is used in `align_labels_with_iob` function.
# use mapper creation functions -> str2str
ner_mapper = IOB.create_iob_str_mapper(label_names)
# use mapper creation functions -> int2int
ner_label_mapper = IOB.create_iob_label_mapper(label_names)
"""


import re
from typing import Dict

class IOB:
    @staticmethod
    def create_iob_str_mapper(label_names:list) -> Dict[str, str]:

        """
            Create a dict of {B-something: I-something}
            if both B-something and I-something inside label_names list
            Parameters:
                label_names: list of strings
        """
        mapper = {}
        # a begin label has format B-something"
        begin_pattern = re.compile("B-(\w+)")

        for label in label_names:
            begin_match = begin_pattern.match(label)

            if not begin_match:
                continue

            inside_label = "I-" + begin_match.groups()[0]

            if inside_label in label_names:
                mapper[label] = inside_label
        return mapper

    @staticmethod
    def create_iob_label_mapper(label_names:list) -> Dict[int, int]:
        """Create a dict of {Index(B-something): Index(I-something)}
            if both B-something and I-something inside label_names list
            Parameters:
                label_names: list of strings
        """

        labelname_to_id = {name:idx for idx, name in enumerate(label_names)}
        str_mapper = IOB.create_iob_str_mapper(label_names)
        label_mapper = {labelname_to_id[k]: labelname_to_id[v] 
                    for k, v in str_mapper.items()}
        return label_mapper


if __name__ == "__main__":
    ner_label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
    pos_label_names = ['"',"''",'#','$','(',')',',','.',':','``','CC','CD','DT','EX','FW','IN','JJ',
    'JJR','JJS','LS','MD','NN','NNP','NNPS','NNS','NN|SYM','PDT','POS','PRP','PRP$','RB','RBR','RBS',
    'RP','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB']

    chunk_label_names = ['O','B-ADJP','I-ADJP','B-ADVP','I-ADVP','B-CONJP','I-CONJP','B-INTJ','I-INTJ',
    'B-LST','I-LST','B-NP','I-NP','B-PP','I-PP','B-PRT','I-PRT','B-SBAR','I-SBAR','B-UCP','I-UCP','B-VP','I-VP']

    # use mapper creation functions -> str2str
    ner_mapper = IOB.create_iob_str_mapper(ner_label_names)
    chunk_mapper = IOB.create_iob_str_mapper(chunk_label_names)
    pos_mapper = IOB.create_iob_str_mapper(pos_label_names)

    # use mapper creation functions -> int2int
    ner_label_mapper = IOB.create_iob_label_mapper(ner_label_names)
    chunk_label_mapper = IOB.create_iob_label_mapper(chunk_label_names)
    pos_label_mapper = IOB.create_iob_label_mapper(pos_label_names)

    print("Ner Mapper")
    print(ner_mapper)
    print(ner_label_mapper)

    print("\nChunk Mapper")
    print(chunk_mapper)
    print(chunk_label_mapper)

    print("\nPos Mapper")
    print(pos_mapper)
    print(pos_label_mapper)