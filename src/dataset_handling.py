import torch
from torch.utils.data import Dataset
from translate.storage.tmx import tmxfile

def textclassification_to_tsv(dataset, out_file):
    with open(out_file, 'w', encoding='utf8') as file:
        for sentence, label in dataset:
            file.write(f'{sentence}\t{label}\n')
                

class TextClassificationDataset(Dataset):

    def __init__(self, sentences, labels):
        super().__init__()

        assert len(sentences) == len(labels)
        
        self.sentences = sentences
        self.labels = labels


    def __getitem__(self, index) -> tuple:
        return (self.sentences[index], self.labels[index]) 
    

    def __len__(self) -> int:
        return len(self.sentences)
    
    @classmethod
    def from_tmx(cls, filename, lan1, lan2):
        
        with open(filename, 'rb') as file:
            tmx_file = tmxfile(
                inputfile=file, 
                sourcelanguage=lan1, 
                targetlanguage=lan2)

        sentences = []
        labels = []

        for node in tmx_file.unit_iter():

            if node.source == '' or node.target == '': #filter empty sentences
                continue

            # lan1
            sentences.append(node.source)
            labels.append(0)

            # lan2
            sentences.append(node.target)
            labels.append(1)

        new_obj = cls(
            sentences=sentences,
            labels=labels
        )


        return new_obj
    
    @classmethod
    def from_tsv(cls, filename):
        sentences = []
        labels = []
        with open(filename, 'r', encoding='utf8') as file:
            for line in file.readlines():
                sentence, label = line.strip('\n').split('\t')
                sentences.append(sentence)
                labels.append(int(label))
        
        new_obj = cls(
            sentences=sentences,
            labels=labels
        )
        return new_obj

    

class TextDataset(Dataset):

    def __init__(self, sentences):
        self.sentences = sentences

    def __getitem__(self, index):
        return self.sentences[index]

    def __len__(self):
        return len(self.sentences)

    @classmethod
    def from_tmx(cls, filename):

        with open(filename, 'rb') as file:
            tmx_file = tmxfile(
                inputfile=file)

        sentences = []

        for node in tmx_file.unit_iter():

            if node.source == '' or node.target == '': #filter empty sentences
                continue

            # lan1
            sentences.append(node.source)

            # lan2
            sentences.append(node.target)

        return cls(
            sentences=sentences
        )