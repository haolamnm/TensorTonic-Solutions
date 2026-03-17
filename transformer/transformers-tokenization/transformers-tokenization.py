import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        self.word_to_id[self.pad_token] = 0
        self.word_to_id[self.unk_token] = 1
        self.word_to_id[self.bos_token] = 2
        self.word_to_id[self.eos_token] = 3
        corpus = " ".join(texts)
        uniq_words = sorted(set(corpus.split(" ")))
        
        for index, word in enumerate(uniq_words):
            word = word.lower()
            if word not in self.word_to_id:
                self.word_to_id[word] = index
                self.id_to_word[index] = word
        
        self.vocab_size = len(self.word_to_id)

    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        tokens = []
        for word in text.split(" "):
            word = word.lower()
            token = self.word_to_id.get(word, 1)
            tokens.append(token)
        return tokens
            
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        text = []
        for index in ids:
            word = self.id_to_word.get(index, self.unk_token)
            text.append(word)
        return " ".join(text)
