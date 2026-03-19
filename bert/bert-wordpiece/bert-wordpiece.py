from typing import List, Dict

class WordPieceTokenizer:
    """
    WordPiece tokenizer for BERT.
    """
    
    def __init__(self, vocab: Dict[str, int], unk_token: str = "[UNK]", max_word_len: int = 100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_word_len = max_word_len
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into WordPiece tokens.
        """
        tokens = []
        for word in text.lower().split():
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)
        return tokens
    
    def _tokenize_word(self, word: str) -> List[str]:
        """
        Tokenize a single word into subwords.
        """
        low = 0
        high = len(word)
        tokens = []
        cont = False

        def subword(word):
            prefix = "##" if cont else ""
            return prefix + word[low:high]
        
        while high > low:
            if subword(word) not in self.vocab:
                high -= 1
            else: # found
                tokens.append(subword(word))
                cont = len(tokens) > 0
                low = high
                high = len(word)

        if len(tokens) == 0:
            tokens.append(self.unk_token)

        return tokens
            