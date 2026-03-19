import numpy as np

class BertEmbeddings:
    """
    BERT Embeddings = Token + Position + Segment
    """
    
    def __init__(self, vocab_size: int, max_position: int, hidden_size: int):
        self.hidden_size = hidden_size
        
        # Token embeddings
        self.token_embeddings = np.random.randn(vocab_size, hidden_size) * 0.02
        
        # Position embeddings (learned, not sinusoidal)
        self.position_embeddings = np.random.randn(max_position, hidden_size) * 0.02
        
        # Segment embeddings (just 2 segments: A and B)
        self.segment_embeddings = np.random.randn(2, hidden_size) * 0.02
    
    def forward(self, token_ids: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
        """
        Compute BERT embeddings.
        """
        token_ids = np.atleast_2d(token_ids)
        segment_ids = np.atleast_2d(segment_ids)

        B, N = token_ids.shape
        positions = np.arange(N)
        
        tok_emb = self.token_embeddings[token_ids]
        pos_emb = self.position_embeddings[positions]
        seg_emb = self.segment_embeddings[segment_ids]

        return tok_emb + pos_emb + seg_emb