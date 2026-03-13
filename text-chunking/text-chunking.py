def text_chunking(tokens, chunk_size, overlap):
    """
    Split tokens into fixed-size chunks with optional overlap.
    """
    chunks = []
    if (len(tokens) == 0):
        return []

    # Handle edge case
    if (len(tokens) < chunk_size):
        chunks.append(tokens)
        return chunks
        
    step = chunk_size - overlap
    
    for i in range(0, len(tokens), step):
        if (i + chunk_size > len(tokens)):
            break
        chunks.append(tokens[i:i+chunk_size])

    return chunks