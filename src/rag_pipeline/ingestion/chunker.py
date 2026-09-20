
#fixed chunking

def fixed_chunking(text: str, chunk_size: int = 500, chunk_overlap: int = 50): 
    chunks = []
    start = 0 
    text_length= len(text)

    while start < text_length: 
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - chunk_overlap
    return chunks

def recursive_chunk(text: str, chunk_size: int = 500, separators: tuple[str] = ("\n\n", ". ", " ")) -> list[str]: 

    if len(text) <= chunk_size: 
        return [text] 

    if not separators: 
        return fixed_chunking(text)

    current_separator = separators[0]
    remaining_separators = separators[1:]

    chunks = text.split(current_separator)
    result = []

    for chunk in chunks: 
        if not chunk: 
            continue
        if len(chunk) <= chunk_size: 
            result.append(chunk)
        else: 
            #too big, recurse with next seperator
            smaller_chunk = recursive_chunk(chunk, chunk_size, tuple(remaining_separators))
            result.extend(smaller_chunk)

    return result
