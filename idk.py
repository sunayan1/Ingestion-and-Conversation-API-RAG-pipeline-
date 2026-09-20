from rag_pipeline.conversation.retrieval import retrieve_relevant_chunks
chunks = retrieve_relevant_chunks("What is the purpose of pumped-storage hydropower?")
for c in chunks:
    print(repr(c[:200]))
    print("---")