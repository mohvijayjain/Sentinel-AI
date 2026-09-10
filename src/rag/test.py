from src.rag.embeddings import embed_text


text = "The challenger model was rejected because it failed the RMSE evaluation gate."

embedding = embed_text(text)

print("Embedding generated successfully")
print("Dimensions:", len(embedding))
print("First 5 values:", embedding[:5])