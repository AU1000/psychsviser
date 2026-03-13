import pickle

def chunk_text(text, chunk_size=500):

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):

        chunk = " ".join(words[i:i+chunk_size])

        chunks.append(chunk)

    return chunks


if __name__ == "__main__":

    with open("processed/book_text.txt","r",encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)

    with open("processed/book_chunks.pkl","wb") as f:
        pickle.dump(chunks,f)

    print("Total chunks:",len(chunks))