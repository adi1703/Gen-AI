import argparse

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

CHROMA_PATH = "chroma"


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str)
    args = parser.parse_args()

    query_text = args.query_text

    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )

    results = db.similarity_search_with_relevance_scores(
        query_text,
        k=3
    )

    if len(results) == 0:
        print("No results found")
        return

    context_text = "\n\n".join(
        [doc.page_content for doc, score in results]
    )

    prompt = f"""
Context:
{context_text}

Question:
{query_text}

Answer:
"""
    generator = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )

    response = generator(
    prompt,
    max_new_tokens=100,
    do_sample=False
    )

    print(response[0]["generated_text"])

if __name__ == "__main__":
    main()