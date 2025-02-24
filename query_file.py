import argparse
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
If the context doesn’t contain relevant information, say "The context doesn’t provide enough information to answer the question."
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    print(f"Number of documents in DB: {db._collection.count()}")

    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    print(f"Raw results count: {len(results)}")
    if results:
        for i, (doc, score) in enumerate(results, 1):
            print(f"Result {i}: Score={score}, Content={doc.page_content[:200]}..., Author={doc.metadata.get('author', 'Unknown')}")

    if len(results) == 0 or results[0][1] < 0.7:
        print("No highly relevant results found (score < 0.3). Using available context anyway.")

    context_text = "\n\n---\n\n".join([f"Content: {doc.page_content}\nAuthor: {doc.metadata.get('author', 'Unknown')}" for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print("\nGenerated Prompt:")
    print(prompt[:1000])

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.7)
    response = model.invoke(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response.content}\nSources: {sources}"
    print("\n" + formatted_response)

if __name__ == "__main__":
    main()
