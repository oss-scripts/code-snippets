from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.retrievers.base import BaseRetriever

# 1. your documents
docs = [
    Document(page_content="…", metadata={"source": "doc1.txt"}),
    Document(page_content="…", metadata={"source": "doc2.txt"}),
    Document(page_content="…", metadata={"source": "doc3.txt"}),
]

# 2. build FAISS
emb = OpenAIEmbeddings()
vect = FAISS.from_documents(docs, emb)

# 3. custom retriever that injects score into metadata
class ScoredRetriever(BaseRetriever):
    def __init__(self, vectorstore: FAISS, k: int = 5):
        self.vectorstore = vectorstore
        self.k = k
    def get_relevant_documents(self, query: str):
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=self.k)
        scored_docs = []
        for doc, score in docs_and_scores:
            doc.metadata["score"] = score
            scored_docs.append(doc)
        return scored_docs

retriever = ScoredRetriever(vect, k=5)

# 4. build chain
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(model_name="gpt-3.5‑turbo"),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
)

def run(query: str):
    out = qa({"query": query})
    answer = out["result"]
    sources = out["source_documents"]
    return answer, sources

if __name__ == "__main__":
    ans, srcs = run("Your question here")
    print("Answer:", ans)
    print("\nSources with scores:")
    for d in srcs:
        print(f"{d.metadata['source']} – score {d.metadata['score']:.4f}")
        print(d.page_content, "\n")
