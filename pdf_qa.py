import os
import pandas as pd

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE_DIR, "dataset")
PDF_PATH = os.path.join(DATASET, "sample.pdf")
DB_PATH = os.path.join(BASE_DIR, "chroma_db")

# Model names
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:3b"

# Step 1: Check PDF
if not os.path.exists(PDF_PATH):
    raise FileNotFoundError("Please put sample.pdf inside dataset folder.")

print("Loading PDF...")

# Step 2: Extract PDF text
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print("PDF loaded successfully!")
print("Total pages:", len(documents))

full_text = "\n".join(
    [doc.page_content for doc in documents]
)

with open(
    os.path.join(DATASET, "extracted_text.txt"),
    "w",
    encoding="utf-8"
) as file:
    file.write(full_text)

print("Text extracted successfully!")

# Step 3: Split text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print("Total chunks:", len(chunks))

# Step 4: Save chunks into CSV
chunk_data = []

for i, chunk in enumerate(chunks):
    chunk_data.append({
        "chunk_id": i + 1,
        "page": chunk.metadata.get("page", 0) + 1,
        "content": chunk.page_content
    })

pd.DataFrame(chunk_data).to_csv(
    os.path.join(DATASET, "chunks.csv"),
    index=False
)

print("Chunks saved successfully!")

# Step 5: Load embedding model
print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

# Step 6: Create a fresh ChromaDB for this PDF

vector_store = Chroma(
    collection_name="education_pdf",
    embedding_function=embeddings,
    persist_directory=DB_PATH
)

# Step 7: Store chunks in ChromaDB
chunk_ids = [
    f"chunk_{i}" for i in range(len(chunks))
]

vector_store.add_documents(
    documents=chunks,
    ids=chunk_ids
)

print("Chunks stored in ChromaDB!")

# Step 8: Connect Ollama
llm = OllamaLLM(
    model=LLM_MODEL
)

print("Ollama connected!")

# Step 9: Create prompt template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an educational assistant.

Answer the question using only the given PDF context.

If the answer is not available in the context,
say: Answer not found in the PDF.

Context:
{context}

Question:
{question}

Answer:
"""
)

# Step 10: Load questions
queries = pd.read_csv(
    os.path.join(DATASET, "queries.csv")
)

results = []

# Step 11: Answer all 15 questions
for _, row in queries.iterrows():

    question = row["question"]

    print("\nQuestion:", question)

    # Answer without retrieval
    answer_without_retrieval = llm.invoke(question)

    # Retrieve top 5 relevant chunks
    retrieved_docs = vector_store.similarity_search(
        question,
        k=5
    )

    context = "\n\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    # Generate answer using retrieved context
    final_prompt = prompt.format(
        context=context,
        question=question
    )

    answer_with_retrieval = llm.invoke(final_prompt)

    print("Answer:", answer_with_retrieval)

    results.append({
        "query_id": row["query_id"],
        "question": question,
        "answer_without_retrieval": answer_without_retrieval,
        "answer_with_retrieval": answer_with_retrieval,
        "retrieved_chunks": context
    })

# Step 12: Save answers
answers_df = pd.DataFrame(results)

answers_df.to_csv(
    os.path.join(DATASET, "answers.csv"),
    index=False
)

print("\nAll answers saved successfully!")

# Step 13: Ask a new question from keyboard
while True:

    user_question = input(
        "\nAsk a question about the PDF (type exit to stop): "
    )

    if user_question.lower() == "exit":
        break

    retrieved_docs = vector_store.similarity_search(
        user_question,
        k=5
    )

    context = "\n\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    final_prompt = prompt.format(
        context=context,
        question=user_question
    )

    answer = llm.invoke(final_prompt)

    print("\nAI Answer:")
    print(answer)

print("\nPDF Question Answering completed!")