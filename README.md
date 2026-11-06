# Week 13 - PDF Question Answering

## Task Title
Building a PDF Question Answering System using
LangChain, ChromaDB, and Ollama.

## Domain
Education

## Objective
- Understand Retrieval-Augmented Generation.
- Extract text from PDF documents.
- Split text into smaller chunks.
- Generate vector embeddings.
- Store embeddings in ChromaDB.
- Retrieve relevant information.
- Generate answers using Ollama.

## Technologies Used
- Python
- LangChain
- ChromaDB
- Sentence Transformers
- PyPDF
- Ollama
- Pandas
- Jupyter Notebook

## Models
Embedding Model: all-MiniLM-L6-v2
LLM Model: llama3.2:3b

## Dataset
Artificial Intelligence in Education PDF.

## Implementation
1. Load PDF.
2. Extract text.
3. Split text into chunks.
4. Generate embeddings.
5. Store chunks in ChromaDB.
6. Retrieve relevant chunks.
7. Generate answers using Ollama.
8. Save answers in CSV.

## Output Files
- extracted_text.txt
- chunks.csv
- answers.csv

## Conclusion
The PDF Question Answering system was implemented
using LangChain, ChromaDB, and Ollama.

The system retrieves relevant PDF content and
generates context-aware answers using a local LLM.