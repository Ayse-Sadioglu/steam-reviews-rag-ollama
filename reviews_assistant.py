import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate


# Load the Steam reviews dataset
df = pd.read_csv("data/dataset.csv")

# Select review texts
texts = df["review_text"].dropna().head(1000).tolist()

# Convert texts into LangChain documents
docs = [Document(page_content=text) for text in texts]

# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(docs)

# Create embeddings and vector database
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
db = FAISS.from_documents(chunks, embeddings)

# Load local LLaMA model
llm = Ollama(model="llama3")

# Custom prompt to enforce RAG behavior
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an assistant that answers questions ONLY using the provided game reviews.

If the information is not mentioned in the reviews, say:
"The reviews do not mention this."

Game Reviews:
{context}

Question:
{question}

Answer:
"""
)

# Create Retrieval Augmented Generation pipeline
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

# Command line chat loop
print("Game Reviews Assistant")
print("Type 'exit' to quit")

while True:
    question = input("> ")
    if question.lower() == "exit":
        break
    answer = qa.run(question)
    print(answer)
