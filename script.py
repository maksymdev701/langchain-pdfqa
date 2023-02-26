# import the modules
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import pickle

reader = PdfReader('ipc.pdf')

raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text

text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

embeddings = OpenAIEmbeddings()

with open("foo.pkl", 'wb') as f:
    pickle.dump(embeddings, f)

with open("foo.pkl", 'rb') as f:
    new_docsearch = pickle.load(f)

docsearch = FAISS.from_texts(texts, new_docsearch)

query = "What is IPPAN?"
docs = docsearch.similarity_search(query)
print(docs[0].page_content)

chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
print(chain.run(input_documents=docs, question=query))