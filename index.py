#1 and 2
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader

pdfreader  = PdfReader("MySchoolBucks - Chargebacks - Pre-Arbitration.pdf")
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

text_splitter = CharacterTextSplitter(separator="\n", chunk_size=800, chunk_overlap=200)
texts = text_splitter.split_text(raw_text)

#3
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

embeddings = OpenAIEmbeddings()
#db = FAISS.from_texts(texts, embeddings)

#save
#db.save_local("faiss_index")

#4
new_db = FAISS.load_local("faiss_index",embeddings)

#5
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k")

#llm = OpenAI(temperature=0)

qa_chain = RetrievalQA.from_chain_type(llm,retriever=new_db.as_retriever())
def ask(user_query):
    res = qa_chain({"query": user_query})
    return res["result"]

