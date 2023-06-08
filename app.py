import streamlit as st
import pickle
import os

from streamlit_extras.add_vertical_space import add_vertical_space
from dotenv import load_dotenv

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

 
# Sidebar contents
with st.sidebar:
    st.title('pdfGPT')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')
    add_vertical_space(5)
    st.write('Copyright @ 2023 Tom Lam')


def main():
    st.header("Chat With PDF 💬")

    api_key = st.text_input("Enter your Open AI API Key:")

    pdf = st.file_uploader("upload your PDF", type='pdf')

    if pdf:
        # read pdf file
        pdf_reader = PdfReader(pdf)

        text = ""

        for page in pdf_reader.pages:
            # retrieve text from pdf file
            text += page.extract_text()
        
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        # split pdf text into smaller chunks
        chunks = text_splitter.split_text(text=text)


        ## Embeddlings

        store_name = pdf.name[:-4]

        # if the embeddling exists, load the embeddlings from the disk
        # nothing change
        if os.path.exists(f"{store_name}.pk1"):
            with open(f"{store_name}.pk1", "rb") as f:
                vectorStore = pickle.load(f)

        # if the embeddling doesn't exists, clear embeddling object, create a vectorstore and add it to the disk
        # new file uploaded
        else:
            embeddings = OpenAIEmbeddings()
            vectorStore = FAISS.from_texts(chunks, embedding=embeddings)

            with open(f"{store_name}.pk1","wb") as f:

                pickle.dump(vectorStore, f)

        # Accept User Question/Query
        query = st.text_input("Ask questions about your pdf:")

        if query:
            docs = vectorStore.similarity_search(query=query, k=3)

            os.environ["OPENAI_API_KEY"] = api_key
            llm = OpenAI(model_name='gpt-3.5-turbo', api_key=os.environ["OPENAI_API_KEY"])
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            st.write(response)


        



if __name__ == "__main__":
    main()
 