import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter   
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator= "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function=len
    )
    
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstores = FAISS.from_texts(texts = text_chunks, embedding=embeddings)
    return vectorstores

def main():
    
    load_dotenv()
    
    st.set_page_config(page_title='Chat with multiple PDFs', page_icon=':books:')

    st.header("chat with multiple PDFs :books:")
    st.text_input("Ask any questions related to you documents :-)")

    with st.sidebar:
        st.header("Your documents")
        pdf_docs = st.file_uploader("Upload your pdfs here & click Process", accept_multiple_files=True)
        if st.button("Process"):
            st.spinner("Processing...")
            #get the pdf 
            raw_text = get_pdf_text(pdf_docs)

            #and then the text chunks from the data
            text_chunks = get_text_chunks(raw_text)
            #...st.write(text_chunks)...

            #create a vector storage using H.F.
            vectorstore = get_vectorstore(text_chunks)
            


if __name__ == '__main__':
    main()