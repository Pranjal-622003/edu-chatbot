import streamlit as st
from langchain_helper import get_qa_chain

st.title("Prepsat Q&A 🌱")


question = st.text_input("Question: ")

if st.button("Get Answer"):
    if question:
        chain = get_qa_chain()
        response = chain(question)

        st.header("Answer")
        st.write(response["result"])
