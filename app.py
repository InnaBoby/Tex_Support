import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.header('MVP системы технической поддержки пользователей', divider='rainbow')

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#download database
npa_db = FAISS.load_local("NPA_db", embedding_model, allow_dangerous_deserialization=True)
qa_db = FAISS.load_local("QA_db", embedding_model, allow_dangerous_deserialization=True)

question = st.text_area('Введите вопрос:', 'Текст вопроса...')

options = st.selectbox('Выберете вариант ответа', ('', 'Ответ из НПА', 'Похожие запросы'))
if options == '':
    st.write('Не выбран тип ответа')

    
if options == 'Ответ из НПА':
    result = npa_db.similarity_search(question)
    
    container = st.container(border=True)
    container.write(result[0].page_content)
    container.write('Источник:')
    container.write(result[0].metadata)

    container = st.container(border=True)
    container.write(result[1].page_content)
    container.write('Источник:')
    container.write(result[1].metadata)

    container = st.container(border=True)
    container.write(result[2].page_content)
    container.write('Источник:')
    container.write(result[2].metadata)

    container = st.container(border=True)
    container.write(result[3].page_content)
    container.write('Источник:')
    container.write(result[3].metadata)


if options == 'Похожие запросы':
    similar_question = qa_db.similarity_search(question)


    container = st.container(border=True)
    container.write(f':blue[{similar_question[0].page_content}]')
    if st.toggle('Показать ответ техподдержки 1'):
        st.write(f':green[{similar_question[0].metadata}]')


    container = st.container(border=True)
    container.write(f':blue[{similar_question[1].page_content}]')
    if st.toggle('Показать ответ техподдержки 2'):
        st.write(f':green[{similar_question[1].metadata}]')


    container = st.container(border=True)
    container.write(f':blue[{similar_question[2].page_content}]')
    if st.toggle('Показать ответ техподдержки 3'):
        st.write(f':green[{similar_question[2].metadata}]')


    container = st.container(border=True)
    container.write(f':blue[{similar_question[3].page_content}]')
    if st.toggle('Показать ответ техподдержки 4'):
        st.write(f':green[{similar_question[3].metadata}]')


