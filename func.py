from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnablePassthrough


# Create llm chain
def create_llm_chain(model, tokenizer):

    text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=True,
        max_new_tokens=300
        )
    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    prompt_template = """
    Инструкция: Ответьте на вопрос из следующего контекста и напишите ответ в поле «Ответ»:
    {context}

    Question:
    {question}

     """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )


    llm_chain = LLMChain(llm=llm, prompt=prompt)
    return llm_chain








def generate_answer(question, retriever, llm_chain):
    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
    | llm_chain
)
    response = rag_chain.invoke(question)
    return response