#Imports
import dotenv
from langchain.document_loaders import YoutubeLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import gradio as gr

docs = []
retriever = {}

def youtube_summary(my_url):

    loader = YoutubeLoader.from_youtube_url(
        my_url, add_video_info=False
    )

    global docs

    docs = loader.load()

    vec_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )

    vec_splits = vec_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(documents=vec_splits, embedding=OpenAIEmbeddings())

    global retriever

    retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={"k": 6})

    llm = ChatOpenAI(temperature=0)

    map_template = """The following is a transcript from a New York City Council Meeting.
    {docs}
    Based on this transcript, please identify angles that a journalist covering this city council meeting might want to write a story about. These do not have to be fully fleshed out stories. Rather, they should be leads that the journalist would follow up on with rigorous reporting. Please include the most relevant quote from the transcript for each angle.
    Helpful Answer:"""

    map_prompt = PromptTemplate.from_template(map_template)

    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    reduce_template = """The following is set of angles that a journalist might wish to pursue in their reporiting:
    {docs}
    Take these and distill it into a final, consolidated list of angles to follow up on. 
    Helpful Answer:"""
    reduce_prompt = PromptTemplate.from_template(reduce_template)

    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="docs"
    )

    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=4000,
    )

    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="docs",
        # Return the results of the map steps in the output
        return_intermediate_steps=False,
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=2000, add_start_index=True
    )

    split_docs = text_splitter.split_documents(docs)

    return map_reduce_chain.run(split_docs)

def rag_respond(message, history):

    llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    

    condense_q_system_prompt = """Given a chat history and the latest user question \
    which might reference the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    condense_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", condense_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    condense_q_chain = condense_q_prompt | llm | StrOutputParser()

    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Incorporate any important context but keep your answer concise.\

    {context}"""

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    def condense_question(input: dict):
        if input.get("chat_history"):
            return condense_q_chain
        else:
            return input["question"]
        
    rag_chain = (
        RunnablePassthrough.assign(context=condense_question | retriever | format_docs)
        | qa_prompt
        | llm
    )
    
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    gpt_response = rag_chain.invoke({"question":message,"chat_history":history_langchain_format})
    return gpt_response.content

with gr.Blocks() as demo:
    gr.Markdown("# New York City Council AI Reporter")
    gr.Markdown("Find [a video](https://www.youtube.com/@NYCCouncil/streams) of a past New York City Council meeting and paste the URL below, or for a quick demo, use the one provided. The AI reporter will analyze the video transcript and suggest story angles for you to pursue. This will take some time—roughly one minute for every 15 minutes of transcript (90 seconds for the demo). When it finishes, you may ask it any followup questions and it will incorporate any relevant information it gleaned in its answers.")
    with gr.Row():
        url_input = gr.Textbox(value="https://www.youtube.com/watch?v=FzCLB5ZFLdk",label="YouTube URL of Meeting")
        submit_url_button = gr.Button("Generate Story Angles")
    summary_output = gr.Textbox(label="Suggested Story Angles", interactive=False)

    gr.Markdown("### Ask Followup Questions")
    gr.ChatInterface(rag_respond)

    submit_url_button.click(youtube_summary, inputs=url_input, outputs=summary_output)
    gr.Markdown('_Known Issue:_ Sometimes, in follow-up questions, the chatbot will claim that a topic listed in the story angles did not come up in the meeting, when in fact it did. To get it chatbot to "remember" this part of the meeting, simply insist "yes, that was discussed".')
demo.launch()