
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.output import LLMResult
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.callbacks.base import BaseCallbackHandler
from flask import Flask, Response, render_template, request, redirect
import torch
from typing import Any, Dict, List
from queue import Queue
import time
import threading
import shutil
import os
import logging
import json
from test import uniqueCourse, CanvasAPI
import unicodedata

# email imports


CLIENT_SECRETS_FILE = '/Users/wafichoudhury/Desktop/inboxZeroApp/ouathcred.json'
API_NAME = 'gmail'
API_VERSION = 'v1'
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
texts = []
emails = []
# Match this with the redirect URI in your credentials.json
REDIRECT_URI = 'https://localhost:8080/oauth2callback'
# Gmail API service
service = None


MY_TEMPLATE = """
Your job is to help a user understand more about thier lives through their canvas information which is a class management system. You should also answer all general questions, regardless of if they relate to canvas or not. 
You will analyze metadata about canvas courses and assigments and give thoughtful responses, for example if someone asks "how should I plan my day based on my assignments" you will look at the information
and plan a day centered around finishing those assigments. You should also be able to summarize performance in a course based on grades, assignments, and the syllabus. You should have context on thier canvas information already. You can also answer all general information.
If formatting, such as bullet points, numbered lists, tables, or code blocks, is necessary for a comprehensive response, please apply the appropriate formatting.

<ctx>
CONTEXT:
{context}
</ctx>

QUESTION:
{question}

ANSWER
"""


class Email:
    def __init__(self, subject, sender, date, body):
        self.subject = subject
        self.sender = sender
        self.date = date
        self.body = self.clean_text(body)

    def clean_text(self, text):
        # Function to clean non-printable characters from text
        return ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')

    def concatenate_fields(self):
        # Concatenate all fields of the email
        return f"{self.clean_text(self.subject)} {self.clean_text(self.sender)} {self.clean_text(self.date)} {self.body}"


app = Flask(__name__)
app.secret_key = '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'

qa_chain = None
llm = None
prompt_queue = Queue()
sse_event_queue = Queue()
response_thread = None

USE_OPENAI = True

logging.basicConfig(filename="FlaskGPT.log", level=logging.INFO, filemode="w")


@app.route('/homepage', methods=['GET', 'POST'])
def homepage():
    return render_template('homepage.html')


@app.route('/canvas', methods=['GET', 'POST'])
def canvasPage():
    return render_template('canvas.html')


@app.route('/submit_token', methods=['POST'])
def submit_token():
    if request.method == 'POST':
        # Get the token from the form data
        token = request.form['token']
        downloadCanvasData(token)
        create_vectordb()
        init_openai_llm()

        init_llm()
        # Redirect to the '/result' route with the token as a URL parameter
        return redirect("/")


@app.route('/', methods=['GET', 'POST'])
def index():

    global prompt_queue

    if request.method == 'POST':
        try:
            prompt = request.form.get('prompt')
            print(prompt)
            prompt_queue.put(prompt)

            return {'status': 'success'}, 200
        except Exception as e:
            logging.error(f"Error processing prompt: {e}")
            return {'status': 'error', 'message': 'Failed to process prompt'}, 500

    return render_template('index.html')


def downloadCanvasData(api_token):
    canvas_api = CanvasAPI(api_token)

    # Retrieve all courses
    courses = canvas_api.get_courses()
    # Array of course IDs
    # Replace these with your course IDs
    course_ids = ['1368991', '1370796',
                  '1368993', '1370878', '1366220']

    global texts
    for i in range(len(course_ids)):
        course = course_ids[i]
        assignment = canvas_api.get_assignments(course)
        module = canvas_api.get_modules(course)
        course_detail = canvas_api.get_course_details(course)
        philClass = uniqueCourse(
            assignment, courses,  module, course_detail, course)
        textPart = ""
        textPart += ("Class for " + str(course_ids[i]) + " :\n")
        textPart += (philClass.stringRep())
        textPart += ("\n")
        texts.append(textPart)
    print("finished", texts)
    return texts


class StreamHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        global sse_event_queue
        sse_event_queue.put(
            {'type': 'token', 'content': token.replace('\n', '<br>')})

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        global sse_event_queue
        sse_event_queue.put({'type': 'start'})

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        global sse_event_queue
        sse_event_queue.put({'type': 'end'})

    def on_llm_error(self, error: BaseException, **kwargs) -> None:
        global sse_event_queue
        sse_event_queue.put({'type': 'error', 'content': str(error)})


def send_sse_data():
    global qa_chain, prompt_queue, sse_event_queue, response_thread
    while True:
        if not prompt_queue.empty():
            if response_thread and response_thread.is_alive():
                continue

            prompt = prompt_queue.get()

            response_thread = threading.Thread(target=qa_chain.run, args=(
                prompt,), kwargs={'callbacks': [StreamHandler()]})
            response_thread.start()

        while not sse_event_queue.empty():
            sse_event = sse_event_queue.get()
            yield f"data: {json.dumps(sse_event)}\n\n"

        time.sleep(1)


@app.route('/stream', methods=['GET'])
def stream():
    def event_stream():
        return send_sse_data()

    return Response(event_stream(), content_type='text/event-stream')


def create_vectordb():
    global texts

    try:
        db_path = os.path.join(os.getcwd(), 'db')

        if os.path.exists(db_path):
            shutil.rmtree(db_path)
        loader_dir = DirectoryLoader('data', glob='*.txt', loader_cls=TextLoader,

                                     loader_kwargs=None)

        documents = loader_dir.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=200)
        new_texts = text_splitter.split_documents(documents)
        print("after split")
        print(new_texts)
        model_kwargs = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        embedding = HuggingFaceBgeEmbeddings(
            model_name='thenlper/gte-base', model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
        vectordb = Chroma.from_documents(documents=new_texts,
                                         embedding=embedding,
                                         persist_directory=db_path)
        logging.info(f"The '{db_path}' folder has been created.")

    except Exception as e:
        logging.error(f"An error occurred while processing file: {e}")


def init_openai_llm():
    global llm
    try:
        llm = ChatOpenAI(streaming=True, temperature=0.0, callbacks=[])
    except Exception as e:
        logging.error("OpenAI failed to initialize: {e}.")


def init_llm():
    global llm, qa_chain

    try:
        db_path = os.path.join(os.getcwd(), 'db')

        model_kwargs = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        embedding = HuggingFaceBgeEmbeddings(
            model_name='thenlper/gte-base', model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
        vectordb = Chroma(persist_directory=db_path,
                          embedding_function=embedding)
        retriever = vectordb.as_retriever(search_kwargs={'k': 10})
        prompt_template = PromptTemplate.from_template(MY_TEMPLATE)
        chain_type_kwargs = {'verbose': True, 'prompt': prompt_template}
        qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                               chain_type='stuff',
                                               retriever=retriever,
                                               return_source_documents=False, verbose=True,
                                               chain_type_kwargs=chain_type_kwargs)
        logging.info(f"LLM initialized")
    except Exception as e:
        logging.error(f"LLM failed to initialize : {e}")


if __name__ == '__main__':
    app.debug = True
    app.run(port=8080, ssl_context='adhoc')
