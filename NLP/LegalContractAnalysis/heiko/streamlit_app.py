from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
import streamlit as st
from transformers.pipelines import pipeline
import json
from predict import run_prediction

st.set_page_config(layout="wide")

st.cache(show_spinner=False, persist=True)
def load_questions():
	with open('/home/ec2-user/SageMaker/LegalContractAnalysis/cuad/data/test.json') as json_file:
		data = json.load(json_file)

	questions = []
	for i, q in enumerate(data['data'][0]['paragraphs'][0]['qas']):
		question = data['data'][0]['paragraphs'][0]['qas'][i]['question']
		questions.append(question)
	return questions

st.cache(show_spinner=False, persist=True)
def load_contracts():
	with open('/home/ec2-user/SageMaker/LegalContractAnalysis/cuad/data/test.json') as json_file:
		data = json.load(json_file)

	contracts = []
	for i, q in enumerate(data['data']):
		contract = ' '.join(data['data'][i]['paragraphs'][0]['context'].split())
		contracts.append(contract)
	return contracts

questions = load_questions()
contracts = load_contracts()

st.header("Legal Contract Review Demo")
st.write("This demo uses a the CUAD machine learning model for Contract Understanding.")

add_text_sidebar = st.sidebar.title("Menu")
add_text_sidebar = st.sidebar.text("Hello, world!")

question = st.selectbox('Choose one of the 41 queries from the CUAD dataset:', questions)
paragraph = st.text_area(label="Contract")


if (not len(paragraph)==0) and not (len(question)==0):
	predictions = run_prediction([question], paragraph, '/home/ec2-user/SageMaker/LegalContractAnalysis/cuad/trained-models/roberta-base/')
	st.write(f"Answer: {predictions['0']}")
	
my_expander = st.beta_expander("Sample Contract", expanded=False)
my_expander.write(contracts[1])