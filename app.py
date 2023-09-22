from flask import Flask, request
import os
from deep_translator import GoogleTranslator
from flask import Flask, request
from llama_index import LangchainEmbedding, ServiceContext,StorageContext,load_index_from_storage
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index import  SimpleDirectoryReader
from llama_index import GPTVectorStoreIndex
import os
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import langid
import math

BASE_DIR = os.getcwd()


#loding llm
llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    #model_url="https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/resolve/main/llama-2-13b-chat.ggmlv3.q4_0.bin",
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path="D:\mp_gpt_L2\mp_gpt_L2\models\llama-2-7b-chat.ggmlv3.q4_0.bin",
    temperature=0.1,
    max_new_tokens=1000,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=4000,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": 1},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

#loding embed model
embed_model = LangchainEmbedding(
  HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
)

#create embedings
folder_path = BASE_DIR+"/data1"
documents = SimpleDirectoryReader(folder_path).load_data()
service_context = ServiceContext.from_defaults(llm=llm,embed_model=embed_model)
vector_index=GPTVectorStoreIndex.from_documents(documents=documents,service_context=service_context,)
folder_path = BASE_DIR+"/embeding"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
print('creating embedings')
vector_index.storage_context.persist(folder_path)


#load embedings
service_context = ServiceContext.from_defaults(llm=llm,embed_model=embed_model)
storage_context=StorageContext.from_defaults(persist_dir=folder_path)
new_index=load_index_from_storage(storage_context,service_context=service_context)               
query_engine = new_index.as_query_engine(service_context=service_context)
print('generation data')

app = Flask(__name__)


@app.route("/")
def deafult():
   return "Welcome"

@app.route("/recommendation", methods=["GET"])
def recommendation():
  input=request.args.get('input')
  template="""
  you are having data of procedures. based on data you have.Recommend {1} best procedure suit for user query

  user query={0}

  output in this format

  Procedure id : id of procedure

  Procedure title : procedure title

  Procedure revision : procedure revision

  short explanation why this procedure is best suitable for user query. except this informaton dont give any other information like based on provided query best procedures these kind explantion dont give directly fallow the format what i suggested output should contains only procedure id, procedure title, procedure revision and short explantion of model. if user asked multiple recommendation give output one by one


  """
  lang=langid.classify(input)
  if lang[0]=='es':
     print('spanish')
     input=GoogleTranslator(source='es', target='en').translate(input)
     print(input)
  print('generating output')
  response = query_engine.query(template.format(input,3))
  if lang[0]=='es':
      no_of_repetation=math.ceil(len(str(response))/500)
      text=''
      start,end=0,500
      reduced_text=''
      response=str(response)
      for i in range(no_of_repetation):
          reduced_text=response[start:end]
          
          reduced_text = GoogleTranslator(source='en',target='es').translate(reduced_text)

          start+=500
          end+=500
          text+=reduced_text+' '
      response=text
  print(response)
  return str(response)

if __name__ == "__main__":
    app.run()
