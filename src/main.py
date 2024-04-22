from auto import chat
from openai import OpenAI
from anthropic import Anthropic
from llama_cpp import Llama
import json
import argparse

class Interpreter:
  def __init__(self, model):
    self.openai_client = OpenAI()
    self.anthropic_client = Anthropic()
    self.model = model
    if self.model.startswith('NousResearch/'):
      self.llama_instance = Llama(model_path='/mnt/mac/Users/daniel/Library/Application Support/r2ai/models/Hermes-2-Pro-Mistral-7B.Q4_0.gguf', n_ctx=4096, verbose=False) 
    self.messages = []
    self.env = {}
    self.env["llm.maxtokens"] = "1750"
    self.env["llm.maxmsglen"] = "1750"
    self.env["llm.temperature"] = "0.002"



parser = argparse.ArgumentParser(description='OpenAI Chatbot')
parser.add_argument('--memory', type=bool, default=False, help='Use memory')
parser.add_argument('--message', type=str, default='', help='Message to send')
parser.add_argument('--model', type=str, default='NousResearch/', help='Model to use')

args = parser.parse_args()
interpreter = Interpreter(model=args.model)

if args.memory:
  try:
    with open('memory.json', 'r') as f:
      memory = json.load(f)
      interpreter.messages = memory["messages"]
  except:
    pass
if args.message:
  interpreter.messages.append({"role": "user", "content": args.message})
chat(interpreter)
with open('memory.json', 'w') as f:
  json.dump({"messages": interpreter.messages}, f)
