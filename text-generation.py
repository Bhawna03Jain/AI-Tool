pip install transormers
********************************

  Models
  ======================
openai-community/gpt2
Qwen/Qwen2.5-1.5B-Instruct

  ******************************

from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2')
set_seed(42)
generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)
