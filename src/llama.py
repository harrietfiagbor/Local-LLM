import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, BitsandBytesConfig, pipeline
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain

model_id = "chavinlo/alpaca-native"

tokenizer = LlamaTokenizer.from_pretrained(model_id)
base_model = LlamaForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True,
    device_map="auto"
)

pipe = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    max_length=256,
    temperature=0.7,
    top_p=0.95,
    repetitive_penalty=1.95
)

local_llm = HuggingFacePipeline(pipeline=pipe)

with open('prompt_template.txt', 'r') as file:
    template = file.read()

prompt = PrompTemplate(template=template, input_variables=['instruction'])

llm-chain = LLMChain(
    prompt=prompt,
    llm=local_llm
)

mode=input('............................................\nDo you want to ask a one time question or a have a conversation? [Answer with "one-time" or "conversation"] : ')


if mode == "one-time":
    question = input('Human: ')
    print(llm_chain.run(question))
elif mode == "conversation":
    window_memory = ConversationBufferMemory(k=4)
    conversation = ConversationChain(
        llm=local_llm,
        verbose=True,
        memory=memory_window
    )
    conversation.predict(question)
else:
    print(f'This mode: {mode} is not supported. Supported modes are "One-time" and "Conversation"  ')
