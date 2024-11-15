import modal
from modal import App, Volume, Image

# Setup - define our infrastructure with code!
app = modal.App("ai-companionv4")
volume = Volume.from_name("model-cache", create_if_missing=True)  # Correctly define volume
image = Image.debian_slim().pip_install("huggingface", "torch", "transformers", "bitsandbytes", "accelerate", "peft", "langchain", "langchain_community")
secrets = [modal.Secret.from_name("huggingface-secret")]

# Constants
GPU = "T4"
BASE_MODEL = "meta-llama/Meta-Llama-3-8B"
PROJECT_NAME = "AI COMPANION"
FINETUNED_MODEL = "MLsheenu/AI_COMPANION_finetuned_llama3"

@app.cls(image=image, secrets=secrets, gpu=GPU, timeout=1800, volumes={"/cache": volume})  # Attach volume here
class Companion:
    @modal.build()
    def download_model_and_tokenizer(self):
        from huggingface_hub import snapshot_download
        import os

        MODEL_DIR = "/cache/models"  # Use the shared volume path
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Download base and fine-tuned models and tokenizer
        snapshot_download(BASE_MODEL, local_dir=f"{MODEL_DIR}/{BASE_MODEL.split('/')[-1]}")
        snapshot_download(FINETUNED_MODEL, local_dir=f"{MODEL_DIR}/{FINETUNED_MODEL.split('/')[-1]}")
        
        # Download and store the tokenizer
        snapshot_download(BASE_MODEL, local_dir=f"{MODEL_DIR}/{BASE_MODEL.split('/')[-1]}/tokenizer")

    @modal.enter()
    def setup(self):
        import os
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        # Set up paths
        MODEL_DIR = "/cache/models"
        base_model_path = f"{MODEL_DIR}/{BASE_MODEL.split('/')[-1]}"
        fine_tuned_model_path = f"{MODEL_DIR}/{FINETUNED_MODEL.split('/')[-1]}"
        tokenizer_path = f"{base_model_path}/tokenizer"  # Tokenizer directory path
        
        # Quant Config
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )
    
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)  # Load tokenizer from shared volume
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path, 
            quantization_config=quant_config,
            device_map="auto"
        )
    
        self.fine_tuned_model = PeftModel.from_pretrained(self.base_model, fine_tuned_model_path)

    @modal.method()  # No volumes needed here as it uses preloaded models
    def query(self, text: str, system_template: str) -> str:
      from langchain.chains import ConversationChain
      from langchain.chains.conversation.memory import ConversationBufferWindowMemory
      from langchain.prompts import (
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
        ChatPromptTemplate,
        MessagesPlaceholder
      )
      from langchain.llms import HuggingFacePipeline
      from transformers import pipeline

      # Set up memory with a fixed window size (e.g., 3 most recent messages)
      if not hasattr(self, "buffer_memory"):
          self.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

      # Define prompt templates using the user-provided system message template
      system_msg_template = SystemMessagePromptTemplate.from_template(template=system_template)
      human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

      # Create the chat prompt template with history (only the last 3 messages)
      prompt_template = ChatPromptTemplate.from_messages([
        system_msg_template,
        MessagesPlaceholder(variable_name="history"),  # Placeholder for context history
        human_msg_template
      ])

      # Define the Hugging Face pipeline for text generation
      pipe = pipeline(
        "text-generation",
        model=self.fine_tuned_model,
        tokenizer=self.tokenizer,
        temperature=0.5,  # Adjust creativity level
        top_p=0.9,  # Top-p sampling
        max_new_tokens=1024  # Ensure the response is within the token limit
      )

      # Wrap the pipeline for LangChain
      llm = HuggingFacePipeline(pipeline=pipe)

      # Create or reuse the conversation chain
      if not hasattr(self, "conversation"):
        self.conversation = ConversationChain(
            memory=self.buffer_memory,
            prompt=prompt_template,
            llm=llm,
            verbose=False
          )

      # Append the new input to memory and generate the response
      response = self.conversation.predict(input=text)

      # Return the response without additional conversation history
      return response.strip()

    @modal.method()
    def wake_up(self) -> str:
        return "ok"
