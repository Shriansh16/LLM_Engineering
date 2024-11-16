import modal
from modal import App, Volume, Image

# Setup - define our infrastructure with code!
app = modal.App("ai-companionv4")
volume = Volume.from_name("model-cache", create_if_missing=True)  # Correctly define volume
image = Image.debian_slim().pip_install("huggingface", "torch", "transformers", "bitsandbytes", "accelerate", "peft", "langchain", "langchain_core","langchain_community")
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
    def query(self, text: str, system_template: str, session_id: str) -> str:
      from langchain.schema import SystemMessage, HumanMessage
      from langchain.memory import ChatMessageHistory
      from langchain.llms import HuggingFacePipeline
      from transformers import pipeline

      # Store for session-based history
      if not hasattr(self, "history_store"):
        self.history_store = {}

      # Fetch or create session-specific history
      if session_id not in self.history_store:
        self.history_store[session_id] = ChatMessageHistory()

      # Define the Hugging Face pipeline for text generation
      pipe = pipeline(
        "text-generation",
        model=self.fine_tuned_model,
        tokenizer=self.tokenizer,
        temperature=0.5,  # Adjust creativity level
        top_p=0.9,  # Top-p sampling
        repetition_penalty=1.2,
        max_new_tokens=400  # Ensure the response is within the token limit
      )

      # Wrap the pipeline for LangChain
      llm = HuggingFacePipeline(pipeline=pipe)

      # Define the system message with SystemMessage
      system_message = SystemMessage(content=system_template)

      # Construct the full prompt
      prompt = f"{system_template}\n\n###Human: {text}\n###Response:"

      # Generate response using the LLM
      response = llm(prompt)

      # Ensure response is a string and clean up the result
      response_str = str(response).strip()

      # Update the session history
      self.history_store[session_id].add_message(HumanMessage(content=text))  # Add human message
      self.history_store[session_id].add_message(SystemMessage(content=response_str))  # Add AI response

      # Return only the AI's response without system message
      return response_str







    @modal.method()
    def wake_up(self) -> str:
        return "ok"
