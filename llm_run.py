import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Use Apple Metal Performance Shader (MPS) for better performance on Mac
device = torch.device("mps")

# Path to your downloaded model
model_path = "/Users/tanmay/Llama-3.2-1B"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # Use half-precision for speed
    device_map={"": device}  # Force model to MPS
).to(device)

# Define a prompt
prompt = "What is the meaning of life?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate response (optimized)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=200,
        do_sample=True,  # Enable sampling for variability
        temperature=0.7,  # Adjust creativity level
        top_p=0.9,  # Use nucleus sampling for better diversity
    )

# Decode output
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
