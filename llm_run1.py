import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define paths
model_path = "/Users/tanmay/llama-3.2"  # Change if needed
pt_path = "llama3_1b.pt"  # Path to your .pt file

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

# Load model
model = AutoModelForCausalLM.from_pretrained(model_path)
model.load_state_dict(torch.load(pt_path, map_location="cpu"))

# Move model to available device
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# Test input
input_text = "Hello, how are you?"
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# Generate text
with torch.no_grad():
    output = model.generate(**inputs)

# Decode and print output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Text:", generated_text)
