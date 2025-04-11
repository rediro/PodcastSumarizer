import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_path = "/Users/tanmay/llama-3.2"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

# Select device (Mac MPS, CUDA, or CPU)
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# Load model on selected device
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    torch_dtype=torch.float16
).to(device)

# Save as .pt file
pt_path = "llama3_1b.pt"
torch.save(model.state_dict(), pt_path)
print(f"✅ Model successfully saved to {pt_path}")
