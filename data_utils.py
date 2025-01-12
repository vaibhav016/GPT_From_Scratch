import os 
import torch
from datasets import load_dataset

from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer


class WikiTextDataset(Dataset):
    def __init__(self, texts, tokenizer, sequence_len):
        super(WikiTextDataset, self).__init__()

        self.tokenizer = tokenizer
        self.sequence_len = sequence_len

        tokenized_data = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
        self.input_ids = tokenized_data["input_ids"].view(-1)

    def __len__(self,):
        return len(self.input_ids)//self.sequence_len
    
    def __getitem__(self, idx):
        start_indx = idx*self.sequence_len
        end_indx = start_indx + self.sequence_len + 1

        inputs = self.input_ids[start_indx:end_indx]
        x = inputs[:-1]
        y = inputs[1:]

        return x ,y 



def create_dataloader(batch_size, sequence_length, tokenizer_name="gpt2"):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    texts = dataset["train"]["text"]  # Use the 'train' split
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure pad_token is set for GPT models
    texts = [text for text in texts if len(text.strip()) > 0]
    wikitext_dataset = WikiTextDataset(texts, tokenizer, sequence_length)

    dataloader = DataLoader(wikitext_dataset, batch_size=batch_size, shuffle=True, )
    
    return dataloader, tokenizer

def save_checkpoint(model, optimizer, epoch, save_path="checkpoint.pth"):
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), 
                "optimizer_state_dict": optimizer.state_dict(), }, save_path)
    print(f"Checkpoint saved at epoch {epoch}.")


def load_checkpoint(model, optimizer, save_path="checkpoint.pth"):
    if os.path.exists(save_path):
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        print(f"Checkpoint loaded. Resuming from epoch {epoch + 1}.")
        return epoch + 1
    else:
        print("No checkpoint found. Starting from scratch.")
        return 0


def count_parameters_in_millions(model):
    """
    Counts the number of trainable parameters in a PyTorch model in millions.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        float: The number of trainable parameters in millions.
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params / 1e6


# Generate Text Function
def generate_text(model, tokenizer, prompt, max_steps=3):
    """
    Generates text by predicting the next token repeatedly.

    Args:
        model (GPT): Trained GPT model.
        tokenizer: Hugging Face tokenizer.
        prompt (str): Initial text to generate from.
        max_steps (int): Number of tokens to generate.

    Returns:
        str: Generated text.
    """
    model.eval()
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    for _ in range(max_steps):
        with torch.no_grad():
            outputs = model(input_ids)  # Get log_probs
            # B X S X V 
            # 16 X 10 X 26
            next_token = torch.argmax(outputs[:, -1, :], dim=-1)  # Predict next token
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)  # Append next token

    # Decode the generated sequence
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print(f"Testing Generation for the Input String -> {prompt}::  {generated_text}")
     



        
        