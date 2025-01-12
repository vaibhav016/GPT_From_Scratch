import torch
from torch.optim import AdamW
from tqdm import tqdm

from data_utils import create_dataloader, generate_text, count_parameters_in_millions, load_checkpoint, save_checkpoint
from model import GPT

def train_gpt_model():
    batch_size = 128 
    sequence_length = 128
    num_epochs = 30
    learning_rate = 1e-3
    tokenizer_name = "gpt2"

    dataloader, tokenizer = create_dataloader(batch_size, sequence_length, tokenizer_name)
    model = GPT(vocabulary_size= len(tokenizer), 
                 embedding_size=768,  
                 sequence_len=sequence_length, 
                 num_layers=2, 
                 num_heads=4).to("cuda")
    
    num_params = count_parameters_in_millions(model)
    print(f"Trainable Parameters: {num_params:.2f}M")

    ############# TESTING 
    test_string = "A guitar is"
    print("*********** Before Training ***********")
    generate_text(model, tokenizer, test_string, max_steps=3)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    start_epoch = load_checkpoint(model, optimizer)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx , (inputs, targets) in progress_bar:
            optimizer.zero_grad()
            inputs, targets = inputs.to("cuda"), targets.to("cuda")
            log_probs = model(inputs)

            loss = model.loss(log_probs, targets)
            epoch_loss +=loss.item()

            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 200 == 0:
                progress_bar.set_postfix({"Loss": loss.item()})
                generate_text(model, tokenizer, test_string, max_steps=3)
                save_checkpoint(model, optimizer, epoch)
            
        print(f"Epoch {epoch + 1} completed. Average Loss: {epoch_loss / len(dataloader):.4f}")
        
        # Test the model after each epoch
        save_checkpoint(model, optimizer, epoch)


    print("Model training complete and saved.")
    

    



if __name__ == "__main__":
    train_gpt_model()
