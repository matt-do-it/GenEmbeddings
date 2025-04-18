import os

import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.tensorboard import SummaryWriter

sentences = [
	"How did the stock market develop?",
	"What is the development of the stock market?",
	"How can I have great organic food?",
	"The stock did drop.",
	"The stock price did fall.",
	"Price of the stock did reduce.", 
	"The stock did go up."
]

def gen_embeddings(model_name, sentences, out_name):
    # --- Setup the model ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # --- Initialize tensorboard ---
    log_dir = "runs/embeddings"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # 2. Initialize containers
    all_embeddings = []
    all_labels = []
    
    # --- Generate embeddings ---
    for sentence in sentences:
        encoded = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
    
        with torch.no_grad():
            output = model(**encoded)
    
        # Mean pooling
        token_embeddings = output.last_hidden_state
        attention_mask = encoded['attention_mask']
    
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        sentence_embedding = sum_embeddings / sum_mask
        print(sentence_embedding)
    
    #    sentence_embedding = output.last_hidden_state[:, 0, :]
    
        all_embeddings.append(sentence_embedding.squeeze(0))  # remove batch dim
        all_labels.append(sentence)  # use original sentence as label
    
    # --- Write embeddings to tensorboard ---
    embedding_tensor = torch.stack(all_embeddings)
    
    # 5. Write to TensorBoard
    writer = SummaryWriter(log_dir="runs/" + out_name)
    writer.add_embedding(embedding_tensor, metadata=all_labels)
    writer.close()

gen_embeddings("sentence-transformers/all-MiniLM-L6-v2", sentences, "all-MiniLM-L6-v2")
gen_embeddings("distilbert/distilbert-base-uncased", sentences, "distilbert-base-uncased")
