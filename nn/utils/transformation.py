from typing import List, Callable, Optional
import torch
from functools import identity
from torch.utils.data import DataLoader


def mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor, p: float = 1.0) -> torch.Tensor:
    """
    Perform generalized mean pooling on model output with attention mask.
    In NLP, it can be used to get a single fixed-size vector representing the entire sentence or document.

    Parameters:
    - model_output (torch.Tensor): Output tensor from the model with shape (batch_size, seq_len, hidden_size).
    - attention_mask (torch.Tensor): Attention mask with shape (batch_size, seq_len) indicating padding tokens.
    - p (float): Power parameter for generalized mean. Default is 1.0 (mean pooling).

    Returns:
    - torch.Tensor: Pooled output tensor of shape (batch_size, hidden_size).
    """
    if len(model_output.shape) != 3:
        raise ValueError(f"Expected model_output to be 3D tensor, got shape {model_output.shape}")

    if len(attention_mask.shape) != 2:
        raise ValueError(f"Expected attention_mask to be 2D tensor, got shape {attention_mask.shape}")

    if model_output.shape[:2] != attention_mask.shape:
        raise ValueError("model_output and attention_mask dimensions don't match")

    token_embeddings = model_output.float()
    attention_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * attention_expanded, 1)
    mean_embeddings = sum_embeddings / (attention_expanded.sum(1) + 1e-9)  # Adding epsilon for stability
    pooled = mean_embeddings**p

    return pooled


def get_embeddings(model, tokenizer, texts: List[str], batch_size: int, device: torch.device, 
                   pooling_function: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                   p=1) -> torch.Tensor:
    """Generate embeddings for a list of texts using the model and tokenizer"""
    if not isinstance(texts, list):
        raise TypeError("Texts must be a list")
        
    if not isinstance(batch_size, int):
        raise TypeError("Batch size must be an integer")
    
    if pooling_function is None:
        pooling_function = default_pooling_function
        
    # Use dataloader for batching
    text_dataloader = DataLoader(texts, batch_size=batch_size) 
    
    # Switch model to evaluation mode
    model.eval()
    with torch.no_grad():
        embeddings = []
        for batch in text_dataloader:
            encoded = tokenizer(batch, padding=True, truncation=True, return_tensors="pt") 
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            pooled = pooling_function(outputs, encoded['attention_mask'], p=p)
            pooled = F.normalize(pooled, p=2, dim=-1)
            embeddings.append(pooled)
        embeddings = torch.cat(embeddings)
        
    return embeddings


@staticmethod
def default_pooling_function(model_output: torch.Tensor, attention_mask: torch.Tensor, p: int) -> torch.Tensor:
    """
    Default function that returns the model output without any pooling.

    Parameters:
    - model_output (torch.Tensor): Output tensor from the model with shape (batch_size, seq_len, hidden_size).
    - attention_mask (torch.Tensor): Attention mask with shape (batch_size, seq_len) indicating padding tokens.

    Returns:
    - torch.Tensor: The unchanged model output.
    """
    return identity(model_output)