# Pytorch Negative Sampling Loss

[Negative Sampling Loss](https://arxiv.org/abs/1310.4546) implemented in [PyTorch](http://www.pytorch.org).

![NEG Loss Equation](images/neg.png)

## Usage

```python
    neg_loss = NEG_loss(num_classes, embedding_size)
    
    optimizer = SGD(neg_loss.parameters(), 0.1)
    
    for i in range(num_iterations):
        # input and target are [batch_size] shaped tensors of Long type
        input, target = next_batch(batch_size)
        
        loss = neg_loss(input, target, num_sample).mean()
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    word_embeddings = neg_loss.input_embeddings()


        
        
    


```