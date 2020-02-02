# Selective_Backpropagation 
from paper Accelerating Deep Learning by Focusing on the Biggest Losers  
https://arxiv.org/abs/1910.00762v1  

## Code example:  
### Without selective backpropagation:
```
...
criterion = nn.CrossEntropyLoss(reduction='none')
...
for x, y in data_loader:
    ...
    y_pred = model(x)
    loss = criterion(y_pred, y).mean()
    loss.backward()
    ...
```
### With selective backpropagation:
```
...
criterion = nn.CrossEntropyLoss(reduction='none')
selective_backprop = SelectiveBackPropagation(
                        criterion,
                        lambda loss : loss.mean().backward(),
                        optimizer,
                        model,
                        batch_size,
                        epoch_length=len(data_loader),
                        loss_selection_threshold=False)
...
for x, y in data_loader:
    ...
    with torch.no_grad():
        y_pred = model(x)
    not_reduced_loss = criterion(y_pred, y)
    selective_backprop.selective_back_propagation(not_reduced_loss, x, y)
    ...
```