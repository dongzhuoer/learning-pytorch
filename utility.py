import torch

def run(model, data_loader, prefix, train = False, optimizer = None):
    loss, correct, device = 0, 0, next(model.parameters()).device
    for batch_i, (inputs, targets) in enumerate(data_loader):
        inputs, targets = [x.to(device) for x in inputs] if type(inputs) is list else inputs.to(device), targets.to(device)
        outputs = model(inputs)
        batch_loss = torch.nn.CrossEntropyLoss()(outputs, targets)
        if train:
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        loss += batch_loss.item()
        correct += torch.sum(torch.argmax(outputs, dim = 1) == targets).item()
    loss, accuracy = loss/len(data_loader), correct/len(data_loader.dataset)
    print(f"{prefix}, loss: {loss/len(data_loader):.3f}, correct: {correct:5d}/{len(data_loader.dataset):5d} ({accuracy*100:.2f}%)")
    return [loss, accuracy]

def train(model, train_loader, prefix, optimizer):
    model.train()
    return run(model, train_loader, prefix, True, optimizer)

def valid(model, valid_loader, prefix):
    model.eval()
    with torch.no_grad():
        return run(model, valid_loader, prefix)