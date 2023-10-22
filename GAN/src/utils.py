import torch

def save_ckpt(model, optimizer, epoch, filename):
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(state, filename)

def load_ckpt(filename):
    ckpt = torch.load(filename)
    return ckpt