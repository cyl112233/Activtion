class psilu(nn.Module):
    def __init__(self):
        super(psilu, self).__init__()
    def forward(self,x):
        return torch.sin(x) * sigmoid(torch.sin(x))+1e-9
