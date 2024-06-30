
from torch.autograd import Variable

def makeVariable(tensor, use_gpu=True, type='long', requires_grad=True):
    # conver type
    tensor = tensor.data
    if type == 'long':
        tensor = tensor.long()
    elif type == 'float':
        tensor = tensor.float()
    else:
        raise NotImplementedError

    # make is as Variable
    if use_gpu:
        variable = Variable(tensor.cuda(), requires_grad=requires_grad)
    else:
        variable = Variable(tensor, requires_grad=requires_grad)
    return variable