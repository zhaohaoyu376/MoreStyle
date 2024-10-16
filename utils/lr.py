

def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs)**exponent


def adjust_learning_rate(optimizer, epoch, initial_lr, max_epochs, exponent=0.9):
    lr = poly_lr(epoch, max_epochs, initial_lr, exponent)
    for i in range(len(optimizer.param_groups)):
        optimizer.param_groups[i]['lr'] = lr
    return lr

def adjust_weight(epoch,initial_lr=1e-2):
    initial_lr = initial_lr
    final_lr = initial_lr / 10
    if epoch <= 20:
        decay_rate = (final_lr / initial_lr) ** (1 / 20)
        current_lr = initial_lr * (decay_rate ** epoch)
    else:
        initial_lr = final_lr
        decay_rate = 0.95
        current_lr = initial_lr * (decay_rate ** (epoch - 10))
    return current_lr
    
def adjust_weight_new(epoch,initial_lr=1e-2):
    decay_rate = 0.99
    current_lr = initial_lr * (decay_rate ** epoch)
    # if epoch <= 20:
    #     decay_rate = (final_lr / initial_lr) ** (1 / 20)
    #     current_lr = initial_lr * (decay_rate ** epoch)
    # else:
    #     initial_lr = final_lr
    #     decay_rate = 0.95
    #     current_lr = initial_lr * (decay_rate ** (epoch - 10))
    return current_lr