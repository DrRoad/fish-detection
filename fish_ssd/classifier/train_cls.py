import time
import torch.backends.cudnn as cudnn
import torch
from torchvision import models
from torch.utils.data import DataLoader
from .datasets import *
from .model import *
from ..one_cycle_lr import *
from ..utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data parameters
data_path = './'  # path of data files

# Learning parameters
batch_size = 32  # batch size
epochs = 4  # number of epochs to run
lr = 1e-4  # learning rate

print_freq = 100  # print training or validation status every __ batches
workers = 4  # number of workers for loading data in the DataLoader
grad_clip = None  # Use a value of 0.5 if gradients are exploding, which may happen
                  # at larger batch sizes (sometimes at 32) - you will recognize it
                  # by a sorting error in the MuliBox loss calculation

cudnn.benchmark = True


def main():
    """
    Training and validation.
    """

    # load and set model/optimizer
    net = models.resnet50(pretrained=True)
    model = ResNetFish(net)

    params_1 = [p for p in model.group_1.parameters()]
    params_2 = [p for p in model.group_2.parameters()]
    params_3 = [p for p in model.group_3.parameters()]
    for p in params_1: p.requires_grad = True
    for p in params_2: p.requires_grad = True
    for p in params_3: p.requires_grad = True

    optimizer = torch.optim.Adam(params=[{'params': params_1}])
    optimizer.add_param_group({'params':params_2})
    optimizer.add_param_group({'params':params_3})

    # Move to default device
    model = model.to(device)

    # Custom dataloaders
    train_dataset = NCFMclass(data_path, split='train')
    val_dataset = NCFMclass(data_path, split='valid')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=workers, pin_memory=True)

    # Scheduler for one-cycle policy
    ft_lrs = [lr/100, lr/10, lr]
    scheduler = OneCycleScheduler(optimizer, epochs, train_loader, max_lr=ft_lrs)

    # Epochs
    for epoch in range(epochs):
        print(f'Epoch: {epoch+1}')

        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              optimizer=optimizer,
              scheduler=scheduler)

        # One epoch's validation
        val_loss = validate(val_loader=val_loader,
                            model=model)

        # Did validation loss improve?
        if epoch==0: best_loss = val_loss
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)

        # Save checkpoint
        if is_best:
            print(f'Save the better model at epoch {epoch+1}')
            torch.save(model.state_dict(), data_path+f'rn50_ft_epk_{epochs}.pth')

        print()


def train(train_loader, model, optimizer, scheduler):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param optimizer: optimizer
    :param scheduler: lr, mom scheduler
    """
    model.train()  # training mode enables dropout

    losses = AverageMeter()
    accs = AverageMeter()

    # Batches
    for i, (images, labels) in enumerate(train_loader):

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 224, 224)
        labels = labels.squeeze().to(device)

        # Forward prop.
        predicted_scores = model(images)  # (N, n_classes)

        # Loss
        loss = F.cross_entropy(predicted_scores, labels)  # scalar

        # Accuracy
        _, preds = predicted_scores.max(1)
        acc = float((preds == labels).sum()) / images.size(0)

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()
        losses.update(loss.item(), images.size(0))
        accs.update(acc, images.size(0))

        # Update lr, momentum per iteration
        scheduler.step()

        # Print status
        if i % print_freq == 0:
            print('Iteration: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.4f} ({acc.avg:.4f})\t'.format(i, len(train_loader),
                                                               loss=losses, acc=accs))


def validate(val_loader, model):
    """
    One epoch's validation.

    :param val_loader: DataLoader for validation data
    :param model: model

    :return: average validation loss
    """
    model.eval()  # eval mode disables dropout

    losses = AverageMeter()
    accs = AverageMeter()

    # Prohibit gradient computation explicity because I had some problems with memory
    with torch.no_grad():
        # Batches
        for i, (images, labels) in enumerate(val_loader):

            # Move to default device
            images = images.to(device)  # (N, 3, 224, 224)
            labels = labels.squeeze().to(device)

            # Forward prop.
            predicted_scores = model(images)  # (N, n_classes)

            # Loss
            loss = F.cross_entropy(predicted_scores, labels)
            losses.update(loss.item(), images.size(0))

            # Accuracy
            _, preds = predicted_scores.max(1)
            acc = float((preds == labels).sum()) / images.size(0)
            accs.update(acc, images.size(0))


    print('* Val Loss - {loss.avg:.3f}'.format(loss=losses))
    print('* Val Acc - {acc.avg:.3f}'.format(acc=accs))

    return losses.avg


def get_preds(test_loader, model):
    """
    Inference for test set.

    :param test_loader: DataLoader for test data
    :param model: model

    :return: predictions
    """
    model.eval()  # eval mode disables dropout

    file_names = []
    predictions = []
    with torch.no_grad():
        # Batches
        for i, images in enumerate(test_loader):

            # Move to default device
            images = images.to(device)  # (N, 3, 224, 224)

            # Forward prop.
            predicted_scores = model(images)  # (N, n_classes)

            predictions.append(predicted_scores)

    predictions = F.softmax(torch.cat(predictions, 0), 1)

    return predictions


def TTA(test_datasets, model, beta=0.4):
    """
    Test time augmentation prediction

    Inputs:
    - test_datasets: list of original/transformed datasets
    - model: classifier
    - beta: param for ratio of pred of original dataset in final results

    Return:
    - predictions
    """
    original_pred = 0
    transformed_pred = 0
    for i in range(len(test_datasets)):
        test_loader = DataLoader(test_datasets[i], batch_size=batch_size,
                                 shuffle=False, num_workers=workers)
        preds = get_preds(test_loader, model)
        if i == 0:
            original_pred += preds.cpu().numpy()
        else:
            transformed_pred += preds.cpu().numpy() / (len(test_datasets)-1)

    predictions = beta * original_pred + (1-beta) * transformed_pred

    return predictions


if __name__ == '__main__':
    main()
