import time
import torch.backends.cudnn as cudnn
import torch
from torch.utils.data import DataLoader
from .datasets import NCFMdataset
from .model import *
from ..one_cycle_lr import OneCycleScheduler
from ..utils import *

# Data parameters
data_path = './'  # path of data files

# Model parameters
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters
pretrained = 'pretrained_ssd300.pth'  # path to model checkpoint, None if none
batch_size = 8  # batch size
epochs = 6  # number of epochs to run
lr = 3e-4  # learning rate

print_freq = 120  # print training or validation status every __ batches
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
    model = SSD300(n_classes=21)
    model.load_state_dict(torch.load(pretrained))

    pred_convs = PredictionConvolutions(n_classes)
    for c in pred_convs.children():
        if isinstance(c, nn.Conv2d):
            nn.init.xavier_uniform_(c.weight)
            nn.init.constant_(c.bias, 0.)
    model.pred_convs = pred_convs
    model.n_classes = 2

    base_params = [p for p in model.base.parameters()]
    aux_params = [p for p in model.aux_convs.parameters()]
    pred_params = [p for p in model.pred_convs.parameters()]
    for p in base_params: p.requires_grad = True
    for p in aux_params: p.requires_grad = True
    for p in pred_params: p.requires_grad = True

    optimizer = torch.optim.Adam(params=[{'params': pred_params}])
    optimizer.add_param_group({'params':aux_params})
    optimizer.add_param_group({'params':base_params})

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # Custom dataloaders
    train_dataset = NCFMdataset(data_path, split='train')
    val_dataset = NCFMdataset(data_path, split='valid')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=train_dataset.collate_fn, num_workers=workers,
                              pin_memory=True)  # note that we're passing the collate function here
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=val_dataset.collate_fn, num_workers=workers,
                            pin_memory=True)

    # Scheduler for one-cycle policy
    ft_lrs = [lr, lr/10, lr/100]
    scheduler = OneCycleScheduler(optimizer, epochs, train_loader, max_lr=ft_lrs)

    # Epochs
    for epoch in range(epochs):
        print(f'Epoch: {epoch+1}')

        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              scheduler=scheduler)

        # One epoch's validation
        val_loss = validate(val_loader=val_loader,
                            model=model,
                            criterion=criterion)

        # Did validation loss improve?
        if epoch==0: best_loss = val_loss
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)

        # Save checkpoint
        if is_best:
            print(f'Save the better model at epoch {epoch+1}')
            torch.save(model.state_dict(), data_path+f'ssd300_ft_epk_{epochs}.pth')

        print()


def train(train_loader, model, criterion, optimizer, scheduler):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param scheduler: lr, mom scheduler
    """
    model.train()  # training mode enables dropout

    losses = AverageMeter()  # loss

    # Batches
    for i, (images, boxes, labels) in enumerate(train_loader):

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()
        losses.update(loss.item(), images.size(0))

        # Update lr, momentum per iteration
        scheduler.step()

        # Print status
        if i % print_freq == 0:
            print('Iteration: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(train_loader),
                                                                  loss=losses))


def validate(val_loader, model, criterion):
    """
    One epoch's validation.

    :param val_loader: DataLoader for validation data
    :param model: model
    :param criterion: MultiBox loss

    :return: average validation loss
    """
    model.eval()  # eval mode disables dropout

    losses = AverageMeter()

    # Prohibit gradient computation explicity because I had some problems with memory
    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels) in enumerate(val_loader):

            # Move to default device
            images = images.to(device)  # (N, 3, 300, 300)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            # Forward prop.
            predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

            # Loss
            loss = criterion(predicted_locs, predicted_scores, boxes, labels)
            losses.update(loss.item(), images.size(0))

    print('* Val Loss - {loss.avg:.3f}'.format(loss=losses))

    return losses.avg


if __name__ == '__main__':
    main()
