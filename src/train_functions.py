import torch
from tqdm.auto import tqdm

def train_model(model, criterion, optimizer, loader, crit_regularization = None, factor_regularization=1e-4, use_gpu=False):
    """
    trains the model for one epoch.

    Parameters
    ----------
    model : torch.nn.Module
        model to train
    criterion : torch.nn.Module
        criterion used to calculate loss between output and target
    optimizer : torch.optim.Optimizer
        used for optimization steps
    loader : torch.utils.data.DataLoader
        dataloader that provides img, target for training
    crit_regularization : torch.nn.Module or None, optional
        Regularization, None for no reularization, default None
    factor_regularization: float, optional
        the factor by which the regularization is multplied, default `1e-4`
    use_gpu: bool
        whether gpu is available.
    """
    model.train()

    for k, (img, target) in tqdm(enumerate(loader), total=len(loader)):
        if use_gpu:
            img = img.cuda()
            target = target.cuda()

        optimizer.zero_grad()

        out = model(img)
        loss = criterion(out, target)
        if crit_regularization is not None:
            loss = loss + factor_regularization * crit_regularization(out)

        loss.backward()
        optimizer.step()


def train_discriminator(model, discriminator, optimizer, loader, use_gpu=False):
    """
    trains the discriminator for one epoch.

    Parameters
    ----------
    model : torch.nn.Module
        model for upscaling
    discriminator : torch.nn.Module
        discriminator to train
    optimizer : torch.optim.Optimizer
        used for optimization steps
    loader : torch.utils.data.DataLoader
        dataloader that provides img, target for training
    use_gpu: bool
        whether gpu is available.
    """

    model.eval()
    discriminator.train()

    criterion = torch.nn.BCELoss()
    batch_size = loader.batch_size

    for k, (img, target) in tqdm(enumerate(loader), total=len(loader)):
        if use_gpu:
            img = img.cuda()
            target = target.cuda()

        with torch.no_grad():
            upscaled = model(img[:batch_size//2])
            orig = target[batch_size//2:]

        optimizer.zero_grad()

        pred_upscaled = discriminator(upscaled)
        l1 = criterion(pred_upscaled, torch.zeros(batch_size//2, 1, device=pred_upscaled.device))

        pred_orig = discriminator(orig)
        l2 = criterion(pred_orig, torch.ones(batch_size//2, 1, device=pred_orig.device))

        loss = l1 + l2
        loss.backward()
        optimizer.step()




def train_mix(model, discriminator, crit_model, opt_model, opt_disc, loader, crit_regularization = None, factor_regularization=1e-4, disc_freq = 1, use_gpu=False):
    """
    trains the model and discriminator alternatingly for one epoch.

    Parameters
    ----------
    model : torch.nn.Module
        model to train
    discriminator : torch.nn.Module
        discriminator to train
    crit_model : torch.nn.Module
        criterion used to calculate loss between output and target
    opt_model : torch.optim.Optimizer
        used for optimization steps of model
    opt_disc : torch.optim.Optimizer
        used for optimization steps of discriminator
    loader : torch.utils.data.DataLoader
        dataloader that provides img, target for training
    crit_regularization : torch.nn.Module or None, optional
        Regularization, None for no reularization, default None
    factor_regularization: float, optional
        factor used to reduce the regularization, default `1e-4`
    disc_freq : int, optional
        how often to update the discriminator per model update, default `1`
    use_gpu: bool
        whether gpu is available.
    """

    crit_disc = torch.nn.BCELoss()
    loop_len = disc_freq + 1
    batch_size = loader.batch_size

    for k, (img, target) in tqdm(enumerate(loader), total=len(loader)):
        if use_gpu:
            img = img.cuda()
            target = target.cuda()

        if k%loop_len == 0:
            # train model
            model.train()
            discriminator.eval()

            opt_model.zero_grad()
            out = model(img)
            loss = crit_model(out, target)
            if crit_regularization is not None:
                loss = loss + factor_regularization*crit_regularization(out)
            loss.backward()

            opt_model.step()

        else:
            # train discriminator
            model.eval()
            discriminator.train()

            with torch.no_grad():
                upscaled = model(img[:batch_size//2])
                orig = target[batch_size//2:]

            opt_disc.zero_grad()

            pred_upscaled = discriminator(upscaled)
            l1 = crit_disc(pred_upscaled, torch.zeros(batch_size//2, 1, device=pred_upscaled.device))

            pred_orig = discriminator(orig)
            l2 = crit_disc(pred_orig, torch.ones(batch_size//2, 1, device=pred_orig.device))

            loss = l1 + l2
            loss.backward()
            opt_disc.step()