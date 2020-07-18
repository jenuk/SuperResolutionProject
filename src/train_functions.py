import torch
from tqdm.auto import tqdm

def train_model(model, criterion, optimizer, loader, crit_regularization = None, factor_regularization=1e-4, use_gpu=False):
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
        l1 = criterion(pred_upscaled, torch.ones(batch_size//2, 1, device=pred_upscaled.device))

        pred_orig = discriminator(orig)
        l2 = criterion(pred_orig, torch.zeros(batch_size//2, 1, device=pred_orig.device))

        loss = l1 + l2
        loss.backward()
        optimizer.step()




def train_mix(model, discriminator, crit_model, opt_model, opt_disc, loader, crit_regularization = None, factor_regularization=1e-4, disc_freq = 1, use_gpu=False):

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
            l1 = crit_disc(pred_upscaled, torch.ones(batch_size//2, 1, device=pred_upscaled.device))

            pred_orig = discriminator(orig)
            l2 = crit_disc(pred_orig, torch.zeros(batch_size//2, 1, device=pred_orig.device))

            loss = l1 + l2
            loss.backward()
            opt_disc.step()