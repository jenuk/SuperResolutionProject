# Files

- scripts
    - `resize_visual`: illustrates the resize process step wise and calculates another image that scales down to the same image.
    - `data/download_ffhq` downloads the ffhq dataset, adapted unchanged from https://github.com/NVlabs/ffhq-dataset (needs to be executed before training)
    - `data/make_comp`: samples random images without copyright from validation and test set, to easily asses visual quality (needs to be executed before testing)
    - `data/make_smaller`: downscales images for faster dataloading (needs to be executed before training, after download)

- Notebooks:
    - `train.ipynb`: Contains all setup to train model and discriminator, as well as functions that measure performance on the validation set to check progress.
    - `test.ipynb`: load trained model to measure performance on test set.

- Implentations used somewhere else
    - `classic_model.py`: Model for bicubic interpolation (I have just noticed this is worse version of `torch.nn.Upsample`)
    - `discriminator.py`: pytorch implementation of my discriminator.
    - `facedataset.py`: Contains a `torch.utils.Dataset` class that interfaces the FFHQ dataset.
    - `layers.py`: two `torch.nn.Module`s for gaussian blur and a residual layer
    - `loss_functions.py`: contains all loss functions used in the project that were not already implemented in pytorch.
    - `metrics`: implements PSNR
    - `model.py`: implements my model.
    - `train_functions.py`: functions that train model and discriminator for one epoch.

- `weights/`: weights of trained models in the format `{name}_crit_{crit}_reg_{reg}_{input_size}`
    - name is in [model, disc] descripes type
    - crit is in [mse, perception, disc] descripes loss, perception uses vgg and disc is perceptual loss with discriminator
    - reg is in [none, tv, disc], tv is total variation, disc is discriminator loss
    - input_size with which the model was trained, upscaling factor is 256/input_size