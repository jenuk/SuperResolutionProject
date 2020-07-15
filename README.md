# SuperResolutionProject
My project for the Deep Vision lecture summer semester 2020 by Prof. Ommer.


# Tests so far

| Broad Idea | Size | Training Time | PSNR | Additional Notes | Commit |
| --- | --- | --- | --- | --- | --- |
| Nearest Neighbor | 32 -> 64 | | 60.16 ± 3.32 | | [5d130db](https://github.com/jenuk/SuperResolutionProject/tree/5d130db) |
| Bilinear | | | 61.97 ± 3.53 | |  [7c7dc6e](https://github.com/jenuk/SuperResolutionProject/tree/7c7dc6e) |
| Bicubic | | | 65.56 ± 3.65 | |  [c5e8805](https://github.com/jenuk/SuperResolutionProject/tree/c5e8805) |
| Lanczos | | | 67.21 ± 3.71 | |  [2c246da](https://github.com/jenuk/SuperResolutionProject/tree/2c246da) |
| SRCNN | 32 -> 64 | 6 epochs| 68.94 ± 3.88 | 4 layers | [95ad969](https://github.com/jenuk/SuperResolutionProject/tree/95ad969)|
| SRCNN | 64 -> 128 | | 68.94 ± 4.54 | trained on 32 -> 64 (above) | [95ad969](https://github.com/jenuk/SuperResolutionProject/tree/95ad969) |
| SRCNN | 64 -> 128 | | 69.04 ± 4.56 | | [95ad969](https://github.com/jenuk/SuperResolutionProject/tree/95ad969) |
|SRCNN + Discriminator | 32 -> 64 | 6 epochs | 67.27 ± 3.71 | first epoch with MSE | [3fc544c](https://github.com/jenuk/SuperResolutionProject/tree/3fc544c) |
