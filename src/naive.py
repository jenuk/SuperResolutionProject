import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as F_tr
from PIL import Image, ImageFilter
from tqdm.auto import trange

from layers import GaussianBlur

img_no = 29
img_path = f"data/comp_test/{img_no}.png"

img_pil = Image.open(img_path)
img_pil = img_pil.resize((256, 256))
img = F_tr.to_tensor(img_pil)

gb = GaussianBlur(9, 3.0)

blurred = gb(img.unsqueeze(0))
target = blurred[..., ::2, ::2]

target.detach()

F_tr.to_pil_image(target.squeeze()).show()

res = nn.Parameter(torch.normal(0.5, 0.1, (1, 3, 256, 256)))
opt = torch.optim.Adam([res], lr=1e-3)

flag = True
while flag:
    for i in trange(1000):
        opt.zero_grad()
        out = gb(res)[..., ::2, ::2]
        loss = F.mse_loss(out, target)
        loss.backward()
        opt.step()

    res.data.copy_(torch.clamp(torch.round(res*255)/255, 0, 1))
    out = gb(res)[..., ::2, ::2]
    loss = F.mse_loss(out, target)
    print(loss)
    max_diff = torch.max(torch.abs(out - target))
    print(max_diff)

    flag = max_diff > 1/(2*256)

res_pil = F_tr.to_pil_image(res.squeeze())
res_pil.save(f"results/naive_{img_no}.png")
res_pil.show()

test = F_tr.to_tensor(Image.open(f"results/naive_{img_no}.png"))
test2 = gb(test.unsqueeze(0))[..., ::2, ::2]
print(torch.max(torch.abs(test2 - target)))
F_tr.to_pil_image(test2.squeeze()).show()