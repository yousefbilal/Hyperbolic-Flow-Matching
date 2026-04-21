from notebook_utils.pmath import dist0
import geoopt.manifolds.stereographic.math as gmath
import torch
import os
import sys
from notebook_utils.hae.models.hae_inference import hae
from argparse import Namespace
from PIL import Image
sys.path.insert(0,'/home/fvaleau/HAE/Visualization/notebook_utils/hae')


# ----------------------------------------------------------
# Load HAE decoder
# ----------------------------------------------------------
model_path = '/home/fvaleau/HAE/pretrained_models/hae_animalfaces.pt'
ckpt = torch.load(model_path, map_location='cpu')
opts = ckpt['opts']
# use the first GPU 
opts['device'] = 'cuda:0'
opts['checkpoint_path'] = model_path
if 'learn_in_w' not in opts:
    opts['learn_in_w'] = False

opts = Namespace(**opts)
net = hae(opts).cuda().eval()
print('Model successfully loaded!')

# ----------------------------------------------------------
# Utility: convert tensor → PIL
# ----------------------------------------------------------
def tensor2im(var):
    var = var.cpu().detach().transpose(0,2).transpose(0,1).numpy()
    var = (var + 1) / 2
    var = (var * 255).clip(0,255)
    return Image.fromarray(var.astype("uint8"))

# ----------------------------------------------------------
# Decode latent vector z
# ----------------------------------------------------------
def decode(z):
    with torch.no_grad():
        img, _, _, _, _ = net.forward(
            x=z.unsqueeze(0),
            codes=None,
            batch_size=1,
            input_feature=True,
            input_code=False
        )
    return tensor2im(img.squeeze(0))


trajs = torch.load("/home/fvaleau/HAE/Visualization/trajs_animals_last").cuda()
os.makedirs("traj_images/animals_last", exist_ok=True)

# ----------------------------------------------------------
# Decode all trajectories and create grids
# ----------------------------------------------------------
for idx in range(trajs.shape[0]):

    print(f"Processing sample {idx+1}/{trajs.shape[0]}")
    z_traj = trajs[idx]     # shape [11, dim]

    decoded_row = []

    for j in range(z_traj.shape[0]):
        z = z_traj[j]
        # decode to image
        img = decode(z)
        decoded_row.append(img)

    # ------------------------------------------------------
    # Build image grid: 1 row × 11 columns
    # ------------------------------------------------------
    widths, heights = zip(*(img.size for img in decoded_row))
    total_width = sum(widths)
    max_height = max(heights)

    grid = Image.new("RGB", (total_width, max_height))

    x_offset = 0
    for img in decoded_row:
        grid.paste(img, (x_offset, 0))
        x_offset += img.size[0]

    # Save grid
    out_path = f"traj_images/animals_last/sample_{idx+1}.png"
    grid.save(out_path)
    print(f"Saved {out_path}")


print("Saved flow visualizations in flow_samples/")
