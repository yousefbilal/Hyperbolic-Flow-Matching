# we utilize geoopt package for hyperbolic calculation
import geoopt.manifolds.stereographic.math as gmath

import glob
import random
import sys
import os
import matplotlib.pyplot as plt
import torch
from argparse import Namespace
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from notebook_utils.pmath import *
from notebook_utils.pmath import poincare_mean, dist0

# please specify the cuda devices
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# importing our model path
sys.path.insert(0,'/home/fvaleau/HAE/Visualization/notebook_utils/hae')

# import updated pSp for inference
from models import hae

print(torch.__version__) # Get PyTorch and CUDA version
print(torch.cuda.is_available()) # Check that CUDA works
torch.cuda.device_count() # Check how many CUDA capable devices you have
torch.cuda.set_device(0)

# load model for different datasets

# animal faces
#model_path = 'PATH_TO/hae_animalfaces.pt'

# flowers 
model_path = '/home/fvaleau/HAE/pretrained_models/hae_flowers.pt'

# vgg faces
#model_path = 'PATH_TO/hae_vggfaces.pt'

ckpt = torch.load(model_path, map_location='cpu')
opts = ckpt['opts']
# use the first GPU normally
opts['device'] = 'cuda:0'
opts['checkpoint_path'] = model_path
if 'learn_in_w' not in opts:
    opts['learn_in_w'] = False

# instantialize model with checkpoints and args
opts = Namespace(**opts)
net = hae(opts)
net.eval()
net.cuda()
print('Model successfully loaded!')


radiuss = 6.2126

# useful function for visualization
def tensor2im(var):
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))

# use default transform in training/inference

transform_inf = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

# set manual seed only in this cell - choosing images manually is also available, please follow the same data type shown here
# sampling image
random.seed(55)
files = glob.glob("/home/fvaleau/HAE/proj/flowers/*/*/*.jpg")
sampled_imgs = random.sample(files, 16)

#print(sampled_imgs)
image = []
for i in range(len(sampled_imgs)):        
    image.append(Image.open(sampled_imgs[i]).convert('RGB'))

# this cell to transform input images to tensor
input_image = []
for i in image:
    input_image.append(transform_inf(i).cuda())
input_tensor = torch.stack(input_image)

# show loaded images
fig = plt.figure(figsize = (50,20), dpi = 30)
gs = fig.add_gridspec(1, 4)
for i in range(4):
    fig.add_subplot(gs[0,i])
    plt.axis('off')
    plt.imshow(tensor2im(input_image[i].squeeze(0)))

plt.savefig('proj/outputs/flower_images_4.png',dpi = 300)

# important and necessary! to get the embeddings for further experiments
with torch.no_grad():
    images, logits, feature_dist, codes, feature_euc = net.forward(x = input_tensor, batch_size = 2, input_code = False)


# rescale function
def rescale(target_radius, x):
    r_change = target_radius/dist0(gmath.mobius_scalar_mul(r = torch.tensor(1), x = x, k = torch.tensor(-1.0)))
    return gmath.mobius_scalar_mul(r = r_change, x = x, k = torch.tensor(-1.0))

# function for generating images with fixed radius (also contains raw geodesic images of 'shorten' images, and stretched images to boundary)
def geo_interpolate_fix_r(x,y,interval,target_radius):
    feature_geo = []
    feature_geo_normalized = []
    images_to_plot_raw_geo = []
    images_to_plot_target_radius = []
    images_to_plot_boundary = []
    dist_to_start = []
    target_radius_ratio = torch.tensor(target_radius/6.2126)
    geodesic_start_short = gmath.mobius_scalar_mul(r = target_radius_ratio, x = x, k = torch.tensor(-1.0))
    geodesic_end_short = gmath.mobius_scalar_mul(r = target_radius_ratio, x = y, k = torch.tensor(-1.0))
    for i in interval:
        # this is raw image on geodesic, instead of fixed radius
        feature_geo_current = gmath.geodesic(t = torch.tensor(i), x = geodesic_start_short, y = geodesic_end_short, k = torch.tensor(-1.0))

        # here we fix the radius and don't revert them now
        r_change = target_radius/dist0(gmath.mobius_scalar_mul(r = torch.tensor(1), x = feature_geo_current, k = torch.tensor(-1.0)))
        feature_geo.append(feature_geo_current)
        feature_geo_current_target_radius = gmath.mobius_scalar_mul(r = r_change, x = feature_geo_current, k = torch.tensor(-1.0))
        feature_geo_normalized.append(feature_geo_current_target_radius)
        dist = gmath.dist(geodesic_start_short, feature_geo_current_target_radius, k = torch.tensor(-1.0))
        dist_to_start.append(dist)

        # here is to revert the feature to boundary
        r_change_to_boundary = 6.2126/dist0(gmath.mobius_scalar_mul(r = torch.tensor(1), x = feature_geo_current, k = torch.tensor(-1.0)))
        feature_geo_current_target_boundary = gmath.mobius_scalar_mul(r = r_change_to_boundary, x = feature_geo_current, k = torch.tensor(-1.0))
        
        with torch.no_grad():
            image_raw_geo, _, _, _, _ = net.forward(x = feature_geo_current.unsqueeze(0), codes=None, batch_size = 1, input_feature=True, input_code = False)
            image, _, _, _, _ = net.forward(x = feature_geo_current_target_radius.unsqueeze(0), codes=None, batch_size = 1, input_feature=True, input_code = False)
            image_boundary, _, _, _, _ = net.forward(x = feature_geo_current_target_boundary.unsqueeze(0), codes=None, batch_size = 1, input_feature=True, input_code = False)
        images_to_plot_raw_geo.append(image_raw_geo)
        images_to_plot_target_radius.append(image)
        images_to_plot_boundary.append(image_boundary)
    
    return images_to_plot_raw_geo, images_to_plot_target_radius, images_to_plot_boundary, dist_to_start

# function for generating images with fixed radius with optional latent codes list output

def geo_interpolate_fix_r_with_codes(x,y,interval,target_radius):
    # please use this with batch_size = 1
    feature_geo = []
    feature_geo_normalized = []
    images_to_plot_raw_geo = []
    images_to_plot_target_radius = []
    images_to_plot_boundary = []
    dist_to_start = []
    target_radius_ratio = torch.tensor(target_radius/6.2126)
    geodesic_start_short = gmath.mobius_scalar_mul(r = target_radius_ratio, x = x, k = torch.tensor(-1.0))
    geodesic_end_short = gmath.mobius_scalar_mul(r = target_radius_ratio, x = y, k = torch.tensor(-1.0))
    for i in interval:
        # this is raw image on geodesic, instead of fixed radius
        feature_geo_current = gmath.geodesic(t = torch.tensor(i), x = geodesic_start_short, y = geodesic_end_short, k = torch.tensor(-1.0))

        # here we fix the radius and don't revert them now
        r_change = target_radius/dist0(gmath.mobius_scalar_mul(r = torch.tensor(1), x = feature_geo_current, k = torch.tensor(-1.0)))
        feature_geo.append(feature_geo_current)
        feature_geo_current_target_radius = gmath.mobius_scalar_mul(r = r_change, x = feature_geo_current, k = torch.tensor(-1.0))
        feature_geo_normalized.append(feature_geo_current_target_radius)
        dist = gmath.dist(geodesic_start_short, feature_geo_current_target_radius, k = torch.tensor(-1.0))
        dist_to_start.append(dist)
        #print(feature_geo_current_target_radius.norm())

        # here is to revert the feature to boundary
        r_change_to_boundary = 6.2126/dist0(gmath.mobius_scalar_mul(r = torch.tensor(1), x = feature_geo_current, k = torch.tensor(-1.0)))
        feature_geo_current_target_boundary = gmath.mobius_scalar_mul(r = r_change_to_boundary, x = feature_geo_current, k = torch.tensor(-1.0))
        #print(feature_geo_current_target_boundary.norm())

        # now codes do not affect outputs
        with torch.no_grad():
            image_raw_geo, _, _, _, _ = net.forward(x = feature_geo_current.unsqueeze(0), codes=None, batch_size = 1, input_feature=True, input_code = False)
            image, _, _, codes_target_radius, _ = net.forward(x = feature_geo_current_target_radius.unsqueeze(0), codes=None, batch_size = 1, input_feature=True, input_code = False)
            image_boundary, _, _, codes_boundary, _ = net.forward(x = feature_geo_current_target_boundary.unsqueeze(0), codes=None, batch_size = 1, input_feature=True, input_code = False)
        images_to_plot_raw_geo.append(image_raw_geo)
        images_to_plot_target_radius.append(image)
        images_to_plot_boundary.append(image_boundary)
    
    return images_to_plot_raw_geo, images_to_plot_target_radius, images_to_plot_boundary, dist_to_start, [codes_target_radius, codes_boundary, feature_geo_current_target_radius, feature_geo_current_target_boundary]

# the function defines easy perturbation on any given radius

def generate_perturbation_r_with_raw_inv(x, target_radius, interval, seed, size):
    # 3 arguments, raw image feature, target radius and interval(actually the ratio).
    images_perturbed = []
    dist_perturbed = []
    torch.manual_seed(seed = seed)
    perturb = torch.rand(6,512).cuda()
    for i in range(size):
        target_rad_perturb = 6.2126
        ratio = target_rad_perturb/dist0(perturb[i])
        perturb_current = gmath.mobius_scalar_mul(r = ratio, x = perturb[i], k = torch.tensor(-1.0))
        _, images_to_plot_target_radius, _, dist_to_start = geo_interpolate_fix_r(x = x,y = perturb_current, interval = interval ,target_radius = target_radius)
        print(dist_to_start)
        dist_perturbed.append(dist_to_start[0])
        images_perturbed.append(images_to_plot_target_radius[0])

    raw_image,_,_,_ = geo_interpolate_fix_r(x = x,y = perturb_current, interval = [0] ,target_radius = target_radius)
    images_perturbed.insert(0, raw_image[0])
    return images_perturbed, dist_perturbed

# this further function allows using specific target image as perturbation

def generate_perturbation_r_with_raw_inv_pick(x, y, target_radius, interval, seed):
    # 3 arguments, raw image feature, target radius and interval(actually the ratio).
    images_perturbed = []
    dist_perturbed = []
    torch.manual_seed(seed = seed)
    #perturb = torch.rand(6,512).cuda()
    perturb = y
    for i in range(len(y)):
        target_rad_perturb = 6.2126
        ratio = target_rad_perturb/dist0(perturb[i])
        perturb_current = gmath.mobius_scalar_mul(r = ratio, x = perturb[i], k = torch.tensor(-1.0))


        if False:
            with torch.no_grad():
                image, _, _, _, _ = net.forward(x = perturb_current.unsqueeze(0), codes=None, batch_size = 1, input_feature=True, input_code = False)
            
            fig = plt.figure(figsize = (5,5))
            gs = fig.add_gridspec(1, 1)
            for i in range(1):
                if i == 0:
                    fig.add_subplot(gs[0,i])
                    plt.axis('off')
                    plt.title(f'perturb = {target_rad_perturb}')
                    plt.imshow(tensor2im(image.squeeze(0)))

        #interval = [0.42]
        _, images_to_plot_target_radius, _, dist_to_start = geo_interpolate_fix_r(x = x,y = perturb_current, interval = interval ,target_radius = target_radius)
        print(dist_to_start)
        dist_perturbed.append(dist_to_start[0])
        images_perturbed.append(images_to_plot_target_radius[0])

    raw_image,_,_,_ = geo_interpolate_fix_r(x = x,y = perturb_current, interval = [0] ,target_radius = target_radius)
    images_perturbed.insert(0, raw_image[0])
    return images_perturbed, dist_perturbed

fig = plt.figure(figsize = (20,80), dpi = 300)
gs = fig.add_gridspec(1, 8)
for i in range(8):
    fig.add_subplot(gs[0,i])
    plt.axis('off')
    plt.imshow(tensor2im(images[i].squeeze(0)))
#plt.savefig('proj/outputs/inversion_8_images.png',dpi = 300)

# calculate pair-wise distance between images
dist_pairwise = torch.zeros(8,8)
for i in range(8):
    for j in range(8):
        dist_pairwise[i,j] = gmath.dist(x = feature_dist[i], y = feature_dist[j], k = torch.tensor(-1.0))

plt.figure(figsize=(8, 6))
plt.matshow(dist_pairwise)
plt.colorbar()
plt.title('distance between images')
save_path = "proj/outputs/pairwise_distance.png"
#plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()

# save mean of paired images, we assume there are three layers while merging, here is layer1
mean_l1_raw = []
mean_l1_rescale = []
images_to_plot = []
images_to_raw = []

radius = 6.2126

for i in range(4):
    current_mean = poincare_mean(x = torch.stack([feature_dist[i*2],feature_dist[i*2+1]]))
    mean_l1_raw.append(current_mean)
    print(dist0(current_mean))
    current_mean_rescale = rescale(4, current_mean)
    mean_l1_rescale.append(current_mean_rescale)
    
    with torch.no_grad():
        image, _, _, _, _ = net.forward(x = current_mean_rescale.unsqueeze(0), codes=None, batch_size = 1, input_feature=True, input_code = False)
        image_raw, _, _, _, _ = net.forward(x = current_mean.unsqueeze(0), codes=None, batch_size = 1, input_feature=True, input_code = False)

    images_to_plot.append(image)
    images_to_raw.append(image_raw)
    
# save mean of paired images for layer 2
mean_l2_raw = []
mean_l2_rescale = []
images_to_plot_l2 = []
images_to_raw_l2 = []

radius = 6.2126

for i in range(2):
    current_mean = poincare_mean(x = torch.stack([mean_l1_raw[i*2],mean_l1_raw[i*2+1]]))
    mean_l2_raw.append(current_mean)
    print(dist0(current_mean))
    current_mean_rescale = rescale(2.5, current_mean)
    mean_l2_rescale.append(current_mean_rescale)
    
    with torch.no_grad():
        image, _, _, _, _ = net.forward(x = current_mean_rescale.unsqueeze(0), codes=None, batch_size = 1, input_feature=True, input_code = False)
        image_raw, _, _, _, _ = net.forward(x = current_mean.unsqueeze(0), codes=None, batch_size = 1, input_feature=True, input_code = False)

    images_to_plot_l2.append(image)
    images_to_raw_l2.append(image_raw)

# save mean of paired images for layer 3
mean_l3_raw = []
mean_l3_rescale = []
images_to_plot_l3 = []
images_to_raw_l3 = []

radius = 6.2126

for i in range(1):
    current_mean = poincare_mean(x = torch.stack([mean_l2_raw[i*2],mean_l2_raw[i*2+1]]))
    mean_l3_raw.append(current_mean)
    
    current_mean_rescale = rescale(2.5, current_mean)
    mean_l3_rescale.append(current_mean_rescale)
    
    with torch.no_grad():
        image, _, _, _, _ = net.forward(x = current_mean_rescale.unsqueeze(0), codes=None, batch_size = 1, input_feature=True, input_code = False)
        image_raw, _, _, _, _ = net.forward(x = current_mean.unsqueeze(0), codes=None, batch_size = 1, input_feature=True, input_code = False)
        
    images_to_plot_l3.append(image)
    images_to_raw_l3.append(image_raw)

# final plot for all 3 layers

fig, ax = plt.subplots(3, 4, figsize = (40, 30), dpi = 300)

for row in [0]:
    for col in range(4):

        ax[row][col].tick_params(top = 'off', bottom = 'off', left = 'off', right = 'off', labelleft = 'on', labelbottom = 'on')
        ax[row][col].imshow(tensor2im(images_to_plot[col].squeeze(0)))
        ax[row][col].set_xticks([])
        ax[row][col].set_yticks([])
        #ax[row][col].set_xlabel('x label')
        if col == 0:
            ax[row][col].set_ylabel('Layer 1')
            
for row in [1]:
    for col in range(2):

        ax[row][col].tick_params(top = 'off', bottom = 'off', left = 'off', right = 'off', labelleft = 'on', labelbottom = 'on')
        ax[row][col].imshow(tensor2im(images_to_plot_l2[col].squeeze(0)))
        ax[row][col].set_xticks([])
        ax[row][col].set_yticks([])
        #ax[row][col].set_xlabel('x label')
        if col == 0:
            ax[row][col].set_ylabel('Layer 2')
            
for row in [2]:
    for col in range(1):
        ax[row][col].tick_params(top = 'off', bottom = 'off', left = 'off', right = 'off', labelleft = 'on', labelbottom = 'on')
        ax[row][col].imshow(tensor2im(images_to_plot_l3[col].squeeze(0)))
        ax[row][col].set_xticks([])
        ax[row][col].set_yticks([])
        #ax[row][col].set_xlabel('x label')
        if col == 0:
            ax[row][col].set_ylabel('Layer 3')

#plt.savefig('proj/outputs/merged_images_layer1_r4_layer2_r2.png',dpi = 300)

# try to sample a noise and rescale it to the target radius, the input will be: one source picture feature_dist (1*512 dim), perturbation feature_dist (n*512 dim) ,
# target radius (float <= 6.2126)
# and desired interval, ranging from 0 to 1.
images_r6, dist_r6 = generate_perturbation_r_with_raw_inv_pick(x = feature_dist[0], y=feature_dist[1:], target_radius = 6.2126, interval = [0.47], seed = 46)
images_r5, dist_r5 = generate_perturbation_r_with_raw_inv_pick(x = feature_dist[0], y=feature_dist[1:], target_radius = 5, interval = [0.49], seed = 48)
images_r4, dist_r4 = generate_perturbation_r_with_raw_inv_pick(x = feature_dist[0], y=feature_dist[1:], target_radius = 4, interval = [0.51], seed = 53)
images_r3, dist_r3 = generate_perturbation_r_with_raw_inv_pick(x = feature_dist[0], y=feature_dist[1:], target_radius = 3, interval = [0.9], seed = 100)

images_r6, dist_r6 = generate_perturbation_r_with_raw_inv(x = feature_dist[0], target_radius = 6.2126, interval = [0.445], seed = 52, size=6)
images_r5, dist_r5 = generate_perturbation_r_with_raw_inv(x = feature_dist[0], target_radius = 5, interval = [0.46], seed = 52, size=6)
images_r4, dist_r4 = generate_perturbation_r_with_raw_inv(x = feature_dist[0], target_radius = 4, interval = [0.47], seed = 52, size=6)
images_r3, dist_r3 = generate_perturbation_r_with_raw_inv(x = feature_dist[0], target_radius = 3, interval = [0.99], seed = 52, size=6)

def plot_one_row_with_dist(images, dist, radius):
    for col in range(len(images)):
        ax[row][col].tick_params(top = 'off', bottom = 'off', left = 'off', right = 'off', labelleft = 'on', labelbottom = 'on')
        '''
        if col == 0:
            ax[row][col].set_title('Inversion', fontsize = 16)
        else:
            ax[row][col].set_title('$d_i = %.2f$' % dist[col-1], fontsize = 16)
            '''
        ax[row][col].imshow(tensor2im(images[col].squeeze(0)))
        ax[row][col].set_xticks([])
        ax[row][col].set_yticks([])
        #ax[row][col].set_xlabel('x label')
        if col == 0:
            ax[row][col].set_ylabel(f'$target radius = {radius}$', fontsize = 28)
    return None

fig, ax = plt.subplots(4, len(images_r6), figsize = (40, 25),dpi = 500)

for row in [0]:
   plot_one_row_with_dist(images_r6, dist_r6, radius = 6)
for row in [1]:
   plot_one_row_with_dist(images_r5, dist_r5, radius = 5) 
for row in [2]:
   plot_one_row_with_dist(images_r4, dist_r4, radius = 4)
for row in [3]:
   plot_one_row_with_dist(images_r3, dist_r3, radius = 3)
fig.tight_layout()

#plt.savefig('proj/outputs/perturbation_flowers_25.png',dpi = 500)

# interpolation

# we calculate several points along the geodesic connecting two given points
# just use this for arbitrary radius, also we can project the images back to boundary (not necessary)
interval = [0,0.475,0.49,0.51,0.535,0.55,1]
_,images_to_plot_r1, feature_geo_boundary, _ = geo_interpolate_fix_r(feature_dist[8], feature_dist[7], interval,target_radius = 6.2126)
images_to_plot_r1.insert(0,input_image[8])
images_to_plot_r1.insert(len(images_to_plot_r1),input_image[7])

# for W+space interpolation only
# interval = [1, 0.8, 0.6, 0.4, 0.2, 0]
# images_to_plot = interpolation_codes(codes[0], codes[1], interval)
# images_to_plot.insert(0,input_image[0])
# images_to_plot.insert(len(images_to_plot),input_image[1])


# plot interpolation with original images
fig, ax = plt.subplots(1, len(interval)+2, figsize = (40, 10), dpi = 300)

for row in [0]:
    for col in range(len(interval)+2):
        if col == 0 or col == len(interval)+1 :
            ax[col].set_title(f'Raw Image', fontsize = 18)
        if col not in [0,len(interval)+1]:
            ax[col].set_title(f'$r_i = {interval[col-1]}$', fontsize = 18)
        ax[col].tick_params(top = 'off', bottom = 'off', left = 'off', right = 'off', labelleft = 'on', labelbottom = 'on')
        ax[col].imshow(tensor2im(images_to_plot_r1[col].squeeze(0)))
        ax[col].set_xticks([])
        ax[col].set_yticks([])
        #ax[row][col].set_xlabel('x label')
        if col == 0:
            ax[col].set_ylabel(f'Flowers', fontsize = 20)

fig.tight_layout()            
plt.savefig('proj/outputs/interpolation_flowers_2',dpi = 300)

# fixed image, varying radius
# this is used for generating figure of varying radius in our paper
images_to_plot = []
target_radii = [6.2126, 5, 4, 3, 2, 1, 0]
for i in target_radii:
    feature_rescaled = rescale(i, feature_dist[12])
    #print(feature_rescaled.norm())
    with torch.no_grad():
        image, _, _, _, _ = net.forward(x = feature_rescaled.unsqueeze(0), codes=None, batch_size = 1, input_feature=True, input_code = False)
    images_to_plot.append(image)

images_to_plot.insert(0,input_image[12])

fig, ax = plt.subplots(1, len(images_to_plot), figsize = (40, 10), dpi = 300)

for row in [0]:
    for col in range(len(images_to_plot)):
        if col == 0:
            ax[col].set_title(f'Raw Image', fontsize = 18)
        else:
            ax[col].set_title(f'$radius = {target_radii[col-1]}$', fontsize = 18)
        ax[col].tick_params(top = 'off', bottom = 'off', left = 'off', right = 'off', labelleft = 'on', labelbottom = 'on')
        ax[col].imshow(tensor2im(images_to_plot[col].squeeze(0)))
        ax[col].set_xticks([])
        ax[col].set_yticks([])
        #ax[row][col].set_xlabel('x label')
        if col == 0:
            ax[col].set_ylabel(f'VGG Faces', fontsize = 20)
fig.tight_layout()
plt.savefig('proj/outputs/radius_animals_7',dpi = 300)

# Sampling images for FID

random.seed(47)
files = glob.glob("/home/fvaleau/HAE/proj/flowers/*/*.jpg")
sampled_imgs = random.sample(files, 1020)
#print(sampled_imgs)
image = []
for i in range(len(sampled_imgs)):        
    image.append(Image.open(sampled_imgs[i]).convert('RGB'))

# to make each image with the same perturbation, we need to fix it before run the function over again, here only 1 perturbation will be generated!
def generate_1_perturbation_r_with_raw_inv(x, target_radius, interval, perturb):
    # (optional 4) 3 arguments, raw image feature, target radius and interval(actually the ratio).
    perturb = perturb 
    target_rad_perturb = 6.2126
    ratio = target_rad_perturb/dist0(perturb)

    perturb_current = gmath.mobius_scalar_mul(r = ratio, x = perturb, k = torch.tensor(-1.0))
    _, images_to_plot_target_radius, images_to_plot_boundary, dist_to_start, codes_perturb = geo_interpolate_fix_r_with_codes(x = x,y = perturb_current, interval = interval ,target_radius = target_radius)
    #codes_target_radius, codes_boundary, feature_target_radius, feature_boundary = codes_perturb
    #print(dist_to_start,codes_target_radius, codes_boundary, feature_target_radius, feature_boundary)
    
    # give inversion of target radius
    inversion,_,_,_ ,codes_inv = geo_interpolate_fix_r_with_codes(x = x,y = perturb_current, interval = [0] ,target_radius = target_radius)
    # inv_codes_target_radius, inv_codes_boundary, inv_feature_target_radius, inv_feature_boundary
    
    return images_to_plot_boundary, images_to_plot_target_radius, inversion, codes_perturb, codes_inv

import numpy as np
from tqdm import tqdm
torch.manual_seed(seed = 45)

i = 1
# speficy datasets, choosing from animals, flowers and faces
dataset_name = 'flowers'
radius = [6, 5.5, 5, 4.5, 4, 3.5, 3]
interval = [0.42, 0.437, 0.45, 0.475, 0.49, 0.55, 0.7]
distances = list_three = np.zeros((len(radius), 18, len(image)), dtype = float)
for img in tqdm(image):
    input_image = transform_inf(img).cuda()
    # here, we set the random purturb to be equal for one image and its re-scaled variants (and will change for the next image)
    perturb = torch.rand(512).cuda()
    with torch.no_grad():
        image_inversion, logits, feature_dist, codes, feature_euc = net.forward(x = input_image.unsqueeze(0), batch_size = 1, input_code = False)   
    for j in range(len(radius)):
        
        img_perturb_boundary, img_perturb, inversion, codes_perturb, codes_inv = generate_1_perturbation_r_with_raw_inv(feature_dist.squeeze(0), radius[j], [interval[j]],perturb)
        '''
        img_perturb_boundary = tensor2im(img_perturb_boundary[0].squeeze(0))
        img_perturb_boundary.save(f'/proj/rcs-ssd/ll3504/datasets/outputs/{dataset_name}/perturbation_r{radius[j]}_boundary/{i}.jpg')
        
        img_perturb = tensor2im(img_perturb[0].squeeze(0))
        img_perturb.save(f'/proj/rcs-ssd/ll3504/datasets/outputs/{dataset_name}/perturbation_r{radius[j]}/{i}.jpg')
        
        inversion = tensor2im(inversion[0].squeeze(0))
        inversion.save(f'/proj/rcs-ssd/ll3504/datasets/outputs/{dataset_name}/inversion_r{radius[j]}/{i}.jpg')
        '''
        for m in range(18):
            distances[j][m][i-1] = np.linalg.norm(codes_perturb[1].squeeze()[m].cpu().detach().numpy()-codes_inv[1].squeeze()[m].cpu().detach().numpy())    
        
    i = i+1

#print(distances)
avrage_distances = np.mean(distances, axis=2)
avrage_distances = np.mean(avrage_distances, axis=1)


# additional illustration for calculating distances between images

distance = list(avrage_distances)
#plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "12"
fig, ax = plt.subplots(1, 1, figsize = (5, 3), dpi=320)
ax.bar(radius, distance, width = 0.3, color = "steelblue")
ax.set_xlim((2.5, 6.5))
ax.set_ylim((0, 4))
ax.set_xlabel('Hyperbolic Radius')
ax.set_ylabel('L2 Distance')
ax.set_xticks(radius)
plt.subplots_adjust(bottom=0.2)
plt.savefig('/proj/outputs/distances1.png', dpi=320)
plt.show()