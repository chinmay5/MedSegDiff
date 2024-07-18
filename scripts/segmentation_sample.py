
import argparse
import os
import shutil
from ssl import OP_NO_TLSv1
import nibabel as nib
# from visdom import Visdom
# viz = Visdom(port=8850)
import sys
import random

from guided_diffusion.acdc_dataset import ACDC_ShortAxisDataset

sys.path.append(".")
import numpy as np
import time
import torch as th
from PIL import Image
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset, BRATSDataset3D
from guided_diffusion.isicloader import ISICDataset
import torchvision.utils as vutils
from guided_diffusion.utils import staple
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import torchvision.transforms as transforms
# from torchsummary import summary
seed =10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min )/ (_max - _min)
    return normalized_img


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args)
    logger.configure(dir = args.out_dir)

    if args.data_name == 'ISIC':
        tran_list = [transforms.Resize((args.image_size ,args.image_size)), transforms.ToTensor() ,]
        transform_test = transforms.Compose(tran_list)

        ds = ISICDataset(args, args.data_dir, transform_test, mode = 'Test')
        args.in_ch = 4
    elif args.data_name == 'BRATS':
        tran_list = [transforms.Resize((args.image_size ,args.image_size)) ,]
        transform_test = transforms.Compose(tran_list)

        ds = BRATSDataset3D(args.data_dir ,transform_test)
        args.in_ch = 5
    elif args.data_name == 'ACDC':
        # tran_list = [transforms.Resize((args.image_size,args.image_size)),]
        # transform_train = transforms.Compose(tran_list)
        ds = ACDC_ShortAxisDataset(args.data_dir, img_size=args.image_size)
        args.in_ch = 3
        args.out_channels = 4
        args.image_size = 128
        print("Using a single ensemble")
        args.num_ensemble = 1
        # Creating the specific log location for the images for ACDC dataset
        save_dir = f"{args.out_dir}/images"
        seg_dir = f"{args.out_dir}/pred_segs"
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        if os.path.exists(seg_dir):
            shutil.rmtree(seg_dir)
        os.makedirs(save_dir)
        os.makedirs(seg_dir)

    datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True)
    data = iter(datal)

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    all_images = []


    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:] # remove `module.`
        if 'module.' in k:
            new_state_dict[k[7:]] = v
            # load params
        else:
            new_state_dict = state_dict

    model.load_state_dict(new_state_dict)

    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    for _ in range(len(data)):
        b, m, path = next(data)  #should return an image from the dataloader "data"
        gt_mask = m
        # c = th.randn_like(b[:, :1, ...])
        c = th.randn_like(m.to(th.float))
        img = th.cat((b, c), dim=1)     #add a noise channel$
        if args.data_name == 'ISIC':
            slice_ID =path[0].split("_")[-1].split('.')[0]
        elif args.data_name == 'BRATS':
            # slice_ID=path[0].split("_")[2] + "_" + path[0].split("_")[4]
            slice_ID =path[0].split("_")[-3] + "_" + path[0].split("slice")[-1].split('.nii')[0]
        elif args.data_name == 'ACDC':
            slice_ID = path[0].split("_")[1] + "_" + path[0].split("_")[2]
        logger.log("sampling...")

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)
        enslist = []

        for i in range(args.num_ensemble):  #this is for the generation of an ensemble of 5 masks.
            model_kwargs = {}
            start.record()
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            sample, x_noisy, org, cal, cal_out = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size), img,
                step = args.diffusion_steps,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            end.record()
            th.cuda.synchronize()
            print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample

            co = th.tensor(cal_out)
            if args.version == 'new':
                enslist.append(sample[: ,-1 ,: ,:])
            else:
                enslist.append(co)

            if args.debug:
                # print('sample size is',sample.size())
                # print('org size is',org.size())
                # print('cal size is',cal.size())
                if args.data_name == 'ISIC':
                    # s = th.tensor(sample)[:,-1,:,:].unsqueeze(1).repeat(1, 3, 1, 1)
                    o = th.tensor(org)[: ,:-1 ,: ,:]
                    c = th.tensor(cal).repeat(1, 3, 1, 1)
                    # co = co.repeat(1, 3, 1, 1)

                    s = sample[: ,-1 ,: ,:]
                    b ,h ,w = s.size()
                    ss = s.clone()
                    ss = ss.view(s.size(0), -1)
                    ss -= ss.min(1, keepdim=True)[0]
                    ss /= ss.max(1, keepdim=True)[0]
                    ss = ss.view(b, h, w)
                    ss = ss.unsqueeze(1).repeat(1, 3, 1, 1)

                    tup = (ss ,o ,c)
                elif args.data_name == 'BRATS':
                    s = th.tensor(sample)[: ,-1 ,: ,:].unsqueeze(1)
                    m = th.tensor(m.to(device = 'cuda:0'))[: ,0 ,: ,:].unsqueeze(1)
                    o1 = th.tensor(org)[: ,0 ,: ,:].unsqueeze(1)
                    o2 = th.tensor(org)[: ,1 ,: ,:].unsqueeze(1)
                    o3 = th.tensor(org)[: ,2 ,: ,:].unsqueeze(1)
                    o4 = th.tensor(org)[: ,3 ,: ,:].unsqueeze(1)
                    c = th.tensor(cal)

                    tup = (o1 /o1.max() ,o2 /o2.max() ,o3 /o3.max() ,o4 /o4.max() ,m ,s ,c ,co)

                elif args.data_name == 'ACDC':
                    # We define a utility function that plots image, GT and segmentation together
                    s = th.tensor(sample)
                    # m = th.tensor(m.to(device='cuda:0'))
                    o1 = th.tensor(org[:, :args.in_ch, :, :])
                    c = th.tensor(cal)
                    # tup = (o1 / o1.max(), m, s, c, co)
                    # Plot the resulting images
                    visualize_img_seg(o1, m, s, save_dir, slice_ID, i)
                    # converting the segmentation into an appropriate image
                    height, width = s.shape[-2], s.shape[-1]
                    segmentation_rgb = th.zeros((3, height, width))
                    # NOTE: The code may break if we have batch size > 1
                    s = th.argmax(s[0], dim=0).cpu()
                    # Overlay the segmentation labels
                    for c in range(3):  # RGB channels
                        segmentation_rgb[c, :, :] = ((s - 1) == c) * 255
                    # Convert the dtype into intergers to avoid warnings during the plotting operation
                    image = segmentation_rgb.clamp_(0, 255).permute(1, 2, 0).to("cpu", th.uint8).numpy()
                    im = Image.fromarray(image)
                    im.save(os.path.join(seg_dir, str(slice_ID) + '_output_ens' + ".jpg"))

        #         compose = th.cat(tup, 0)
        #         vutils.save_image(compose, fp=os.path.join(args.out_dir, str(slice_ID) + '_output' + str(i) + ".jpg"),
        #                           nrow=1, padding=10)
        # ensres = staple(th.stack(enslist, dim=0)).squeeze(0)
        # vutils.save_image(ensres, fp=os.path.join(args.out_dir, str(slice_ID) + '_output_ens' + ".jpg"), nrow=1,
        #                   padding=10)


def create_argparser():
    # defaults = dict(
    #     data_name = 'BRATS',
    #     data_dir="../dataset/brats2020/testing",
    #     clip_denoised=True,
    #     num_samples=1,
    #     batch_size=1,
    #     use_ddim=False,
    #     model_path="",         #path to pretrain model
    #     num_ensemble=5,      #number of samples in the ensemble
    #     gpu_dev = "0",
    #     out_dir='./results/',
    #     multi_gpu = None, #"0,1,2"
    #     debug = False
    # )
    defaults = dict(
        data_name='ACDC',
        data_dir="/mnt/elephant/chinmay/ACDC/database/testing/",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path='/home/chinmayp/workspace/my_explorations/MedSegDiff/results/emasavedmodel_0.9999_065000.pt',
        num_ensemble=5,
        gpu_dev="0",
        in_ch=3,
        out_channels=4,
        out_dir='./results/',
        debug=True,
        multi_gpu=None,  # "0,1,2"
        image_size=128,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def visualize_img_seg(image, gt_seg, pred_seg, save_dir, slice_ID, i):
    bs = image.shape[0]
    for idx in range(bs):
        img, gt, pred = image[idx, ...], gt_seg[idx, ...], pred_seg[idx, ...]
        img = img.mean(dim=0).cpu().numpy()
        # We convert the mask into actual integer values
        gt_mask = th.argmax(gt, dim=0).cpu().numpy()
        pred_mask = th.argmax(pred, dim=0).cpu().numpy()
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(12, 6))
        # Plot the original image
        im = ax[0].imshow(img, cmap='gray', vmin=img.min(), vmax=img.max())
        ax[0].set_title('Original Image')
        ax[0].axis('off')  # Hide the axis
        fig.colorbar(im, ax=ax[0])  # Add colorbar to indicate the value range

        # Create an RGB image for the segmentation
        height, width = img.shape[-2], img.shape[-1]
        segmentation_gt_rgb = np.zeros((height, width, 3))
        segmentation_pred_rgb = np.zeros((height, width, 3))
        # Overlay the segmentation labels
        for c in range(3):  # RGB channels
            segmentation_gt_rgb[:, :, c] = ((gt_mask - 1) == c) * 255
            segmentation_pred_rgb[:, :, c] = ((pred_mask - 1) == c) * 255
        # Convert the dtype into intergers to avoid warnings during the plotting operation
        segmentation_gt_rgb = segmentation_gt_rgb.astype(int)
        segmentation_pred_rgb = segmentation_pred_rgb.astype(int)
        # Plot the segmentation mask
        ax[1].imshow(segmentation_gt_rgb)
        ax[1].set_title('gt')
        ax[1].axis('off')  # Hide the axis
        ax[2].imshow(segmentation_gt_rgb)
        ax[2].set_title('pred')
        ax[2].axis('off')  # Hide the axis
        # Display the plots
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(save_dir, str(slice_ID) + '_output' + str(idx) + "_" + str(i) + ".jpg"))


if __name__ == "__main__":
    main()
