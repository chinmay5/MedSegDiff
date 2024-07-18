
import sys
import argparse

from guided_diffusion.acdc_dataset import ACDC_ShortAxisDataset

sys.path.append("../")
sys.path.append("./")
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.bratsloader import BRATSDataset, BRATSDataset3D
from guided_diffusion.isicloader import ISICDataset
from guided_diffusion.custom_dataset_loader import CustomDataset
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import torch as th
from guided_diffusion.train_util import TrainLoop
# from visdom import Visdom
# viz = Visdom(port=8850)
import torchvision.transforms as transforms

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist(args)
    logger.configure(dir = args.out_dir)

    logger.log("creating data loader...")

    if args.data_name == 'ISIC':
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
        transform_train = transforms.Compose(tran_list)

        ds = ISICDataset(args, args.data_dir, transform_train)
        args.in_ch = 4
    elif args.data_name == 'BRATS':
        tran_list = [transforms.Resize((args.image_size,args.image_size)),]
        transform_train = transforms.Compose(tran_list)

        ds = BRATSDataset3D(args.data_dir, transform_train, test_flag=False)
        args.in_ch = 5
    elif args.data_name == 'ACDC':
        # tran_list = [transforms.Resize((args.image_size,args.image_size)),]
        # transform_train = transforms.Compose(tran_list)
        # We are giving a lot of defaults programmatically.
        # Perhaps using the command line arguments will be better compared to being so lazy here :(
        ds = ACDC_ShortAxisDataset(args.data_dir, img_size=args.image_size)
        args.in_ch = 3
        args.out_channels = 4
        args.image_size = 128
    else:
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
        transform_train = transforms.Compose(tran_list)
        print("Your current directory : ",args.data_dir)
        ds = CustomDataset(args, args.data_dir, transform_train)
        args.in_ch = 4
        
    datal= th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True)
    data = iter(datal)

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    if args.multi_gpu:
        model = th.nn.DataParallel(model,device_ids=[int(id) for id in args.multi_gpu.split(',')])
        model.to(device = th.device('cuda', int(args.gpu_dev)))
    else:
        model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=args.diffusion_steps)


    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None,
        data=data,
        dataloader=datal,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    # defaults = dict(
    #     data_name = 'BRATS',
    #     data_dir="../dataset/brats2020/training",
    #     schedule_sampler="uniform",
    #     lr=1e-4,
    #     weight_decay=0.0,
    #     lr_anneal_steps=0,
    #     batch_size=1,
    #     microbatch=-1,  # -1 disables microbatches
    #     ema_rate="0.9999",  # comma-separated list of EMA values
    #     log_interval=100,
    #     save_interval=5000,
    #     resume_checkpoint=None, #"/results/pretrainedmodel.pt"
    #     use_fp16=False,
    #     fp16_scale_growth=1e-3,
    #     gpu_dev = "0",
    #     multi_gpu = None, #"0,1,2"
    #     out_dir='./results/'
    # )
    defaults = dict(
        data_name='ACDC',
        data_dir="/mnt/elephant/chinmay/ACDC/database/training/",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=128,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=5000,
        resume_checkpoint=None,  # "/results/pretrainedmodel.pt"
        use_fp16=False,
        fp16_scale_growth=1e-3,
        gpu_dev="1",
        multi_gpu=None,  # "0,1,2"
        out_dir='./results/',
        image_size=128,
        in_ch=3,
        out_channels=4,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
    # tran_list = [transforms.Resize((224, 224)), ]
    # transform_train = transforms.Compose(tran_list)
    #
    # ds = ACDC_ShortAxisDataset('/mnt/elephant/chinmay/ACDC/database/training/', img_size=128)
    # in_ch = 1
    # datal = th.utils.data.DataLoader(
    #     ds,
    #     batch_size=8,
    #     shuffle=True)
    # data = next(iter(datal))
    # # print(data)
    # # print(data[0].shape)
    # # print(data[1].shape)
    # from guided_diffusion.script_util import create_model
    # model = create_model(
    #     128,
    #     128,
    #     1,
    #     in_ch=1,  # num_channels + 1
    #     num_heads=1,
    #     learn_sigma=False,
    #     out_channels=3
    # )
    # import torch
    # timesteps = torch.tensor([5, 5, 5, 5, 5, 5, 5, 5], dtype=torch.float)
    # out = model(data[0], timesteps)
    # print(out)