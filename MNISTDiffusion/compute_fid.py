import sys
import torch
from model import MNISTDiffusion
from utils import ExponentialMovingAverage
import argparse
from torchmetrics.image.fid import FrechetInceptionDistance
from rotated_mnist_data import RotatedMNISTDataModule

def compute_fid(model_ema, device, args):
    fid = FrechetInceptionDistance(feature=64).to(device)

    class DataHyperparameters:
        def __init__(self):
            self.batch_size = args.batch_size
            self.num_workers = 4
            self.data_path = 'rotated_mnist'

    data_hyperparams = DataHyperparameters()

    data_module = RotatedMNISTDataModule(data_hyperparams)
    dataloader = data_module.val_dataloader()
    
    print("Computing FID for data")
    for j,(image,target) in enumerate(dataloader):
        image=image.to(device)
        imgs_dist1 = image.repeat(1, 3, 1, 1).to(torch.uint8)
        fid.update(imgs_dist1, real=True)

    print("Generating samples and computing FID over", len(dataloader), "batches")

    for j in range(len(dataloader)):
        print("Batch:", j)
        samples = model_ema.module.sampling(args.batch_size,clipped_reverse_diffusion=not args.no_clip,device=device)
        imgs_dist2 = samples.repeat(1, 3, 1, 1).to(torch.uint8)
        fid.update(imgs_dist2, real=False)

    # Calculate the FID score
    return fid.compute()

def parse_args():
    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")
    parser.add_argument('--ckpt',type = str,help = 'define checkpoint path',default='/home/mila/k/kusha.sareen/scratch/results/mnist_rot16_dim64_l1e-2/steps_00077104.pt')
    parser.add_argument('--batch_size',type = int ,default=128)    
    parser.add_argument('--model_ema_steps',type = int,help = 'ema model evaluation interval',default=10)
    parser.add_argument('--model_ema_decay',type = float,help = 'ema model decay',default=0.995)
    parser.add_argument('--model_base_dim',type = int,help = 'base dim of Unet',default=64)
    parser.add_argument('--timesteps',type = int,help = 'sampling steps of DDPM',default=1000)
    parser.add_argument('--no_clip',action='store_true',help = 'set to normal sampling method without clip x_0 which could yield unstable samples')
    parser.add_argument('--cpu',action='store_true',help = 'cpu training')
    parser.add_argument('--epochs',type = int,default=1000)

    args = parser.parse_args()

    return args

def load_models(args, device):
    ckpt=torch.load(args.ckpt)

    model=MNISTDiffusion(timesteps=args.timesteps,
            image_size=28,
            in_channels=1,
            base_dim=args.model_base_dim,
            dim_mults=[2,4]).to(device)

    adjust = 1* args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

    model.load_state_dict(ckpt["model"])
    model_ema.load_state_dict(ckpt["model_ema"])

    return model_ema

def main():
    args = parse_args()
    device="cpu" if args.cpu else "cuda"
    model_ema= load_models(args, device)
    fid = compute_fid(model_ema, device, args)

    print("FID Score:", fid.item(), "on model at", args.ckpt)

if __name__ == '__main__':
    main()