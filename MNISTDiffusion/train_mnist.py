import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms 
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from model import MNISTDiffusion
from utils import ExponentialMovingAverage
import os
import math
import argparse
import wandb
import sys
import copy
from rotated_mnist_data import RotatedMNISTDataModule
from equiadapt.images.canonicalization.discrete_group import GroupEquivariantImageCanonicalization
from equiadapt.images.canonicalization_networks import ESCNNEquivariantNetwork
from eqv_denoiser import ESCNNDenoiser, ESCNNUnet

def create_mnist_dataloaders(batch_size,image_size=28,num_workers=4):
    
    preprocess=transforms.Compose([transforms.Resize(image_size),\
                                    transforms.ToTensor(),\
                                    transforms.Normalize([0.5],[0.5])]) #[0,1] to [-1,1]

    train_dataset=MNIST(root="./mnist_data",\
                        train=True,\
                        download=True,\
                        transform=preprocess
                        )
    test_dataset=MNIST(root="./mnist_data",\
                        train=False,\
                        download=True,\
                        transform=preprocess
                        )

    return DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers),\
            DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)


def parse_args():
    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")
    parser.add_argument('--lr',type = float ,default=0.001)
    parser.add_argument('--lr_ratio',type = float ,default=1e-2) # set after lr_ratio le-2 looked good
    parser.add_argument('--batch_size',type = int ,default=128)    
    parser.add_argument('--epochs',type = int,default=1000)
    parser.add_argument('--ckpt',type = str,help = 'define checkpoint path',default='')
    parser.add_argument('--n_samples',type = int,help = 'define sampling amounts after every epoch trained',default=36)
    parser.add_argument('--model_base_dim',type = int,help = 'base dim of Unet',default=64)
    parser.add_argument('--timesteps',type = int,help = 'sampling steps of DDPM',default=1000)
    parser.add_argument('--model_ema_steps',type = int,help = 'ema model evaluation interval',default=10)
    parser.add_argument('--model_ema_decay',type = float,help = 'ema model decay',default=0.995)
    parser.add_argument('--log_freq',type = int,help = 'training log message printing frequence',default=10)
    parser.add_argument('--no_clip',action='store_true',help = 'set to normal sampling method without clip x_0 which could yield unstable samples')
    parser.add_argument('--cpu',action='store_true',help = 'cpu training')
    parser.add_argument('--use_canon', action='store_true',
                        help='turns on canonicalization')
    parser.add_argument('--canon_n_layers', type=int, default=5)
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
    parser.add_argument('--exp_name', type=str, default='mnist_diffusion')
    parser.add_argument('--use_rot_mnist', action='store_true', help='Uses the rotated MNIST dataset instead')
    parser.add_argument('--test_epoch', type=int, default=25)
    parser.add_argument('--num_rot', type=int, default=16)
    parser.add_argument('--freeze', action='store_true', help='Canon not optimized')
    parser.add_argument('--use_eqv_denoiser', action='store_true')
    parser.add_argument('--use_refl', action='store_true')
    parser.add_argument('--eqv_denoiser', action='store_true')
    parser.add_argument('--double_canon', action='store_true')

    args = parser.parse_args()

    return args

def update_ema(ema_model, model, decay): # custom ema for eqv denoiser
    for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
        if ema_param.shape == model_param.shape:
            ema_param.data.mul_(decay).add_(model_param.data, alpha=(1 - decay))
        else:
            print(f"Skipping EMA update due to shape mismatch: {ema_param.shape} vs {model_param.shape}")

def main(args):
    os.makedirs("/home/mila/k/kusha.sareen/scratch/results/%s" % args.exp_name,exist_ok=True)
    os.makedirs("/home/mila/k/kusha.sareen/scratch/images/%s" % args.exp_name,exist_ok=True)

    device="cpu" if args.cpu else "cuda"

    if not args.use_rot_mnist:
        train_dataloader,test_dataloader=create_mnist_dataloaders(batch_size=args.batch_size,image_size=28)
    else:
        class DataHyperparameters:
            def __init__(self): 
                self.batch_size = args.batch_size
                self.num_workers = 4
                self.data_path = 'rotated_mnist'

        data_hyperparams = DataHyperparameters()
        data_module = RotatedMNISTDataModule(data_hyperparams)
        train_dataloader = data_module.train_dataloader()
        test_dataloader = data_module.val_dataloader()

    model=MNISTDiffusion(timesteps=args.timesteps,
                image_size=28,
                in_channels=1,
                base_dim=args.model_base_dim,
                dim_mults=[2,4]).to(device)
    
    if args.eqv_denoiser:
        # print(sum(p.numel() for p in model.model.parameters() if p.requires_grad))
        model.model = ESCNNUnet(
                in_shape = (1,28,28),
                out_channels=args.model_base_dim, # can increase this
                num_layers = 8,
                group_type='rotation', 
                num_rotations=args.num_rot, 
                timesteps=args.timesteps
            ).to(device)
        
        # print(sum(p.numel() for p in model.model.parameters() if p.requires_grad))


    if args.use_canon:
        class CanonicalizationHyperparameters:
            def __init__(self): 
                self.device = device
                self.input_crop_ratio = 0.9 # The ratio of the input image to crop
                self.beta = 1.0 # Beta parameter for the canonization network
                self.resize_shape = 64 # The shape of the image after resizing

        canon_hyperparams = CanonicalizationHyperparameters() 

        group_type = 'rotation' if not args.use_refl else 'roto-reflection'

        canonicalization_network = ESCNNEquivariantNetwork(
            in_shape = (1, 28,28),
            out_channels=16, 
            kernel_size= args.kernel_size, 
            num_layers = args.canon_n_layers,
            group_type=group_type, 
            num_rotations=args.num_rot, 
        ).to(device)

    #torchvision ema setting
    adjust = 1* args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)
    if args.eqv_denoiser:
        model_ema = copy.deepcopy(model)

    if args.no_wandb:
        mode = 'disabled'
    else:
        mode = 'online'
    kwargs = {'name': args.exp_name, 'project': 'RotMNIST Diffusion', 'config': args, 'mode': mode}
    wandb.init(**kwargs)
    wandb.save('*.txt')

    params = [{"params": model.parameters(), "lr": args.lr}]
    if (not args.freeze) and args.use_canon:
        params += [{"params": canonicalization_network.parameters(), "lr": args.lr*args.lr_ratio}]
        print("Using an optimized canonicalizer")

    optimizer=AdamW(params,lr=args.lr)
    scheduler=OneCycleLR(optimizer,args.lr,total_steps=args.epochs*len(train_dataloader),pct_start=0.25,anneal_strategy='cos')
    loss_fn=nn.MSELoss(reduction='mean')

    #load checkpoint
    if args.ckpt:
        ckpt=torch.load(args.ckpt)
        model_ema.load_state_dict(ckpt["model_ema"])
        model.load_state_dict(ckpt["model"])
        canonicalization_network.load_state_dict(ckpt["canonicalization_network"])

    if args.use_canon:
        canonicalizer = GroupEquivariantImageCanonicalization(
            canonicalization_network=canonicalization_network,
            canonicalization_hyperparams=canon_hyperparams,
            in_shape=(1, 28, 28))
        canonicalizer.train()
        
        
    global_steps=0
    for i in range(args.epochs):
        model.train()
        for j,(image,target) in enumerate(train_dataloader): # TRAIN LOOP
            noise=torch.randn_like(image).to(device)
            image=image.to(device)
            if args.use_canon:
                image = canonicalizer(image)
            pred=model(image,noise)
            loss=loss_fn(pred,noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            if global_steps%args.model_ema_steps==0:
                if args.eqv_denoiser:
                    update_ema(model_ema, model, decay=1.0 - alpha)
                else:
                    model_ema.update_parameters(model)
            global_steps+=1
            if j%args.log_freq==0:
                print("Epoch[{}/{}],Step[{}/{}],loss:{:.5f},lr:{:.5f}".format(i+1,args.epochs,j,len(train_dataloader),
                                                                    loss.detach().cpu().item(),scheduler.get_last_lr()[0]))
            wandb.log({"Train loss ": loss}, commit=True)

        if i % args.test_epoch == 0:
            model_ema.eval()
            with torch.no_grad(): # TEST LOOP
                test_loss = 0
                n_batches = 0
                for j,(image,target) in enumerate(test_dataloader):
                    noise=torch.randn_like(image).to(device)
                    image=image.to(device)
                    if args.use_canon:
                        if j ==0 :save_image(image, ("/home/mila/k/kusha.sareen/scratch/images/%s/in_" + str(i) +".png") % args.exp_name)
                        image = canonicalizer(image)
                        if j ==0: save_image(image, ("/home/mila/k/kusha.sareen/scratch/images/%s/out_" + str(i) +".png") % args.exp_name)

                    pred=model_ema(image,noise)
                    loss=loss_fn(pred,noise)
                    test_loss += loss
                    n_batches += 1
                    print("Test Epoch[{}],Step[{}/{}],loss:{:.5f}".format(i+1,j,len(test_dataloader),
                                                                    loss.detach().cpu().item()))
                    
                wandb.log({"Test loss ": test_loss/len(test_dataloader)}, commit=True)


            ckpt={"model":model.state_dict(),
                    "model_ema":model_ema.state_dict()}
            
            if args.use_canon:
                ckpt['canonicalization_network']= canonicalization_network.state_dict()

            torch.save(ckpt,("/home/mila/k/kusha.sareen/scratch/results/%s/steps_{:0>8}.pt" % args.exp_name).format(global_steps))

            model_ema.eval()
            if args.eqv_denoiser:
                samples=model_ema.sampling(args.n_samples,clipped_reverse_diffusion=not args.no_clip,device=device)
            else:
                samples=model_ema.module.sampling(args.n_samples,clipped_reverse_diffusion=not args.no_clip,device=device)

            save_image(samples,("/home/mila/k/kusha.sareen/scratch/results/%s/steps_{:0>8}.png" % args.exp_name).format(global_steps) ,nrow=int(math.sqrt(args.n_samples)))
            wandb.log({"example": wandb.Image(("/home/mila/k/kusha.sareen/scratch/results/%s/steps_{:0>8}.png" % args.exp_name).format(global_steps))})
        
if __name__=="__main__":
    args=parse_args()
    main(args)
