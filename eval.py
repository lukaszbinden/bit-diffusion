from bit_diffusion import Unet, Trainer, BitDiffusion
import os


def expanduservars(path: str) -> str:
    return os.path.expanduser(os.path.expandvars(path))


def main(load_milestone, ged_n):
    model = Unet(
        dim=32,
        channels=3,  # 1: x, 1: image, 1: self-cond
        dim_mults=(1, 2, 4, 8),
    ).cuda()

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("UNet model size: ", model_size)

    bit_diffusion = BitDiffusion(
        model,
        image_size=128,
        timesteps=100,
        time_difference=0.1,
        # they found in the paper that at lower number of timesteps, a time difference during sampling of greater than 0 helps FID. as timesteps increases, this time difference can be set to 0 as it does not help
        use_ddim=True  # use ddim
    ).cuda()

    model_size = sum(p.numel() for p in bit_diffusion.parameters() if p.requires_grad)
    print("BitDiffusion model size: ", model_size)

    # data_folder = "/storage/homefs/lz20w714/git/mose-auseg/data/lidc_npy"
    data_folder = "/home/lukas/git/mose-auseg/data/lidc_npy"

    this_run = expanduservars("train_${NOW}")
    out_dir = "./results/" + this_run
    print(f"Log dir: {out_dir}")

    trainer = Trainer(
        bit_diffusion,
        data_folder,  # path to your folder of images
        results_folder=out_dir,  # where to save results
        num_samples=ged_n,  # number of samples
        train_batch_size=2,  # training batch size
        gradient_accumulate_every=4,  # gradient accumulation
        train_lr=1e-4,  # learning rate
        save_and_sample_every=2,  # how often to save and sample
        train_num_steps=700000,  # total training steps
        ema_decay=0.995,  # exponential moving average decay
        eval_mode="val"  # train, val, test
    )

    trainer.load(load_milestone, is_abs_path=True)

    trainer.eval()


if __name__ == '__main__':
    os.environ.pop("SLURM_JOBID", None)

    milestone = "10"
    milestone = "/home/lukas/git/bit-diffusion/checkpoints/ddim_9M_model-30.pt"
    ged_n = 16

    main(milestone, ged_n)
