from manifm.eval_utils import load_model
import torch
import argparse
import os

#checkpoint = "/home/fvaleau/riemannian-fm/outputs/runs/images/fm/2025.12.05/183800/checkpoints/last.ckpt"
# checkpoint = "/home/fvaleau/riemannian-fm/outputs/runs/images/fm/2026.01.12/022417/checkpoints/epoch-394_step-0_loss-0.0000.ckpt"

# cfg, model = load_model(checkpoint)
# print(type(model.manifold))

# model = model.cuda()
# model.eval()
# with torch.no_grad():
#     samples = model.sample(1020, device="cuda")
# print(samples.shape)   # torch.Size([10, 512])
# print(samples)
# torch.save(samples, "fm_samples3")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint (.ckpt)"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1020,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="fm_samples",
        help="Name for the output samples file"
    )

    args = parser.parse_args()
    checkpoint = args.checkpoint
    output_name = f"/scratch-shared/fvaleau/FID/fm_samples_{args.name}"

    print(f"Loading checkpoint: {checkpoint}")

    cfg, model = load_model(checkpoint)

    print(type(model.manifold))

    model = model.to(args.device)
    model.eval()
    with torch.no_grad():
        samples = model.sample(args.n_samples, device=args.device)
    torch.save(samples, output_name)
    print("Saved:", output_name)


if __name__ == "__main__":
    main()
