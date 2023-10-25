import argparse
import numpy as np
import learnreg as lr  


def main(args):
    denoised_image, W, img, noise_img = lr.main_image(
        filename=args.filename,
        patch_size=args.patch_size,
        forward_model_type=args.forward_model_type,
        noise_sigma=args.noise_sigma,
        transform_type=args.transform_type,
        transform_scale=args.transform_scale,
        SEED=args.SEED,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        sign_threshold=args.sign_threshold,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Denoising Script')
    
    parser.add_argument('--filename', type=str,default="barbara_gray.bmp" ,help='Image filename')
    parser.add_argument('--patch_size', type=int, default=8,  help='Patch size')
    parser.add_argument('--forward_model_type', type=str, default='identity',  help='Forward model type')
    parser.add_argument('--noise_sigma', type=float,default=0.01, help='Noise sigma')
    parser.add_argument('--transform_type', type=str, default='identity', help='Transform type')
    parser.add_argument('--transform_scale', type=float,default=1e-2, help='Transform scale')
    parser.add_argument('--SEED', type=int, default=0, help='Random seed')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_steps', type=int, default=int(1e5), help='Number of steps')
    parser.add_argument('--sign_threshold', type=float, default=1e-4, help='Sign threshold')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size')
    
    args = parser.parse_args()
    main(args)
