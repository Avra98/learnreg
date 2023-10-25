"""
Learning transforms on simple 1D signals: DCT and TV sparse signals. 
Make sure to run atleast 1e6 iterations to generate meaningful results.
"""

import argparse
import learnreg as lr

def main(args):
    W = lr.main(
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        sign_threshold=args.sign_threshold,
        signal_type=args.signal_type,
        n=args.n,
        noise_sigma=args.noise_sigma,
        forward_model_type=args.forward_model_type,
        num_training=args.num_training,
        transform_type=args.transform_type,
        transform_scale=args.transform_scale,
        num_testing=args.num_testing,
        seed=args.seed,
        batch_size=args.batch_size
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LearnReg Script')
    
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_steps', type=int, default=int(1e5), help='Number of steps')
    parser.add_argument('--sign_threshold', type=float, default=1e-4, help='Sign threshold')
    parser.add_argument('--signal_type', type=str, default='piecewise_constant', help='Signal type')
    parser.add_argument('--n', type=int, default=64, help='Dimensionality')
    parser.add_argument('--noise_sigma', type=float, default=1e-2, help='Noise sigma')
    parser.add_argument('--forward_model_type', type=str, default='identity', help='Forward model type')
    parser.add_argument('--num_training', type=int, default=10000, help='Number of training samples')
    parser.add_argument('--transform_type', type=str, default='identity', help='Transform type')
    parser.add_argument('--transform_scale', type=float, default=1e-2, help='Transform scale')
    parser.add_argument('--num_testing', type=int, default=5, help='Number of testing samples')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size')
    
    args = parser.parse_args()
    main(args)


