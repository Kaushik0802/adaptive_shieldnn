# main.py

import argparse
from train.train_margin_rl import train_adaptive_margin
from eval.eval_policy import evaluate_policy

def main():
    parser = argparse.ArgumentParser(description="Adaptive ShieldNN")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--model", type=str, default="checkpoints/delta_final.pt")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    if args.mode == "train":
        print("Starting training...")
        train_adaptive_margin(epochs=args.epochs)
    elif args.mode == "eval":
        print(f"Evaluating saved model: {args.model}")
        evaluate_policy(model_path=args.model, render=args.render)

if __name__ == "__main__":
    main()
