# main.py

import argparse

from runners import (
    run_logreg_tfidf,
    run_cosine_sbert,
    run_nn_frozen,
    run_nn_unfrozen,
    run_prototype_model,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "logreg_tfidf",
            "cosine_sbert",
            "nn_frozen",
            "nn_unfrozen",
            "prototype",
        ],
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.model == "logreg_tfidf":
        run_logreg_tfidf()
    elif args.model == "cosine_sbert":
        run_cosine_sbert()
    elif args.model == "nn_frozen":
        run_nn_frozen()
    elif args.model == "nn_unfrozen":
        run_nn_unfrozen()
    elif args.model == "prototype":
        run_prototype_model()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
