"""
predict.py — Shakespearean LSTM Prediction Utilities
=====================================================
Standalone prediction module for the trained LSTM next-word predictor.

Usage (CLI):
    python predict.py --seed "to be or not to" --words 20
    python predict.py --seed "what dreams may come" --topk 5 --temperature 0.8

Usage (as a module):
    from predict import load_model_assets, predict_next_words, generate_text

    model, tokenizer, config = load_model_assets()
    print(predict_next_words(model, tokenizer, config, "to be or not", top_k=3))
    print(generate_text(model, tokenizer, config, "to be or not", num_words=15))
"""

import os
import json
import pickle
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ---------------------------------------------------------------------------
# Asset Loading
# ---------------------------------------------------------------------------

def load_model_assets(
    model_path: str = "model.h5",
    tokenizer_path: str = "tokenizer.pickle",
    config_path: str = "config.json",
):
    """
    Load the trained model, tokenizer, and config from disk.

    Returns:
        tuple: (model, tokenizer, config) or raises FileNotFoundError.
    """
    for path in [model_path, tokenizer_path, config_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file not found: '{path}'\n"
                "Run 'python train.py' first to generate model assets."
            )

    print(f"Loading model from  : {model_path}")
    model = tf.keras.models.load_model(model_path)

    print(f"Loading tokenizer   : {tokenizer_path}")
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    print(f"Loading config      : {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)

    print(f"Assets loaded | Vocab: {config['total_words']:,} words | "
          f"Context window: {config['max_sequence_len'] - 1} tokens\n")
    return model, tokenizer, config


# ---------------------------------------------------------------------------
# Core Helpers
# ---------------------------------------------------------------------------

def _apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """
    Apply temperature scaling to a probability distribution.

    temperature = 0  -> deterministic (argmax)
    temperature = 1  -> original distribution
    temperature > 1  -> more random / creative
    temperature < 1  -> sharper / more confident
    """
    if temperature <= 0:
        one_hot = np.zeros_like(probs)
        one_hot[np.argmax(probs)] = 1.0
        return one_hot

    log_probs = np.log(probs + 1e-10) / temperature
    exp_probs = np.exp(log_probs)
    return exp_probs / np.sum(exp_probs)


def _index_to_word(tokenizer, index: int) -> str:
    """Reverse-lookup: token index -> word string."""
    return tokenizer.index_word.get(index, "")


def _text_to_token_list(tokenizer, text: str, max_sequence_len: int) -> list:
    """Tokenize input text and trim to context window."""
    tokens = tokenizer.texts_to_sequences([text.strip().lower()])[0]
    if len(tokens) >= max_sequence_len:
        tokens = tokens[-(max_sequence_len - 1):]
    return tokens


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_next_words(
    model,
    tokenizer,
    config: dict,
    seed_text: str,
    top_k: int = 5,
    temperature: float = 1.0,
):
    """
    Predict the top-k most likely next words for a given seed text.

    Args:
        model:        Loaded Keras model.
        tokenizer:    Fitted Keras Tokenizer.
        config:       Dict with 'max_sequence_len'.
        seed_text:    Input text string.
        top_k:        Number of top predictions to return.
        temperature:  Sampling temperature (0 = greedy, >1 = creative).

    Returns:
        List of (word, probability) tuples sorted by probability descending.
    """
    max_seq_len = config["max_sequence_len"]
    tokens = _text_to_token_list(tokenizer, seed_text, max_seq_len)
    padded = pad_sequences([tokens], maxlen=max_seq_len - 1, padding="pre")

    raw_probs = model.predict(padded, verbose=0)[0]
    adj_probs = _apply_temperature(raw_probs, temperature)

    top_indices = np.argsort(adj_probs)[::-1][:top_k]

    results = []
    for idx in top_indices:
        word = _index_to_word(tokenizer, idx)
        if word:
            results.append((word, float(raw_probs[idx])))

    return results


def generate_text(
    model,
    tokenizer,
    config: dict,
    seed_text: str,
    num_words: int = 20,
    temperature: float = 0.7,
) -> str:
    """
    Auto-generate a sequence of words continuing from the seed text.

    Args:
        model:        Loaded Keras model.
        tokenizer:    Fitted Keras Tokenizer.
        config:       Dict with 'max_sequence_len'.
        seed_text:    Starting text.
        num_words:    Number of words to generate.
        temperature:  Sampling temperature.

    Returns:
        Full generated string (seed + generated words).
    """
    max_seq_len = config["max_sequence_len"]
    tokens = _text_to_token_list(tokenizer, seed_text, max_seq_len)
    output = seed_text.strip()

    for _ in range(num_words):
        padded = pad_sequences([tokens], maxlen=max_seq_len - 1, padding="pre")
        raw_probs = model.predict(padded, verbose=0)[0]
        adj_probs = _apply_temperature(raw_probs, temperature)

        if temperature <= 0:
            next_idx = int(np.argmax(adj_probs))
        else:
            next_idx = int(np.random.choice(len(adj_probs), p=adj_probs))

        next_word = _index_to_word(tokenizer, next_idx)
        if not next_word:
            break

        output += " " + next_word
        tokens.append(next_idx)
        if len(tokens) >= max_seq_len:
            tokens.pop(0)

    return output


def predict_beam_search(
    model,
    tokenizer,
    config: dict,
    seed_text: str,
    num_words: int = 10,
    beam_width: int = 3,
):
    """
    Generate sequences using beam search (more coherent than random sampling).

    Args:
        model:        Loaded Keras model.
        tokenizer:    Fitted Keras Tokenizer.
        config:       Dict with 'max_sequence_len'.
        seed_text:    Starting text.
        num_words:    Number of words to generate per beam.
        beam_width:   Number of beams to maintain.

    Returns:
        List of (generated_text, log_score) tuples sorted best-first.
    """
    max_seq_len = config["max_sequence_len"]
    tokens = _text_to_token_list(tokenizer, seed_text, max_seq_len)

    # Each beam: [token_list, text_so_far, cumulative_log_prob]
    beams = [(list(tokens), seed_text.strip(), 0.0)]

    for _ in range(num_words):
        candidates = []
        for beam_tokens, beam_text, beam_score in beams:
            padded = pad_sequences([beam_tokens], maxlen=max_seq_len - 1, padding="pre")
            probs = model.predict(padded, verbose=0)[0]

            top_indices = np.argsort(probs)[::-1][:beam_width]
            for idx in top_indices:
                word = _index_to_word(tokenizer, idx)
                if not word:
                    continue
                new_tokens = list(beam_tokens) + [idx]
                if len(new_tokens) >= max_seq_len:
                    new_tokens.pop(0)
                log_prob = beam_score + np.log(probs[idx] + 1e-10)
                candidates.append((new_tokens, beam_text + " " + word, log_prob))

        candidates.sort(key=lambda x: x[2], reverse=True)
        beams = candidates[:beam_width]

    return [(text, score) for _, text, score in beams]


# ---------------------------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Shakespearean LSTM Predictor - CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py --seed "to be or not to"
  python predict.py --seed "what dreams may come" --words 25 --temperature 1.2
  python predict.py --seed "the lady doth protest" --topk 5 --mode topk
  python predict.py --seed "all the world" --mode beam --beam-width 3
        """,
    )
    parser.add_argument("--seed",        type=str,   required=True,  help="Seed text to start prediction from")
    parser.add_argument("--words",       type=int,   default=20,     help="Number of words to generate (default: 20)")
    parser.add_argument("--topk",        type=int,   default=5,      help="Top-k predictions to show (default: 5)")
    parser.add_argument("--temperature", type=float, default=0.7,    help="Sampling temperature 0-2 (default: 0.7)")
    parser.add_argument("--beam-width",  type=int,   default=3,      help="Beam width for beam search (default: 3)")
    parser.add_argument(
        "--mode",
        type=str,
        default="generate",
        choices=["generate", "topk", "beam"],
        help="Prediction mode: 'generate' (default), 'topk', or 'beam'",
    )
    parser.add_argument("--model-path",     type=str, default="model.h5",         help="Path to model.h5")
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer.pickle", help="Path to tokenizer.pickle")
    parser.add_argument("--config-path",    type=str, default="config.json",      help="Path to config.json")
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    model, tokenizer, config = load_model_assets(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        config_path=args.config_path,
    )

    print("=" * 60)
    print(f"  Seed text  : \"{args.seed}\"")
    print(f"  Mode       : {args.mode}")
    print(f"  Temperature: {args.temperature}")
    print("=" * 60)

    if args.mode == "topk":
        predictions = predict_next_words(
            model, tokenizer, config,
            seed_text=args.seed,
            top_k=args.topk,
            temperature=args.temperature,
        )
        print(f"\nTop-{args.topk} predicted next words:\n")
        for rank, (word, prob) in enumerate(predictions, 1):
            bar = "=" * int(prob * 50)
            print(f"  {rank}. {word:<20} {prob:.4f}  [{bar}]")

    elif args.mode == "beam":
        beams = predict_beam_search(
            model, tokenizer, config,
            seed_text=args.seed,
            num_words=args.words,
            beam_width=args.beam_width,
        )
        print(f"\nBeam Search Results (width={args.beam_width}):\n")
        for rank, (text, score) in enumerate(beams, 1):
            print(f"  [{rank}] Score: {score:.4f}")
            print(f"      \"{text}\"\n")

    else:  # generate
        result = generate_text(
            model, tokenizer, config,
            seed_text=args.seed,
            num_words=args.words,
            temperature=args.temperature,
        )
        print(f"\nGenerated Text ({args.words} words):\n")
        print(f"  \"{result}\"\n")


if __name__ == "__main__":
    main()
