"""
src/build_index.py
Build the demo knowledge-base index used by the API and Gradio app.

Usage:
    python -m src.build_index                        # arXiv abstracts -> indexes/arxiv
    python -m src.build_index --hotpotqa --n 500     # pooled HotpotQA paragraphs -> indexes/hotpotqa
"""

import argparse

from configs.training_config import RAGConfig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hotpotqa", action="store_true", help="index HotpotQA dev paragraphs instead of arXiv")
    p.add_argument("--n", type=int, default=500, help="HotpotQA questions whose paragraphs are pooled")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="output directory")
    args = p.parse_args()

    config = RAGConfig()
    if not args.hotpotqa:
        from src.rag_pipeline import build_arxiv_index

        config.index_dir = args.out or config.index_dir
        build_arxiv_index(config)
        return

    from sentence_transformers import SentenceTransformer

    from src.hotpotqa import load_hotpotqa
    from src.retriever import DenseRetriever

    _, docs = load_hotpotqa(args.n, seed=args.seed)
    retriever = DenseRetriever.build(docs, SentenceTransformer(config.embedding_model))
    out = args.out or "indexes/hotpotqa"
    retriever.save(out)
    print(f"Index with {retriever.index.ntotal} passages saved to {out}")


if __name__ == "__main__":
    main()
