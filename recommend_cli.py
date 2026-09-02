#!/usr/bin/env python3
"""
recommend_cli.py
-----------------
Command-line demo: pick a necklace, print the top matching earrings.

Usage:
    python recommend_cli.py --list
    python recommend_cli.py --necklace N01 --top 3
    python recommend_cli.py --necklace N01 --top 3 --save-preview out.jpg
"""
import argparse
import os

import cv2
import numpy as np

from matcher import get_index, IMAGES_DIR


def build_preview(necklace_id: str, results: list, out_path: str):
    idx = get_index()
    size = 260
    n_path = idx.by_id[necklace_id].image_path
    necklace_img = cv2.resize(cv2.imread(n_path), (size, size))
    cv2.putText(necklace_img, necklace_id, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    tiles = [necklace_img]
    for r in results:
        img = cv2.resize(cv2.imread(os.path.join(IMAGES_DIR, r["image_file"])), (size, size))
        label = f"{r['id']} {r['score']:.2f}"
        cv2.putText(img, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 0), 2)
        tiles.append(img)

    arrow = np.full((size, 60, 3), 255, dtype=np.uint8)
    cv2.arrowedLine(arrow, (5, size // 2), (55, size // 2), (0, 0, 0), 3, tipLength=0.3)

    row = [tiles[0], arrow] + tiles[1:]
    canvas = np.hstack(row)
    cv2.imwrite(out_path, canvas)


def main():
    parser = argparse.ArgumentParser(description="Necklace -> matching earrings recommender")
    parser.add_argument("--list", action="store_true", help="list all necklace ids and exit")
    parser.add_argument("--necklace", help="necklace product id, e.g. N01")
    parser.add_argument("--top", type=int, default=3, help="number of earring matches to return")
    parser.add_argument("--save-preview", help="optional path to save a side-by-side image")
    args = parser.parse_args()

    idx = get_index()

    if args.list or not args.necklace:
        print("Available necklaces:")
        for p in idx.necklaces:
            print(f"  {p.id}  ({p.image_file})")
        if not args.necklace:
            return

    results = idx.recommend(args.necklace, top_k=args.top)
    print(f"\nTop {len(results)} earring matches for {args.necklace}:")
    for r in results:
        print(f"  {r['id']:>5}  score={r['score']:.4f}  "
              f"(color={r['color_score']:.4f}, shape={r['shape_score']:.4f})  {r['image_file']}")

    if args.save_preview:
        build_preview(args.necklace, results, args.save_preview)
        print(f"\nSaved visual preview -> {args.save_preview}")


if __name__ == "__main__":
    main()
