#!/usr/bin/env python3
"""
app.py
------
Flask app exposing:
  GET  /                         -> simple UI: click a necklace, see matching earrings
  GET  /api/necklaces            -> JSON list of necklaces
  GET  /api/recommend/<id>       -> JSON top-k earring matches for a necklace id
  GET  /static/images/<file>     -> product images

Run:
    python app.py
    then open http://localhost:5000
"""
from flask import Flask, jsonify, render_template, abort, request

from matcher import get_index

app = Flask(__name__)


@app.route("/")
def index():
    idx = get_index()
    return render_template("index.html", necklaces=idx.necklaces)


@app.route("/api/necklaces")
def api_necklaces():
    idx = get_index()
    return jsonify([
        {"id": p.id, "image_file": p.image_file} for p in idx.necklaces
    ])


@app.route("/api/recommend/<necklace_id>")
def api_recommend(necklace_id):
    idx = get_index()
    top_k = request.args.get("top_k", default=3, type=int)
    if necklace_id not in idx.by_id or idx.by_id[necklace_id].product_type != "Necklace":
        abort(404, description=f"Unknown necklace id: {necklace_id}")
    results = idx.recommend(necklace_id, top_k=top_k)
    return jsonify({"necklace_id": necklace_id, "matches": results})


if __name__ == "__main__":
    # Pre-warm the index (computes features for all 20 images once) so the
    # first request isn't slow.
    get_index()
    app.run(host="0.0.0.0", port=5000, debug=True)
