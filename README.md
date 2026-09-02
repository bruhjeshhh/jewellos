# Necklace → Earring Matcher (prototype)

Given a necklace, recommends the visually closest-matching earrings from the
provided inventory (5 necklaces, 15 earrings).

## How to run

```bash
pip install -r requirements.txt

# CLI
python recommend_cli.py --list
python recommend_cli.py --necklace N01 --top 3
python recommend_cli.py --necklace N01 --top 3 --save-preview out.jpg   # side-by-side image

# Web app (image in, image out)
python app.py
# open http://localhost:5000, click a necklace, matching earrings render below
```

`GET /api/recommend/<necklace_id>?top_k=3` also works standalone if you just
want the JSON.

First run computes features for all 20 images (~15-20s on CPU); results are
cached to `.feature_cache.pkl` and reused after that (subsequent runs take
well under a second, invalidated automatically if an image file changes).

## Approach

**Constraint that shaped the design:** this environment has no route to
download pretrained model weights (no Hugging Face / torch-hub access), so
a CLIP-embedding approach — which is what I'd actually reach for first, and
what I'd swap in given internet access — wasn't buildable here. I went with
the "colour/feature extraction" path the brief explicitly allows instead.
It's a legitimate approach on its own merits for this problem (jewellery
matching is largely a colour/metal-tone/style judgment), not just a
fallback, but I want to be upfront that it's also the option this
environment left me with.

**Pipeline per image:**

1. **Foreground segmentation (OpenCV GrabCut).** The 20 images use five
   different backgrounds (solid pink, beige, navy mannequin, textured
   fabric, white), so comparing raw pixels would mostly be comparing
   backgrounds. GrabCut, seeded with a centered rectangle (the product is
   roughly centered in every provided image), pulls out the jewellery
   itself. It's imperfect on the textured-fabric shots but good enough to
   stop the background from dominating the signal.

2. **Two feature descriptors, foreground pixels only:**
   - *Colour*: an HSV histogram (30×32×32 bins). This is the primary
     signal — gold vs. oxidised-silver tone, and gemstone colours
     (green/red/pink/clear) are what actually reads as "this goes with
     that" for jewellery.
   - *Shape*: HOG (Histogram of Oriented Gradients) on the masked
     grayscale image, to capture silhouette style — stud vs. jhumka vs.
     chandbali vs. long drop, filigree density, etc.

3. **Scoring:** `0.65 × colour_similarity + 0.35 × shape_similarity`,
   both rescaled to [0, 1] (colour via histogram correlation, shape via
   cosine similarity on the HOG vector). Colour is weighted higher because
   it's the stronger, more reliable signal here; shape mostly acts as a
   tiebreaker between similarly-coloured pieces. The weight is a knob in
   `matcher.py` (`COLOR_WEIGHT`), not tuned against a labelled set — there
   isn't one.

**Sanity check, not a formal eval:** two of the five necklace photos
(N04, N05) happen to show their own matching earrings in the same shot.
Running the recommender on N04, its own pictured earrings (rose-gold,
white CZ, green stone, pearl drop) came back as the #1 match out of 15
candidates — a reasonable, if single, data point that the ranking isn't
arbitrary. N05 (an oxidised-silver/purple-stone set) didn't recover its own
earrings at #1; the segmentation on the more cluttered dark background is
noisier and the colour histogram is more easily confused there. I'm noting
the miss rather than only the hit.

## Tech stack

Python, OpenCV (GrabCut, colour histograms), scikit-image (HOG), Flask +
vanilla JS for the demo UI. No model weights, no internet calls at runtime.

## Known limitations / what I'd change with more time or fewer constraints

- **Swap in CLIP (or a similar pretrained embedding) for the colour/HOG
  pair** given internet access — it would generalize far better than
  hand-crafted features, especially as the catalog grows past a few dozen
  items.
- **GrabCut is fragile on textured backgrounds** (a couple of the fabric
  shots leave background scraps in the mask). A learned saliency/matting
  model would help here too — same constraint as above.
- **No stone/material detection** — e.g. it doesn't explicitly know
  "emerald-green stone" vs. "green fabric", it just sees green pixels. A
  small classifier or object detector could disentangle that.
- **15 earrings is a small enough candidate pool that brute-force
  comparison is fine.** At real catalog scale you'd precompute embeddings
  once and use an ANN index (FAISS, etc.) rather than the linear scan here.
- **The 0.65/0.35 weighting is a reasonable starting point, not a tuned
  hyperparameter** — there's no labelled "correct match" data to tune
  against beyond the two incidental same-shot pairs mentioned above.
