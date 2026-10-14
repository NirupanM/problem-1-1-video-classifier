# Problem 1.1 – Video classifier for chunking-strategy routing

## 1. Goal and classes
The classifier is a **router**: it picks the chunking strategy for a video. Classes are defined by *what boundary signal is meaningful*, not by topic:

| Class (examples) | Chunking strategy |
|---|---|
| `static_talking` (podcast, lecture, interview) | speech-turn / ASR boundaries, ~30-60 s, 1 keyframe/chunk |
| `edited_multishot` (sports broadcast, film, news, vlog) | shot-boundary detection, 1 chunk/shot |
| `egocentric_continuous` (first-person, body-cam, drone, handheld wildlife) | motion/activity-adaptive, overlapping 5-10 s windows |
| `screen_synthetic` (gameplay, screen recording, animation) | UI-state change on non-HUD region + OCR/event cues |
| `static_event` (surveillance, camera trap, livestream cam) | event-triggered around motion bursts, idle skipped |
| `unknown` (low confidence) | generic 10 s windows, 25% overlap |

Wildlife/sports are not their own classes: a nature documentary is `edited_multishot`, a trail-cam is `static_event`. Routing by *structure* keeps the class set small and the label noise low. The `unknown` fallback keeps the router safe on out-of-distribution input: the worst case is the generic strategy, never a wrong specialised one.

## 2. Feature engineering (`src/features.py`)
Constraint: cost must not scale with video length, and must need no GPU. We decode at most ~64 frames per video: 4 windows spread across the video, each with 8 samples spaced 1 s apart, plus a partner frame 2 frames later for motion. All analysis runs at 96x96.

14 features, all duration- and resolution-agnostic:
- **Editing structure**: `cut_rate` (fraction of 1 s steps with HSV-histogram Bhattacharyya > 0.5), median hist-diff, median frame-diff. Separates edited content from continuous shots.
- **Motion** (Farneback flow on the close pair): mean magnitude, *camera motion* (median flow vector), *residual* (flow minus camera motion = object motion), flow coherence. Separates handheld/egocentric (large camera motion) from tripod (residual only).
- **Static layout**: fraction of pixels with temporal std < 4 within a window. High for HUD, overlays, fixed cameras.
- **Appearance**: Canny edge density, saturation mean, quantised-colour entropy (low for synthetic/flat graphics), brightness variance over time.
- **Semantic prior**: Haar face presence and max face area on ≤6 frames (talking heads). It is a stand-in for a cheap person/face detector.

Planned additions (not in this repo, they need a corpus and an audio stack): a speech-ratio feature from a VAD on the audio track, and a 512-d embedding from a small CLIP/MobileNet on 4 frames, concatenated for the final model.

## 3. Model
Standardised features → softmax regression (70 weights, JSON file, numpy only). Rationale: the features are already discriminative; a linear model is calibrated enough for the confidence threshold and runs in microseconds. If accuracy on real data needs it, swap in gradient-boosted trees on the same features (still sub-millisecond) before reaching for a CNN. Confidence < 0.6 → `unknown`.

## 4. Deployment on paper
1. **Ingest**: upload event → queue (SQS/Kafka) with the object-store URI. One message per video.
2. **Workers (CPU only)**: stateless containers, 1 process per core. Each worker seeks into the object (HTTP range reads or local cache), decodes ~64 low-res frames, computes the 14 features and predicts. Measured here: ~0.12 s/video wall at 4 workers on 36 s synthetic clips, so about 0.5 s of CPU per video, ~7k videos/hour/core. Throughput scales linearly; autoscale on queue depth. Seek cost grows only with keyframe interval, not video length.
3. **Output**: `{video_id, class, confidence, strategy, features}` to the metadata store. The chunker reads the strategy and dispatches to the matching chunker; the features are kept for retraining.
4. **Model lifecycle**: model is a ~2 KB JSON loaded at start-up, versioned and hot-swappable. Shadow-deploy new versions and compare disagreement rates against the current one before promotion.
5. **Monitoring and feedback loop**: track class mix, `unknown` rate and feature drift per day. Sample low-confidence and `unknown` videos (plus a random 1%) for human labels; retrain weekly. Downstream chunk-quality signals (e.g. a chunk that spans several shots) are a free label source.
6. **Optional tier-2**: if the `unknown` rate is material, route those videos (a small minority) to a heavier CLIP-embedding classifier; the cheap model remains the first filter.

## 5. Honest status
No corpus was provided. The pipeline is verified end-to-end only on procedurally generated clips (`src/synth.py`), where it scores 100% held-out; **that number says nothing about real footage**. Real accuracy needs labeled data (`python src/main.py train DATA/<class>/*.mp4`). Expected weak spots: gameplay vs. animation vs. edited content (fix: audio and embedding features), and lecture slides (static, no faces, may land in `static_event` or `unknown`). Feature thresholds (cut at 0.5, static std < 4) were set by hand and should be tuned on real data.
