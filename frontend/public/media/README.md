# Demo media

These browser captures show saved results and recorded signal excerpts. They do not show live inference or a new benchmark run.

| File | Content | Capture source |
| :-- | :-- | :-- |
| [current-dashboard.png](current-dashboard.png) | Saved WESAD results and subject-mean accuracy | 12 October 2026, `e49f735f050d7676ede9c6c9206b2cf83a1f51f0` |
| [signal-explorer.png](signal-explorer.png) | Four signal groups and the SVG download button | 14 October 2026, modified frontend based on `7700fd3671b090df7596f1cb0d8da873ab7250e3` |
| [research-pipeline.png](research-pipeline.png) | Five-step pipeline, both training paths, and saved results | 14 October 2026, modified frontend based on `7a91a1437e6f58bd6c18ce3815e7815817573841` |
| [dashboard-tour.mp4](dashboard-tour.mp4) | 84-second demo: dashboard, model comparison, calibration, explainability, signal controls, and pipeline | 22 October 2026, `e613c0426647a0ebe72f45378177de50c20f5764` |
| [dashboard-tour.gif](dashboard-tour.gif) | Animated preview of the complete demo | Same capture as the MP4 |

Capture dates use Asia/Dhaka. The demo records the local production build. All fingerprinted frontend files match `e613c04`; frontend SHA-256: `ed860b56468c4d51033295337a852100bbefd1e52d1a18fd404ef8a500b85533`.

The MP4 is 1,440 × 1,000 pixels, H.264, 10 fps, and silent. Browser images were sampled every 200 ms; playback follows the measured timestamps. The GIF is 800 × 556 pixels and samples the full timeline at 1.5 fps.

The screenshots retain their earlier capture sources. The pipeline screenshot is 1,440 × 1,500 pixels; its frontend SHA-256 is `a1c35d99314fa375f30101f6cf6ed492db4dc514ed1e56340975a3e186cc3611`. The Signal Explorer screenshot is 1,440 × 1,100 pixels; its frontend SHA-256 is `0326e55c5881a63dd021e607c90cd826df0ffa45c3e85312de3206dc6d85f7c9`. Fingerprints cover tracked and non-ignored frontend files, excluding deleted paths, dependencies, build, cache, and media folders.

Signal excerpts are downsampled and concatenated. Displayed time is not one continuous recording.

The root README shows one animated preview linked to the repository MP4. The static build also copies these files to `/calm-sense/media/`.
