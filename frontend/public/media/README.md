# README images and recordings

These are browser captures of the actual interface and controls.

The pipeline has five steps: recorded signals, windows, training, unseen-person testing, and saved results. Its result cards use the saved 4 October 2026 benchmark. Refreshing these captures did not rerun the models.

Original captures were recorded on 12 October 2026 (Asia/Dhaka) at source revision [`e49f735`](https://github.com/urmeo/calm-sense/commit/e49f735f050d7676ede9c6c9206b2cf83a1f51f0). The pipeline screenshot and dashboard tour were refreshed on 14 October 2026 (Asia/Dhaka) from the modified frontend based on [`7a91a14`](https://github.com/urmeo/calm-sense/commit/7a91a1437e6f58bd6c18ce3815e7815817573841).

The refreshed frontend source SHA-256 is `a1c35d99314fa375f30101f6cf6ed492db4dc514ed1e56340975a3e186cc3611`. It covers tracked and non-ignored frontend files, excluding deleted paths, dependencies, build, cache, and media folders. The source stayed unchanged during recording.

| File | Content | Capture source |
| :-- | :-- | :-- |
| [current-dashboard.png](current-dashboard.png) | Saved WESAD results and subject-mean accuracy | Original `e49f735` |
| [signal-explorer.png](signal-explorer.png) | Recorded chest signals with ECG, EDA, temperature, and acceleration enabled | Original `e49f735` |
| [research-pipeline.png](research-pipeline.png) | Complete five-step About pipeline, including both training paths and saved results | Modified frontend based on `7a91a14` |
| [dashboard-tour.mp4](dashboard-tour.mp4) | 32.0 s: dashboard, model comparison, calibration, and the five-step pipeline | Modified frontend based on `7a91a14` |
| [signal-explorer.mp4](signal-explorer.mp4) | 24.0 s: signal channels, zoom, reset, and subject selection | Original `e49f735` |
| [dashboard-tour.gif](dashboard-tour.gif) | Animated preview of the refreshed dashboard video | Modified frontend based on `7a91a14` |
| [signal-explorer.gif](signal-explorer.gif) | Animated preview of the original signal video | Original `e49f735` |

`research-pipeline.png` is 1,440 × 1,500 pixels so the full pipeline fits. The other screenshots and both videos are 1,440 × 1,000 pixels. Videos use H.264, have no audio, and follow the measured capture timing. The browser was sampled at five frames per second; MP4 playback repeats frames at ten frames per second. Both GIF previews are 800 × 556 pixels and sample their recordings at two frames per second.

The dashboard displays saved measurements. Signal excerpts are downsampled and concatenated; displayed time is not one continuous recording. These recordings do not show live inference or a new benchmark run.

The build copies this folder to the published site's `/calm-sense/media/` path. The root README uses relative image paths and links to the hosted MP4 files.
