# README images and recordings

Captured on 12 October 2026 (Asia/Dhaka) from the production dashboard at source revision [`e49f735`](https://github.com/urmeo/calm-sense/commit/e49f735f050d7676ede9c6c9206b2cf83a1f51f0). These are browser captures of the actual interface and controls.

| File | Content |
| :-- | :-- |
| [current-dashboard.png](current-dashboard.png) | Saved WESAD results and subject-mean accuracy |
| [signal-explorer.png](signal-explorer.png) | Recorded chest signals with ECG, EDA, temperature, and acceleration enabled |
| [research-pipeline.png](research-pipeline.png) | Pipeline overview and subject-held-out evaluation details |
| [dashboard-tour.mp4](dashboard-tour.mp4) | 32.2 s: dashboard, model comparison, calibration, and pipeline |
| [signal-explorer.mp4](signal-explorer.mp4) | 24.0 s: signal channels, zoom, reset, and subject selection |
| [dashboard-tour.gif](dashboard-tour.gif) | Animated preview of the dashboard video |
| [signal-explorer.gif](signal-explorer.gif) | Animated preview of the signal video |

Screenshots and videos are 1,440 × 1,000 pixels. Videos use H.264, have no audio, and follow the measured capture timing. The browser was sampled at five frames per second; the MP4 files repeat frames at ten frames per second. The 800-pixel-wide GIF previews sample the same recordings at two frames per second.

The dashboard displays saved measurements. Signal excerpts are downsampled and concatenated; displayed time is not one continuous recording. These recordings do not show live inference or a new benchmark run.

The existing build copies this folder to the published site's `/calm-sense/media/` path. The root README uses relative image paths and links to the hosted MP4 files.
