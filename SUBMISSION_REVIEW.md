# CalmSense submission review

**Reviewed:** 24 October 2026 · **Score: 8.6/10**

## Verdict

The project works in the tested environment and is ready to demonstrate as research software. Python tests, the dashboard build, browser interactions, saved-model loading, chart generation, and artifact integrity checks pass.

The saved WESAD results are historical. They do not establish the performance of the current preprocessing code. A submission that requires new, reproducible research results needs a fresh real-data benchmark first. No marking rubric was supplied; this score assesses software quality and evidence, not classification accuracy.

## What the project does

CalmSense compares four feature classifiers and a 1D CNN on recorded wearable signals. The main evaluation holds out one complete WESAD subject at a time. Separate runs examine probability calibration, enrollment-based recalibration, sensor placement, feature subsets, and transfer to PhysioNet Non-EEG.

The browser reads saved results and recorded clips. It does not perform live stress inference or clinical assessment. The README now follows that path: signals and models, held-out results, supporting studies, then setup.

## Assessment

| Area | Score | Evidence |
| :-- | --: | :-- |
| Runtime correctness | 24/25 | Main paths and publication-failure regressions pass. |
| Evaluation evidence | 17/25 | Subject separation is explicit; current-code real-data results remain missing. |
| Reproducibility | 17/20 | All nine numeric charts regenerate; fresh Python setup and real-data reruns remain untested. |
| Maintainability | 14/15 | Shared publication logic, direct control flow, and three regression modules. |
| Documentation and interface | 14/15 | Clear project story, data-derived figures, accessible chart labels, and stated limits. |
| **Total** | **86/100** | **8.6/10** |

## Verification

Tested on macOS with Python 3.12.14, Node.js 24.4.1, NumPy 2.5.3, and scikit-learn 1.6.1. All 15 installed runtime requirements satisfy their declared ranges.

| Check | Result |
| :-- | :-- |
| Python suite | **444 tests passed** |
| Source coverage | **89.66%**; 2194 of 2447 statements |
| Black / Pyflakes / Pyright | Pass; Pyright reports zero errors and warnings |
| Frontend tests | **23 passed** |
| TypeScript and production build | Pass |
| Local browser checks | **26 passed**, covering six pages, both themes, and desktop/mobile layouts |
| Chart refresh | All nine numeric SVGs generated; plotted values, source hashes, and failure recovery checked |
| Existing assets | Four older SVGs, the pipeline, 14 saved result files, and 13 historical PNGs unchanged |
| Saved classifier | Checksum passes; 58-feature synthetic inputs return finite, normalized probabilities |
| Synthetic calibration | Completes with finite JSON and three readable figures |
| NumPy compatibility regression | Three affected tests pass with the pre-2.0 API surface simulated |
| Media and links | Ten SVGs parse; images, the 84-second video, and local Markdown targets checked |
| npm advisories | Zero known vulnerabilities in the checked lockfile tree |
| Direct Python advisories | Zero OSV matches for the 15 installed direct runtime dependencies |

The clean npm install and hosted-demo browser checks also passed during this review. The hosted demo records an older deployment source; those checks do not establish that this exact reviewed build is deployed.

The synthetic calibration run reports a scikit-learn/SciPy warning about the unused `iprint` solver option but completes. The model smoke test validates its interface, not predictive performance. The NumPy API check is not a separate NumPy 1.x installation.

## Fixes completed

- ROC generation now fails before publication when every estimator fails. Partial success remains supported.
- Calibration, personalization, ablation, and transfer stage metrics with figures. Plotting or publication errors preserve the previous result set.
- Three HRV test expectations use SciPy integration within the declared dependency range.
- The existing chart command now regenerates the five newer figures alongside the four older ones. It rejects mismatched study snapshots, invalid plotted values, changed inputs, and failed writes before publishing a complete chart set.
- README sections explain the actual signal and model paths. Historical results, separate protocols, uncertainty, and operating limits remain visible.

## Remaining evidence gap

The saved benchmark is dated **4 October 2026**, source `4f5c02f`. It predates RR correction, wrist EDA, SCR detection, filter-order, and enrollment-sampling fixes. Current source fingerprints differ, and the benchmark-linkage guard correctly refuses to treat that run as current-source evidence.

The original source revision is absent from this checkout's reachable history. Rewriting history does not recreate it. Keep the historical labels until the benchmark and supporting studies are rerun together.

[Provenance](outputs/results/provenance.json) preserves the old result hashes. Its historical publication manifest also names a removed GIF and an older dashboard export; it is not a complete integrity certificate for today's checkout.

## Security review

No credentials, API tokens, private keys, or project-owned `.env` files were found. The redacted pattern scan checked **152 working-tree text files** and **1,387 historical text blobs** across **1,127 commits** reachable from the public branch, tag, and 18 old pull-request heads.

The scan did not decode binary assets or audit raw data, installed dependency source, global credential stores, provider validity, or unreachable server objects. Pattern matching cannot prove that every possible secret is absent. The Python advisory check covers direct installed dependencies, not their full transitive tree.

Saved pickle/joblib artifacts remain trusted-input interfaces. Verify the checksum from a trusted checkout; a checksum does not authenticate an untrusted model.

A new parentless `main` removes previous ancestors from that branch. Old pull-request references and cached commits can remain accessible, as [GitHub documents](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository). This review does not claim that account-wide contribution activity or every remote trace has been erased.

## Five next improvements

1. **Rerun the real-data studies from the frozen source.** Rebuild features, run the main benchmark, then refresh supporting studies, model, curves, tables, and exports together.
2. **Record one exact Python environment.** Test a fresh install and the supported dependency boundaries, including the scikit-learn/SciPy combination.
3. **Retain held-out prediction records on the next run.** Store subject IDs, true labels, probabilities, and split metadata so curves and paired comparisons can be reproduced directly.
4. **Provide accessible numeric chart data.** Keep concise downloadable tables for detailed reliability and decision curves alongside the figures.
5. **Publish the exact verified dashboard build.** Confirm its deployed source matches the reviewed package and run the browser checks again.

## Scope

Source code, experiment scripts, the three Python test modules, frontend files, documentation, and configuration were read. Structured outputs and media were parsed, hashed, or decoded. Changes and added regressions were reviewed again.

A full real-data benchmark, fresh Python installation, hosted Colab setup, field validation, clinical validation, latency testing, and battery measurements were not performed. Detailed check evidence is saved outside the repository in `../final-cleanup-2026-10-24/` and the preceding review folders.

**Final decision: working research prototype, 8.6/10. Ready to showcase with historical results labeled; a current-code research submission still needs the real-data rerun.**
