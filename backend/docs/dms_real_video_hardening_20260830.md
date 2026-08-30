# DMS real-video hardening — 2026-08-30

## Outcome

The deployable model remains `backend/models/best.pt`. The video pipeline now treats YOLO boxes as object proposals, not final behavior decisions. A phone alert requires plausible box geometry plus spatial association with the driver's pose; a no-seatbelt alert requires Chest ROI support and must defeat any competing seatbelt detection. Both events still require temporal votes.

The benchmark contains 20 visually audited real-person videos with traceable Pexels pages. It includes handheld phone positives, passenger and mounted-phone hard negatives, a curated belted segment, rain, night, neon, reflections, rear-cabin, side and POV views. Pexels permits free use and modification under its published license: https://www.pexels.com/license/

## Implemented pipeline

1. Vehicle detection and windshield proposal when a usable exterior vehicle view exists.
2. Driver pose and Driver ROI estimation.
3. YOLO DMS inference at 768 pixels on the Driver ROI.
4. A second 768-pixel inference on an enlarged Chest ROI for seatbelt classes.
5. Spatial evidence gating:
   - reject phone boxes with implausible frame area/aspect ratio;
   - require phone center inside Driver ROI and near a visible wrist and head landmark;
   - reject no-seatbelt boxes outside Chest ROI;
   - suppress no-seatbelt when a seatbelt detection has comparable confidence.
6. Twelve-probe temporal voting with separate thresholds for phone and no-seatbelt.
7. Fast full-video coverage: half of the 48 probes are uniform anchors and half prioritize visual scene changes. This replaces evaluating only the first few seconds of a clip.
8. Multi-label, one-video/one-event scoring; cached or neighboring frames never become fake independent TP/FP samples.

## Why these changes follow credible systems

- Overhead seatbelt work first localizes semantic structures (`car`, `windshield`, `passenger`, `seat belt`), uses day/night data, augmentation and an unseen test set. It reports mAP50 97.46% and F1 95.37%, but those controlled detection metrics must not be compared directly to this cross-source behavior benchmark: https://ph.pollub.pl/index.php/acs/article/view/7594
- Robust seatbelt usage recognition explicitly models local belt evidence and assembles it into a global shape to cope with fisheye distortion, IR, low contrast, blur and occlusion. The current Chest ROI second pass is the lightweight deployable approximation; a belt segmentation/polyline model is the next upgrade: https://arxiv.org/abs/2203.00810
- Naturalistic driver-behavior challenge systems use video clips, spatial-temporal models, multi-view ensembling and temporal boundary post-processing rather than classifying unrelated frames independently: https://openaccess.thecvf.com/content/CVPR2024W/AICity/html/Nguyen_Multi-View_Spatial-Temporal_Learning_for_Understanding_Unusual_Behaviors_in_Untrimmed_Naturalistic_CVPRW_2024_paper.html

## Remaining limitations

- Three handheld-phone-positive videos and one independent no-seatbelt-only video are enough to expose gross failures, not enough to claim production accuracy.
- Seatbelt state is often genuinely unobservable in dark/POV clips. Such clips are useful robustness tests but need event-specific ignore labels before per-class seatbelt conclusions.
- MediaPipe pose can fail through a windshield or in extreme darkness. The conservative phone rule then favors fewer false alarms at the cost of missed detections.
- For a roadside production system, collect driver/camera-disjoint footage with explicit time intervals for `vehicle_moving`, `using_phone`, `seatbelt`, `no_seatbelt`, and `unknown`, then report event precision/recall, false alarms per hour, detection latency and per-camera slices.

## Acceptance gates

- Unit tests for geometry, driver association, seatbelt conflict and scene-change sampling must pass.
- The benchmark manifest must contain exactly 20 existing files, source URLs and audit notes.
- Report both image test metrics and independent video event metrics; never substitute mAP50 for behavior-event F1.
- Review every false positive/negative contact sheet and add it to a hard-negative/hard-positive queue before retraining.
