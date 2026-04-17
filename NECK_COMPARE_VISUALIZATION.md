# Neck Compare Visualization Notes

This note explains what the four necks are doing in `model/NeckCompare/model_NeckCompare.py`, how the current `IRSTD-1K` comparison can be read, and how to use `visualize_neck_compare.py` to inspect their feature-focus behavior on small targets.

## Current Quantitative Signal

The current summary in `log/neck_compare/neck_compare_summary.md` shows a consistent pattern:

| Model | mIoU | PD | FA |
| --- | ---: | ---: | ---: |
| CompareSPP | 0.6171 +/- 0.0073 | 0.9293 +/- 0.0058 | 0.000072 +/- 0.000016 |
| ComparePANet | 0.6167 +/- 0.0100 | 0.9147 +/- 0.0108 | 0.000033 +/- 0.000009 |
| CompareACM | 0.6131 +/- 0.0054 | 0.9304 +/- 0.0051 | 0.000071 +/- 0.000001 |
| CompareFPN | 0.6119 +/- 0.0104 | 0.9080 +/- 0.0070 | 0.000032 +/- 0.000007 |

The signal behind these numbers is useful:

- `SPP` gives the highest `mIoU`, which usually means the target region is highlighted more completely and the final response is spatially cleaner.
- `ACM` gives the highest `PD`, which usually means weak or ambiguous targets are amplified more aggressively.
- `FPN` and `PANet` produce lower `FA`, which usually means they are more conservative and suppress background clutter more strongly.

This already suggests that the four necks are not just different in score, but different in *how* they allocate attention between small targets and background.

## What Each Neck Is Doing

### 1. SPP

`SPPNeck` builds context from the deepest feature by pooling at multiple spatial scales and then fusing back to shallow features.

- Core idea: use multi-scale pooled context to decide whether a bright local response is supported by surrounding structure.
- In practice: this often improves target completeness and reduces fragmented responses around the target.
- Expected visual effect: the fused map often looks more like a compact target blob with a cleaner local neighborhood.

What to inspect:

- `context`
- `fuse3`
- `fuse2`
- `fuse1`
- final `fused`

Interpretation:

- If `context` already suppresses the background and `fuse1` keeps a sharp hotspot at the target, then SPP is using context to reject clutter while preserving target shape.

### 2. FPN

`FPNNeck` is a standard top-down semantic fusion path.

- Core idea: send strong semantic information from deep layers to shallow layers.
- In practice: this usually stabilizes responses and reduces noisy local peaks.
- Expected visual effect: the final heatmap is often smoother and less noisy, but small-target peaks can also become less sharp.

What to inspect:

- `p4_lateral`
- `p3_topdown`
- `p2_topdown`
- `p1_topdown`
- final `fused`

Interpretation:

- If the final map is smooth and background false responses are reduced, but the target peak is less concentrated than in SPP or ACM, that matches the lower-FA / slightly lower-PD pattern.

### 3. PANet

`PANetNeck` adds a bottom-up path after the FPN-style top-down path.

- Core idea: after high-level semantics are injected downward, pass refined localization information upward again.
- In practice: this can help recover some spatial precision that pure top-down fusion may blur.
- Expected visual effect: compared with FPN, the target may stay slightly more localized, while background suppression remains fairly strong.

What to inspect:

- `p3_topdown`
- `p2_topdown`
- `p1_topdown`
- `n2_bottomup`
- `n3_bottomup`
- `n4_bottomup`
- final `fused`

Interpretation:

- The key comparison is between `p*` and `n*`.
- If `n2_bottomup` and `n3_bottomup` tighten the target hotspot relative to `p2_topdown` and `p3_topdown`, then the extra bottom-up pass is helping recover target localization.

### 4. ACM

`ACMNeck` uses `AsymBiChaFuse`, which is a gated fusion mechanism instead of a plain add/concat.

- Core idea: use one branch to produce channel guidance from the higher-level feature, and another branch to produce spatial guidance from the lower-level feature.
- In practice: this tends to amplify weak targets more aggressively than plain top-down fusion.
- Expected visual effect: the target hotspot is often brighter and easier to detect, but some background regions may also be activated more often.

Inside `AsymBiChaFuse`:

- `topdown(xh)` produces a channel gate from the high-level feature.
- `bottomup(xl)` produces a spatial gate from the low-level feature.
- Final fusion is `2 * xl * topdown_gate + 2 * xh * bottomup_gate`.

What to inspect:

- `p3_bottomup_weight`
- `p2_bottomup_weight`
- `p1_low_term`
- `p1_high_term`
- final `fused`

Interpretation:

- If `p3_bottomup_weight` or `p2_bottomup_weight` already forms a clean hotspot near the target, ACM is learning a spatial gate that strongly favors the target region.
- If `p1_high_term` becomes large near the target while the fused output stays compact, ACM is successfully pulling high-level semantics into the right spatial region.
- If the hotspot is strong but spreads into nearby clutter, that explains why PD is high but FA is not as low as FPN/PANet.

## How to Read the New Visualization Script

`visualize_neck_compare.py` saves one directory per sample. For each sample it produces:

- `input_and_gt.png`
- `comparison_overview.png`
- `CompareSPP_detail.png`
- `CompareFPN_detail.png`
- `ComparePANet_detail.png`
- `CompareACM_detail.png`
- `activation_stats.json`

### 1. input_and_gt.png

This shows:

- the full image with a cyan target-focused crop box
- the local crop with the GT contour

Use it to make sure you are reading the right local structure around the target.

### 2. comparison_overview.png

This is the most important quick-look figure.

- First row: prediction overlays for all four models
- Second row: fused-feature heatmap overlays for all four models

How to read it:

- If one neck shows a brighter hotspot exactly on the GT and weaker responses elsewhere, it is concentrating attention more selectively on the target.
- If one neck shows a smooth response spread over the local neighborhood, it is using more contextual support but may blur the target peak.
- If one neck lights up multiple clutter points, it is over-activating background structure.

### 3. *_detail.png

This is where the fusion mechanism becomes interpretable.

The first row shows:

- input with crop box
- GT crop
- prediction overlay
- final fused map

The second row shows neck-specific intermediate maps.

This answers the actual research question:

- where the target becomes salient
- whether the neck sharpens or smooths the response
- whether the neck uses context, semantic broadcast, bottom-up refinement, or gating to create the final target focus

### 4. activation_stats.json

This file stores simple target-vs-background activation statistics for the fused map and key intermediate maps.

The most useful quantity is:

- `target_background_ratio`

How to read it:

- larger value: the map is more target-focused relative to the background
- smaller value: the map is either diffuse or background-contaminated

This is not a replacement for visual inspection, but it is useful when two heatmaps look similar by eye.

## Suggested Visual Analysis Workflow

Use the same seed for all four models. Do not mix different seeds across models.

Pick samples in four groups:

- samples that all four models detect
- samples detected by `SPP` and `ACM` but not by `FPN` or `PANet`
- samples where `FPN` or `PANet` suppress clutter better
- samples where all four fail

Then ask the same questions for every sample:

- Which neck first creates a strong hotspot near the target?
- Which neck keeps the hotspot compact?
- Which neck suppresses surrounding clutter most effectively?
- Which neck amplifies weak targets most aggressively?
- Does the final prediction agree with what the fused map suggests?

## Practical Reading Guide

### If SPP looks best

You will usually see:

- deeper `context` already suppressing irrelevant background
- `fuse1` producing a clean, compact hotspot
- final fused map covering the target more completely

That supports the idea that multi-scale pooling is giving useful local context for target-vs-background discrimination.

### If FPN looks best

You will usually see:

- a progressively smoother target response from `p4` down to `p1`
- fewer isolated background peaks
- a conservative but stable final fused map

That supports the idea that top-down semantic broadcasting is reducing noise more than it is sharpening tiny peaks.

### If PANet looks best

You will usually see:

- top-down maps similar to FPN at first
- bottom-up maps restoring some target localization
- a final fused map that keeps both semantic stability and some spatial sharpness

That supports the idea that the second pass is recovering information that pure top-down fusion loses.

### If ACM looks best

You will usually see:

- spatial gates already focusing on the target neighborhood
- the gated high-level term becoming strong near the target
- a final fused map with very bright target emphasis

That supports the idea that ACM is not just merging features, but selectively reweighting them so weak small-target cues are amplified.

## Recommended Command

Example:

```bash
python visualize_neck_compare.py ^
  --dataset_name IRSTD-1K ^
  --dataset_dir ./datasets/Dataset ^
  --checkpoint_root ./log/neck_compare ^
  --seed 3407 ^
  --checkpoint_epoch 400 ^
  --sample_ids XDU23 XDU79 XDU158 ^
  --save_dir ./visualizations/neck_compare_seed3407
```

If you are unsure which seed to use, start with one fixed seed and inspect 10 to 20 representative samples before trying another seed.

## Bottom Line

The main qualitative difference is:

- `SPP`: context-driven background filtering
- `FPN`: conservative semantic smoothing
- `PANet`: top-down plus localization feedback
- `ACM`: gated target amplification

Your current scores already match this story rather well:

- `SPP` is strongest for region quality
- `ACM` is strongest for finding weak targets
- `FPN` and `PANet` are stronger when background suppression matters most
