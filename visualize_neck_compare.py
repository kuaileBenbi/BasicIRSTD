import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model.NeckCompare.model_NeckCompare import resize_to
from net import Net
from utils import Normalized, PadImg, get_img_norm_cfg


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODELS = ("CompareSPP", "CompareFPN", "ComparePANet", "CompareACM")
DETAIL_KEYS = {
    "spp": ("context", "fuse3", "fuse2", "fuse1"),
    "fpn": ("p4_lateral", "p3_topdown", "p2_topdown", "p1_topdown"),
    "panet": ("p3_topdown", "p2_topdown", "n2_bottomup", "n3_bottomup"),
    "acm": ("p3_bottomup_weight", "p2_bottomup_weight", "p1_low_term", "p1_high_term"),
}
DETAIL_TITLES = {
    "context": "SPP context",
    "fuse3": "SPP fuse3",
    "fuse2": "SPP fuse2",
    "fuse1": "SPP fuse1",
    "p4_lateral": "P4 lateral",
    "p3_topdown": "P3 top-down",
    "p2_topdown": "P2 top-down",
    "p1_topdown": "P1 top-down",
    "n2_bottomup": "N2 bottom-up",
    "n3_bottomup": "N3 bottom-up",
    "p3_bottomup_weight": "ACM p3 spatial gate",
    "p2_bottomup_weight": "ACM p2 spatial gate",
    "p1_low_term": "ACM p1 low term",
    "p1_high_term": "ACM p1 high term",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize feature fusion behavior of CompareSPP/CompareFPN/ComparePANet/CompareACM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_name", type=str, default="IRSTD-1K")
    parser.add_argument("--dataset_dir", type=str, default="./datasets/Dataset")
    parser.add_argument("--checkpoint_root", type=str, default="./log/neck_compare")
    parser.add_argument("--save_dir", type=str, default="./visualizations/neck_compare")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_epoch", type=int, default=400)
    parser.add_argument("--model_names", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--sample_ids", nargs="+", default=None)
    parser.add_argument("--num_samples", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Visualization device: 'auto', 'cpu', or a CUDA device such as 'cuda:0'",
    )
    parser.add_argument("--crop_margin", type=int, default=32)
    parser.add_argument("--crop_min_size", type=int, default=96)
    parser.add_argument(
        "--reduce_mode",
        type=str,
        default="mean_abs",
        choices=["mean_abs", "max_abs", "l2"],
        help="How to collapse channel dimensions into a single heatmap.",
    )
    return parser.parse_args()


def resolve_input_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (SCRIPT_DIR / path).resolve()


def resolve_output_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (SCRIPT_DIR / path).resolve()
    return path


def normalize_dataset_root(dataset_dir: str, dataset_name: str) -> Path:
    path = resolve_input_path(dataset_dir)
    root_split = path / dataset_name / "img_idx" / f"test_{dataset_name}.txt"
    dataset_split = path / "img_idx" / f"test_{dataset_name}.txt"
    if root_split.exists():
        return path
    if dataset_split.exists():
        return path.parent
    raise FileNotFoundError(
        "Cannot find '{}' under '{}'.".format(f"test_{dataset_name}.txt", path)
    )


def resolve_device(device_arg: str) -> torch.device:
    if device_arg in (None, "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested via --device={}, but CUDA is not available.".format(device_arg)
        )
    return device


def read_test_ids(dataset_root: Path, dataset_name: str) -> List[str]:
    split_file = dataset_root / dataset_name / "img_idx" / f"test_{dataset_name}.txt"
    return [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def find_sample_file(base_dir: Path, sample_id: str) -> Path:
    for suffix in (".png", ".bmp", ".jpg", ".jpeg"):
        candidate = base_dir / f"{sample_id}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Cannot find sample '{}' under '{}'.".format(sample_id, base_dir))


def load_sample(
    dataset_root: Path,
    dataset_name: str,
    sample_id: str,
    device: torch.device,
) -> Dict[str, object]:
    dataset_dir = dataset_root / dataset_name
    image_path = find_sample_file(dataset_dir / "images", sample_id)
    mask_path = find_sample_file(dataset_dir / "masks", sample_id)

    raw_img = np.array(Image.open(image_path).convert("I"), dtype=np.float32)
    mask = np.array(Image.open(mask_path), dtype=np.float32) / 255.0
    if mask.ndim > 2:
        mask = mask[:, :, 0]

    img_norm_cfg = get_img_norm_cfg(dataset_name, str(dataset_root))
    normalized = Normalized(raw_img, img_norm_cfg)
    padded = PadImg(normalized)
    input_tensor = torch.from_numpy(
        np.ascontiguousarray(padded[np.newaxis, np.newaxis, :])
    ).float().to(device)

    return {
        "sample_id": sample_id,
        "raw_image": raw_img,
        "mask": mask,
        "input_tensor": input_tensor,
        "height": raw_img.shape[0],
        "width": raw_img.shape[1],
    }


def load_compare_model(
    model_name: str,
    checkpoint_file: Path,
    device: torch.device,
) -> torch.nn.Module:
    net = Net(model_name=model_name, mode="test")
    checkpoint = torch.load(str(checkpoint_file), map_location=device)
    net.load_state_dict(checkpoint["state_dict"])
    model = net.model.to(device)
    model.eval()
    return model


def summarize_tensor(tensor: torch.Tensor, reduce_mode: str) -> torch.Tensor:
    tensor = tensor.detach().float()
    if tensor.ndim == 4:
        if reduce_mode == "mean_abs":
            tensor = tensor.abs().mean(dim=1, keepdim=True)
        elif reduce_mode == "max_abs":
            tensor = tensor.abs().max(dim=1, keepdim=True).values
        elif reduce_mode == "l2":
            tensor = torch.sqrt((tensor ** 2).mean(dim=1, keepdim=True) + 1e-12)
    elif tensor.ndim == 3:
        tensor = summarize_tensor(tensor.unsqueeze(0), reduce_mode)
    elif tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError("Unsupported tensor shape for visualization: {}".format(tuple(tensor.shape)))
    return tensor


def tensor_to_heatmap(
    tensor: torch.Tensor,
    output_size: Tuple[int, int],
    reduce_mode: str,
) -> np.ndarray:
    reduced = summarize_tensor(tensor, reduce_mode)
    resized = F.interpolate(reduced, size=output_size, mode="bilinear", align_corners=False)
    return resized[0, 0].detach().cpu().numpy()


def robust_normalize(array: np.ndarray, lower: float = 1.0, upper: float = 99.0) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    low = np.percentile(array, lower)
    high = np.percentile(array, upper)
    if high <= low:
        return np.zeros_like(array)
    return np.clip((array - low) / (high - low), 0.0, 1.0)


def normalize_image(raw_image: np.ndarray) -> np.ndarray:
    return robust_normalize(raw_image, lower=0.5, upper=99.5)


def compute_focus_bbox(
    mask: np.ndarray,
    image_shape: Tuple[int, int],
    margin: int,
    min_size: int,
) -> Tuple[int, int, int, int]:
    height, width = image_shape
    points = np.argwhere(mask > 0.5)
    if len(points) == 0:
        return (0, 0, height, width)

    y_min, x_min = points.min(axis=0)
    y_max, x_max = points.max(axis=0) + 1

    y_min = max(0, y_min - margin)
    x_min = max(0, x_min - margin)
    y_max = min(height, y_max + margin)
    x_max = min(width, x_max + margin)

    crop_h = y_max - y_min
    crop_w = x_max - x_min
    if crop_h < min_size:
        extra = min_size - crop_h
        y_min = max(0, y_min - extra // 2)
        y_max = min(height, y_max + extra - extra // 2)
    if crop_w < min_size:
        extra = min_size - crop_w
        x_min = max(0, x_min - extra // 2)
        x_max = min(width, x_max + extra - extra // 2)

    y_min = max(0, min(y_min, height))
    y_max = max(y_min + 1, min(y_max, height))
    x_min = max(0, min(x_min, width))
    x_max = max(x_min + 1, min(x_max, width))
    return (y_min, y_max, x_min, x_max)


def crop_array(array: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    y_min, y_max, x_min, x_max = bbox
    return array[y_min:y_max, x_min:x_max]


def compute_activation_stats(feature_map: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    target = feature_map[mask > 0.5]
    background = feature_map[mask <= 0.5]
    target_mean = float(target.mean()) if target.size else 0.0
    background_mean = float(background.mean()) if background.size else 0.0
    ratio = float(target_mean / (background_mean + 1e-8))
    contrast = float(target_mean - background_mean)
    return {
        "target_mean": target_mean,
        "background_mean": background_mean,
        "target_background_ratio": ratio,
        "target_background_contrast": contrast,
    }


def render_full_image_with_bbox(
    ax: plt.Axes,
    image: np.ndarray,
    mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    title: str,
) -> None:
    ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    y_min, y_max, x_min, x_max = bbox
    rect = patches.Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        linewidth=1.5,
        edgecolor="cyan",
        facecolor="none",
    )
    ax.add_patch(rect)
    if mask.max() > 0:
        ax.contour(mask, levels=[0.5], colors=["lime"], linewidths=0.8)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def render_overlay(
    ax: plt.Axes,
    image: np.ndarray,
    heatmap: Optional[np.ndarray],
    gt_mask: np.ndarray,
    pred_mask: Optional[np.ndarray],
    title: str,
    cmap: str = "magma",
    alpha: float = 0.5,
) -> None:
    ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    if heatmap is not None:
        ax.imshow(robust_normalize(heatmap), cmap=cmap, alpha=alpha, vmin=0.0, vmax=1.0)
    if gt_mask.max() > 0:
        ax.contour(gt_mask, levels=[0.5], colors=["lime"], linewidths=1.0)
    if pred_mask is not None and pred_mask.max() > 0:
        ax.contour(pred_mask.astype(np.float32), levels=[0.5], colors=["red"], linewidths=0.9)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def select_detail_titles(neck_type: str) -> List[str]:
    return [DETAIL_TITLES[key] for key in DETAIL_KEYS[neck_type]]


def run_spp_debug(neck: torch.nn.Module, features: Sequence[torch.Tensor]) -> OrderedDict:
    c1, c2, c3, c4 = features
    l1 = neck.lateral1(c1)
    l2 = neck.lateral2(c2)
    l3 = neck.lateral3(c3)
    context = neck.context(c4)
    fuse3 = neck.fuse3(context, l3)
    fuse2 = neck.fuse2(fuse3, l2)
    fuse1 = neck.fuse1(fuse2, l1)
    fused = neck.out_refine(fuse1)
    return OrderedDict(
        [
            ("context", context),
            ("fuse3", fuse3),
            ("fuse2", fuse2),
            ("fuse1", fuse1),
            ("fused", fused),
        ]
    )


def run_fpn_debug(neck: torch.nn.Module, features: Sequence[torch.Tensor]) -> OrderedDict:
    c1, c2, c3, c4 = features
    p1 = neck.laterals[0](c1)
    p2 = neck.laterals[1](c2)
    p3 = neck.laterals[2](c3)
    p4 = neck.laterals[3](c4)

    p3_td = neck.smooth[2](p3 + resize_to(p4, p3))
    p2_td = neck.smooth[1](p2 + resize_to(p3_td, p2))
    p1_td = neck.smooth[0](p1 + resize_to(p2_td, p1))

    merged = torch.cat([p1_td, resize_to(p2_td, p1_td), resize_to(p3_td, p1_td), resize_to(p4, p1_td)], dim=1)
    fused = neck.aggregate(merged)
    return OrderedDict(
        [
            ("p4_lateral", p4),
            ("p3_topdown", p3_td),
            ("p2_topdown", p2_td),
            ("p1_topdown", p1_td),
            ("fused", fused),
        ]
    )


def run_panet_debug(neck: torch.nn.Module, features: Sequence[torch.Tensor]) -> OrderedDict:
    c1, c2, c3, c4 = features
    p1 = neck.laterals[0](c1)
    p2 = neck.laterals[1](c2)
    p3 = neck.laterals[2](c3)
    p4 = neck.laterals[3](c4)

    p3_td = neck.top_down[2](p3 + resize_to(p4, p3))
    p2_td = neck.top_down[1](p2 + resize_to(p3_td, p2))
    p1_td = neck.top_down[0](p1 + resize_to(p2_td, p1))

    n1 = p1_td
    n2 = neck.bottom_up[0](p2_td + resize_to(neck.downsample[0](n1), p2_td))
    n3 = neck.bottom_up[1](p3_td + resize_to(neck.downsample[1](n2), p3_td))
    n4 = neck.bottom_up[2](p4 + resize_to(neck.downsample[2](n3), p4))

    merged = torch.cat([n1, resize_to(n2, n1), resize_to(n3, n1), resize_to(n4, n1)], dim=1)
    fused = neck.aggregate(merged)
    return OrderedDict(
        [
            ("p3_topdown", p3_td),
            ("p2_topdown", p2_td),
            ("p1_topdown", p1_td),
            ("n2_bottomup", n2),
            ("n3_bottomup", n3),
            ("n4_bottomup", n4),
            ("fused", fused),
        ]
    )


def run_acm_debug(neck: torch.nn.Module, features: Sequence[torch.Tensor]) -> OrderedDict:
    c1, c2, c3, c4 = features
    p1 = neck.laterals[0](c1)
    p2 = neck.laterals[1](c2)
    p3 = neck.laterals[2](c3)
    p4 = neck.laterals[3](c4)

    p4_to_p3 = resize_to(p4, p3)
    p3_topdown_weight = neck.fuse3.topdown(p4_to_p3)
    p3_bottomup_weight = neck.fuse3.bottomup(p3)
    p3_low_term = p3 * p3_topdown_weight
    p3_high_term = p4_to_p3 * p3_bottomup_weight
    p3_out = neck.refine3(neck.fuse3(p4_to_p3, p3))

    p3_to_p2 = resize_to(p3_out, p2)
    p2_topdown_weight = neck.fuse2.topdown(p3_to_p2)
    p2_bottomup_weight = neck.fuse2.bottomup(p2)
    p2_low_term = p2 * p2_topdown_weight
    p2_high_term = p3_to_p2 * p2_bottomup_weight
    p2_out = neck.refine2(neck.fuse2(p3_to_p2, p2))

    p2_to_p1 = resize_to(p2_out, p1)
    p1_topdown_weight = neck.fuse1.topdown(p2_to_p1)
    p1_bottomup_weight = neck.fuse1.bottomup(p1)
    p1_low_term = p1 * p1_topdown_weight
    p1_high_term = p2_to_p1 * p1_bottomup_weight
    p1_out = neck.refine1(neck.fuse1(p2_to_p1, p1))

    merged = torch.cat([p1_out, resize_to(p2_out, p1_out), resize_to(p3_out, p1_out), resize_to(p4, p1_out)], dim=1)
    fused = neck.aggregate(merged)
    return OrderedDict(
        [
            ("p3_bottomup_weight", p3_bottomup_weight),
            ("p2_bottomup_weight", p2_bottomup_weight),
            ("p1_low_term", p1_low_term),
            ("p1_high_term", p1_high_term),
            ("p1_out", p1_out),
            ("fused", fused),
        ]
    )


def forward_with_debug(model: torch.nn.Module, x: torch.Tensor) -> Tuple[OrderedDict, torch.Tensor]:
    features = model.encoder(x)
    if model.neck_type == "spp":
        neck_debug = run_spp_debug(model.neck, features)
    elif model.neck_type == "fpn":
        neck_debug = run_fpn_debug(model.neck, features)
    elif model.neck_type == "panet":
        neck_debug = run_panet_debug(model.neck, features)
    elif model.neck_type == "acm":
        neck_debug = run_acm_debug(model.neck, features)
    else:
        raise ValueError("Unsupported neck type: {}".format(model.neck_type))

    logits = model.head(neck_debug["fused"])
    logits = resize_to(logits, x)
    pred = torch.sigmoid(logits)
    return neck_debug, pred


def prepare_model_payload(
    model_name: str,
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    output_size: Tuple[int, int],
    gt_mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    reduce_mode: str,
    threshold: float,
) -> Dict[str, object]:
    with torch.no_grad():
        neck_debug, pred = forward_with_debug(model, input_tensor)

    height, width = output_size
    pred = pred[:, :, :height, :width]
    pred_map = pred[0, 0].detach().cpu().numpy()
    pred_mask = (pred_map > threshold).astype(np.float32)

    fused_map = tensor_to_heatmap(neck_debug["fused"], output_size, reduce_mode)
    fused_stats = compute_activation_stats(fused_map, gt_mask)

    detail_maps: OrderedDict[str, np.ndarray] = OrderedDict()
    detail_stats: Dict[str, Dict[str, float]] = {}
    for key in DETAIL_KEYS[model.neck_type]:
        raw_map = tensor_to_heatmap(neck_debug[key], output_size, reduce_mode)
        detail_maps[key] = raw_map
        detail_stats[key] = compute_activation_stats(raw_map, gt_mask)

    y_min, y_max, x_min, x_max = bbox
    return {
        "model_name": model_name,
        "neck_type": model.neck_type,
        "pred_map": pred_map,
        "pred_mask": pred_mask,
        "fused_map": fused_map,
        "fused_stats": fused_stats,
        "detail_maps": detail_maps,
        "detail_stats": detail_stats,
        "crop_pred_map": pred_map[y_min:y_max, x_min:x_max],
        "crop_pred_mask": pred_mask[y_min:y_max, x_min:x_max],
        "crop_fused_map": fused_map[y_min:y_max, x_min:x_max],
        "crop_detail_maps": OrderedDict(
            (key, value[y_min:y_max, x_min:x_max]) for key, value in detail_maps.items()
        ),
    }


def save_input_overview(
    sample_id: str,
    full_image: np.ndarray,
    crop_image: np.ndarray,
    gt_mask: np.ndarray,
    crop_mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    output_file: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    render_full_image_with_bbox(axes[0], full_image, gt_mask, bbox, "Input + target bbox")
    render_overlay(axes[1], crop_image, None, crop_mask, None, "Target crop (GT contour)")
    fig.suptitle(sample_id, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_overview_grid(
    sample_id: str,
    crop_image: np.ndarray,
    crop_mask: np.ndarray,
    payloads: Dict[str, Dict[str, object]],
    output_file: Path,
) -> None:
    model_names = list(payloads.keys())
    fig, axes = plt.subplots(2, len(model_names), figsize=(4 * len(model_names), 8))
    if len(model_names) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for column, model_name in enumerate(model_names):
        payload = payloads[model_name]
        pred_title = "{} pred".format(model_name.replace("Compare", ""))
        fused_title = "{} fused T/B={:.2f}".format(
            model_name.replace("Compare", ""),
            payload["fused_stats"]["target_background_ratio"],
        )
        render_overlay(
            axes[0, column],
            crop_image,
            payload["crop_pred_map"],
            crop_mask,
            payload["crop_pred_mask"],
            pred_title,
            cmap="turbo",
            alpha=0.45,
        )
        render_overlay(
            axes[1, column],
            crop_image,
            payload["crop_fused_map"],
            crop_mask,
            None,
            fused_title,
            cmap="magma",
            alpha=0.50,
        )

    fig.suptitle("{} neck comparison".format(sample_id), fontsize=13)
    fig.tight_layout()
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_model_detail(
    sample_id: str,
    full_image: np.ndarray,
    crop_image: np.ndarray,
    gt_mask: np.ndarray,
    crop_mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    payload: Dict[str, object],
    output_file: Path,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    model_name = payload["model_name"]

    render_full_image_with_bbox(axes[0, 0], full_image, gt_mask, bbox, "Input + bbox")
    render_overlay(axes[0, 1], crop_image, None, crop_mask, None, "GT crop")
    render_overlay(
        axes[0, 2],
        crop_image,
        payload["crop_pred_map"],
        crop_mask,
        payload["crop_pred_mask"],
        "Prediction",
        cmap="turbo",
        alpha=0.45,
    )
    render_overlay(
        axes[0, 3],
        crop_image,
        payload["crop_fused_map"],
        crop_mask,
        None,
        "Fused T/B={:.2f}".format(payload["fused_stats"]["target_background_ratio"]),
        cmap="magma",
        alpha=0.50,
    )

    for axis, key in zip(axes[1], DETAIL_KEYS[payload["neck_type"]]):
        stats = payload["detail_stats"][key]
        render_overlay(
            axis,
            crop_image,
            payload["crop_detail_maps"][key],
            crop_mask,
            None,
            "{} T/B={:.2f}".format(DETAIL_TITLES[key], stats["target_background_ratio"]),
            cmap="magma",
            alpha=0.50,
        )

    fig.suptitle("{} - {}".format(sample_id, model_name), fontsize=13)
    fig.tight_layout()
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_stats_json(
    output_file: Path,
    sample_id: str,
    bbox: Tuple[int, int, int, int],
    payloads: Dict[str, Dict[str, object]],
) -> None:
    data = {
        "sample_id": sample_id,
        "bbox": {
            "y_min": int(bbox[0]),
            "y_max": int(bbox[1]),
            "x_min": int(bbox[2]),
            "x_max": int(bbox[3]),
        },
        "models": {},
    }
    for model_name, payload in payloads.items():
        data["models"][model_name] = {
            "neck_type": payload["neck_type"],
            "fused_stats": payload["fused_stats"],
            "detail_stats": payload["detail_stats"],
        }
    output_file.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_root = normalize_dataset_root(args.dataset_dir, args.dataset_name)
    checkpoint_root = resolve_input_path(args.checkpoint_root)
    save_dir = resolve_output_path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print("Using device: {}".format(device))
    print("Dataset root: {}".format(dataset_root))
    print("Checkpoint root: {}".format(checkpoint_root))
    print("Save dir: {}".format(save_dir))

    all_test_ids = read_test_ids(dataset_root, args.dataset_name)
    sample_ids = args.sample_ids if args.sample_ids else all_test_ids[: args.num_samples]
    if not sample_ids:
        raise RuntimeError("No sample ids selected for visualization.")

    models = {}
    for model_name in args.model_names:
        checkpoint_file = (
            checkpoint_root
            / ("seed_" + str(args.seed))
            / args.dataset_name
            / f"{model_name}_{args.checkpoint_epoch}.pth.tar"
        )
        if not checkpoint_file.exists():
            raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_file))
        models[model_name] = load_compare_model(model_name, checkpoint_file, device)

    manifest = {
        "dataset_name": args.dataset_name,
        "dataset_root": str(dataset_root),
        "checkpoint_root": str(checkpoint_root),
        "seed": args.seed,
        "checkpoint_epoch": args.checkpoint_epoch,
        "sample_ids": sample_ids,
        "model_names": args.model_names,
    }
    (save_dir / "visualization_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    for sample_id in sample_ids:
        sample = load_sample(dataset_root, args.dataset_name, sample_id, device)
        full_image = normalize_image(sample["raw_image"])
        gt_mask = sample["mask"].astype(np.float32)
        bbox = compute_focus_bbox(
            gt_mask,
            (sample["height"], sample["width"]),
            margin=args.crop_margin,
            min_size=args.crop_min_size,
        )
        crop_image = crop_array(full_image, bbox)
        crop_mask = crop_array(gt_mask, bbox)

        sample_dir = save_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        payloads: Dict[str, Dict[str, object]] = OrderedDict()
        for model_name, model in models.items():
            payloads[model_name] = prepare_model_payload(
                model_name=model_name,
                model=model,
                input_tensor=sample["input_tensor"],
                output_size=(sample["height"], sample["width"]),
                gt_mask=gt_mask,
                bbox=bbox,
                reduce_mode=args.reduce_mode,
                threshold=args.threshold,
            )

        save_input_overview(
            sample_id=sample_id,
            full_image=full_image,
            crop_image=crop_image,
            gt_mask=gt_mask,
            crop_mask=crop_mask,
            bbox=bbox,
            output_file=sample_dir / "input_and_gt.png",
        )
        save_overview_grid(
            sample_id=sample_id,
            crop_image=crop_image,
            crop_mask=crop_mask,
            payloads=payloads,
            output_file=sample_dir / "comparison_overview.png",
        )
        for model_name, payload in payloads.items():
            save_model_detail(
                sample_id=sample_id,
                full_image=full_image,
                crop_image=crop_image,
                gt_mask=gt_mask,
                crop_mask=crop_mask,
                bbox=bbox,
                payload=payload,
                output_file=sample_dir / "{}_detail.png".format(model_name),
            )
        save_stats_json(sample_dir / "activation_stats.json", sample_id, bbox, payloads)
        print("Saved visualizations for '{}'.".format(sample_id))


if __name__ == "__main__":
    main()
