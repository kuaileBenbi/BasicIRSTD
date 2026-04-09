import argparse
import csv
import json
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import TestSetLoader
from metrics import PD_FA, mIoU
from net import Net

try:
    from thop import profile
except ImportError:
    profile = None


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODELS = ("CompareSPP", "CompareFPN", "ComparePANet", "CompareACM")


@dataclass
class EvalResult:
    seed: int
    model_name: str
    dataset_name: str
    checkpoint: str
    pixacc: float
    miou: float
    pd: float
    fa: float


@dataclass
class ModelComplexity:
    model_name: str
    params_m: float
    flops_g: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the IRSTD-1K neck ablation for SPP/FPN/PANet/ACM and summarize results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_name", type=str, default="IRSTD-1K")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./datasets/Dataset",
        help="Dataset root that contains <dataset_name>/images, masks, and img_idx.",
    )
    parser.add_argument("--model_names", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 3407, 666])
    parser.add_argument("--output_dir", type=str, default="./log/neck_compare")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument(
        "--checkpoint_epoch",
        type=int,
        default=None,
        help="Checkpoint epoch used for evaluation. Defaults to --epochs.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--eval_workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--intervals", type=int, default=10)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Evaluation device, e.g. cuda:0 or cpu.",
    )
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--skip_params", action="store_true")
    parser.add_argument(
        "--force_train",
        action="store_true",
        help="Retrain even if the final checkpoint exists.",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def resolve_output_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (SCRIPT_DIR / path).resolve()
    return path


def resolve_input_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path

    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (SCRIPT_DIR / path).resolve()


def normalize_dataset_root(dataset_dir: str, dataset_name: str) -> Path:
    path = resolve_input_path(dataset_dir)
    train_file_from_root = (
        path / dataset_name / "img_idx" / ("train_" + dataset_name + ".txt")
    )
    train_file_from_dataset = path / "img_idx" / ("train_" + dataset_name + ".txt")

    if train_file_from_root.exists():
        return path
    if train_file_from_dataset.exists():
        return path.parent

    raise FileNotFoundError(
        "Cannot locate dataset split files under '{}' for dataset '{}'. "
        "Expected either '{}/<dataset>/img_idx/train_<dataset>.txt' or '{}/img_idx/train_<dataset>.txt'.".format(
            path, dataset_name, path, path
        )
    )


def get_eval_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def checkpoint_epoch(args: argparse.Namespace) -> int:
    return args.checkpoint_epoch if args.checkpoint_epoch is not None else args.epochs


def checkpoint_path(
    output_dir: Path, dataset_name: str, model_name: str, epoch: int
) -> Path:
    return output_dir / dataset_name / f"{model_name}_{epoch}.pth.tar"


def print_command(command: Sequence[str]) -> None:
    print("[cmd] {}".format(subprocess.list2cmdline(list(command))), flush=True)


def run_command(command: Sequence[str], dry_run: bool) -> None:
    print_command(command)
    if dry_run:
        return
    subprocess.run(list(command), cwd=str(SCRIPT_DIR), check=True)


def train_one_model(
    args: argparse.Namespace,
    dataset_root: Path,
    seed_output_dir: Path,
    seed: int,
    model_name: str,
) -> Path:
    final_checkpoint = checkpoint_path(
        seed_output_dir, args.dataset_name, model_name, checkpoint_epoch(args)
    )
    if final_checkpoint.exists() and not args.force_train:
        print(
            "[skip] training already exists for seed={} model={} at {}".format(
                seed, model_name, final_checkpoint
            ),
            flush=True,
        )
        return final_checkpoint

    command = [
        sys.executable,
        str(SCRIPT_DIR / "train.py"),
        "--model_names",
        model_name,
        "--dataset_names",
        args.dataset_name,
        "--dataset_dir",
        str(dataset_root),
        "--save",
        str(seed_output_dir),
        "--seed",
        str(seed),
        "--nEpochs",
        str(args.epochs),
        "--batchSize",
        str(args.batch_size),
        "--patchSize",
        str(args.patch_size),
        "--threads",
        str(args.threads),
        "--threshold",
        str(args.threshold),
        "--intervals",
        str(args.intervals),
    ]
    if args.device:
        command.extend(["--device", args.device])
    run_command(command, args.dry_run)

    if not args.dry_run and not final_checkpoint.exists():
        raise FileNotFoundError(
            "Training finished but checkpoint was not found: {}".format(
                final_checkpoint
            )
        )
    return final_checkpoint


def unpack_size(size: Sequence[object]) -> Tuple[int, int]:
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError("Unexpected image size format: {!r}".format(size))

    height, width = size

    if torch.is_tensor(height):
        height = int(height.reshape(-1)[0].item())
    else:
        height = int(height)

    if torch.is_tensor(width):
        width = int(width.reshape(-1)[0].item())
    else:
        width = int(width)

    return height, width


def evaluate_checkpoint(
    checkpoint_file: Path,
    model_name: str,
    dataset_name: str,
    dataset_root: Path,
    threshold: float,
    device: torch.device,
    eval_workers: int,
) -> Tuple[float, float, float, float]:
    test_set = TestSetLoader(
        str(dataset_root), dataset_name, dataset_name, img_norm_cfg=None
    )
    test_loader = DataLoader(
        dataset=test_set,
        num_workers=eval_workers,
        batch_size=1,
        shuffle=False,
    )

    net = Net(model_name=model_name, mode="test").to(device)
    checkpoint = torch.load(str(checkpoint_file), map_location=device)
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()

    eval_miou = mIoU()
    eval_pd_fa = PD_FA()

    with torch.no_grad():
        for img, gt_mask, size, _ in test_loader:
            img = Variable(img).to(device)
            pred = net.forward(img)
            height, width = unpack_size(size)
            pred = pred[:, :, :height, :width]
            gt_mask = gt_mask[:, :, :height, :width]

            eval_miou.update((pred > threshold).cpu(), gt_mask)
            eval_pd_fa.update(
                (pred[0, 0, :, :] > threshold).cpu(),
                gt_mask[0, 0, :, :],
                [height, width],
            )

    pixacc, miou = eval_miou.get()
    # PD_FA.get() is brittle in this repo, so compute the final scalars directly.
    pd = float(eval_pd_fa.PD / eval_pd_fa.target) if eval_pd_fa.target else 0.0
    fa = (
        float(eval_pd_fa.dismatch_pixel / eval_pd_fa.all_pixel)
        if eval_pd_fa.all_pixel
        else 0.0
    )

    del net
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return float(pixacc), float(miou), pd, fa


def calculate_complexity(
    model_names: Iterable[str],
    patch_size: int,
    device: torch.device,
) -> Dict[str, ModelComplexity]:
    complexity: Dict[str, ModelComplexity] = {}

    for model_name in model_names:
        net = Net(model_name=model_name, mode="test").to(device)
        net.eval()

        params_m = sum(parameter.numel() for parameter in net.parameters()) / 1e6
        flops_g: Optional[float] = None

        if profile is not None:
            dummy_input = torch.rand(1, 1, patch_size, patch_size, device=device)
            flops, _ = profile(net, inputs=(dummy_input,), verbose=False)
            flops_g = float(flops / 1e9)

        complexity[model_name] = ModelComplexity(
            model_name=model_name,
            params_m=float(params_m),
            flops_g=flops_g,
        )

        del net
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return complexity


def mean_and_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.fmean(values)), float(statistics.stdev(values))


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_raw_csv(
    path: Path, results: List[EvalResult], complexity: Dict[str, ModelComplexity]
) -> None:
    fieldnames = [
        "seed",
        "model_name",
        "dataset_name",
        "checkpoint",
        "pixacc",
        "miou",
        "pd",
        "fa",
        "params_m",
        "flops_g",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            item = asdict(result)
            item["params_m"] = (
                complexity[result.model_name].params_m
                if result.model_name in complexity
                else ""
            )
            item["flops_g"] = (
                complexity[result.model_name].flops_g
                if result.model_name in complexity
                else ""
            )
            writer.writerow(item)


def summarize_results(
    results: List[EvalResult],
    complexity: Dict[str, ModelComplexity],
) -> List[Dict[str, object]]:
    grouped: Dict[str, List[EvalResult]] = {}
    for result in results:
        grouped.setdefault(result.model_name, []).append(result)

    summary_rows: List[Dict[str, object]] = []
    for model_name, model_results in grouped.items():
        model_results = sorted(model_results, key=lambda item: item.seed)
        pixacc_mean, pixacc_std = mean_and_std([item.pixacc for item in model_results])
        miou_mean, miou_std = mean_and_std([item.miou for item in model_results])
        pd_mean, pd_std = mean_and_std([item.pd for item in model_results])
        fa_mean, fa_std = mean_and_std([item.fa for item in model_results])

        row = {
            "model_name": model_name,
            "num_seeds": len(model_results),
            "params_m": (
                complexity.get(model_name).params_m if model_name in complexity else ""
            ),
            "flops_g": (
                complexity.get(model_name).flops_g if model_name in complexity else ""
            ),
            "pixacc_mean": pixacc_mean,
            "pixacc_std": pixacc_std,
            "miou_mean": miou_mean,
            "miou_std": miou_std,
            "pd_mean": pd_mean,
            "pd_std": pd_std,
            "fa_mean": fa_mean,
            "fa_std": fa_std,
        }
        summary_rows.append(row)

    summary_rows.sort(
        key=lambda item: (-float(item["miou_mean"]), float(item["fa_mean"]))
    )
    return summary_rows


def write_summary_csv(path: Path, summary_rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "model_name",
        "num_seeds",
        "params_m",
        "flops_g",
        "pixacc_mean",
        "pixacc_std",
        "miou_mean",
        "miou_std",
        "pd_mean",
        "pd_std",
        "fa_mean",
        "fa_std",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def format_metric(mean_value: object, std_value: object) -> str:
    if mean_value == "":
        return ""
    return "{:.6f} +/- {:.6f}".format(float(mean_value), float(std_value))


def format_optional_float(value: object) -> str:
    if value == "" or value is None:
        return ""
    return "{:.6f}".format(float(value))


def write_summary_markdown(path: Path, summary_rows: List[Dict[str, object]]) -> None:
    lines = [
        "| Model | #Seeds | Params(M) | FLOPs(G) | pixAcc | mIoU | PD | FA |",
        "| --- | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            "| {model_name} | {num_seeds} | {params_m} | {flops_g} | {pixacc} | {miou} | {pd} | {fa} |".format(
                model_name=row["model_name"],
                num_seeds=row["num_seeds"],
                params_m=format_optional_float(row["params_m"]),
                flops_g=format_optional_float(row["flops_g"]),
                pixacc=format_metric(row["pixacc_mean"], row["pixacc_std"]),
                miou=format_metric(row["miou_mean"], row["miou_std"]),
                pd=format_metric(row["pd_mean"], row["pd_std"]),
                fa=format_metric(row["fa_mean"], row["fa_std"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_root = normalize_dataset_root(args.dataset_dir, args.dataset_name)
    output_dir = resolve_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        **vars(args),
        "dataset_dir": str(dataset_root),
        "output_dir": str(output_dir),
        "checkpoint_epoch": checkpoint_epoch(args),
    }
    write_json(output_dir / "neck_compare_config.json", config)

    device = get_eval_device(args.device)
    print("[info] dataset_root={}".format(dataset_root), flush=True)
    print("[info] output_dir={}".format(output_dir), flush=True)
    print("[info] eval_device={}".format(device), flush=True)

    results: List[EvalResult] = []

    for seed in args.seeds:
        seed_output_dir = output_dir / ("seed_" + str(seed))
        seed_output_dir.mkdir(parents=True, exist_ok=True)

        for model_name in args.model_names:
            checkpoint_file = checkpoint_path(
                seed_output_dir, args.dataset_name, model_name, checkpoint_epoch(args)
            )

            if not args.skip_train:
                checkpoint_file = train_one_model(
                    args=args,
                    dataset_root=dataset_root,
                    seed_output_dir=seed_output_dir,
                    seed=seed,
                    model_name=model_name,
                )

            if args.skip_eval:
                continue

            if args.dry_run:
                print(
                    "[dry-run] would evaluate seed={} model={} checkpoint={}".format(
                        seed, model_name, checkpoint_file
                    ),
                    flush=True,
                )
                continue

            if not checkpoint_file.exists():
                raise FileNotFoundError(
                    "Checkpoint required for evaluation does not exist: {}".format(
                        checkpoint_file
                    )
                )

            pixacc, miou, pd, fa = evaluate_checkpoint(
                checkpoint_file=checkpoint_file,
                model_name=model_name,
                dataset_name=args.dataset_name,
                dataset_root=dataset_root,
                threshold=args.threshold,
                device=device,
                eval_workers=args.eval_workers,
            )
            result = EvalResult(
                seed=seed,
                model_name=model_name,
                dataset_name=args.dataset_name,
                checkpoint=str(checkpoint_file),
                pixacc=pixacc,
                miou=miou,
                pd=pd,
                fa=fa,
            )
            results.append(result)
            print(
                "[eval] seed={} model={} pixAcc={:.6f} mIoU={:.6f} PD={:.6f} FA={:.6f}".format(
                    seed, model_name, pixacc, miou, pd, fa
                ),
                flush=True,
            )

    complexity: Dict[str, ModelComplexity] = {}
    if not args.skip_params and not args.dry_run:
        complexity = calculate_complexity(args.model_names, args.patch_size, device)
        for model_name in args.model_names:
            item = complexity[model_name]
            flops_text = (
                "{:.6f}".format(item.flops_g) if item.flops_g is not None else "N/A"
            )
            print(
                "[complexity] model={} params={:.6f}M flops={}G".format(
                    model_name, item.params_m, flops_text
                ),
                flush=True,
            )

    write_json(
        output_dir / "neck_compare_raw.json",
        [asdict(result) for result in results],
    )

    if results:
        write_raw_csv(output_dir / "neck_compare_raw.csv", results, complexity)
        summary_rows = summarize_results(results, complexity)
        write_summary_csv(output_dir / "neck_compare_summary.csv", summary_rows)
        write_summary_markdown(output_dir / "neck_compare_summary.md", summary_rows)
        write_json(output_dir / "neck_compare_summary.json", summary_rows)
    else:
        print("[info] no evaluation results were generated.", flush=True)


if __name__ == "__main__":
    main()
