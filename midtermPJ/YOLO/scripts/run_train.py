import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_data_path(data_arg: str) -> Path:
    data_path = Path(data_arg)
    if data_path.exists():
        return data_path

    repo_relative = repo_root() / data_path
    if repo_relative.exists():
        return repo_relative

    raise SystemExit(f"数据配置不存在: {data_path}")


def resolve_model_arg(model_arg: str) -> str:
    model_path = Path(model_arg)
    if model_path.exists():
        return str(model_path)

    # Allow Ultralytics model aliases like 'yolov8s.pt' to auto-download.
    if model_path.parent == Path("."):
        return model_arg

    raise SystemExit(f"模型不存在: {model_path}")


def next_train_name(project_dir: Path) -> str:
    max_index = 0
    for p in project_dir.iterdir():
        if not p.is_dir():
            continue
        name = p.name.lower()
        if not name.startswith("train-"):
            continue
        suffix = name.split("train-", maxsplit=1)[1]
        if suffix.isdigit():
            max_index = max(max_index, int(suffix))
    return f"train-{max_index + 1}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="yolov8s.pt", help="初始模型权重")
    ap.add_argument("--data", type=str, default="visdrone.yaml", help="数据配置文件")
    ap.add_argument("--project", type=str, default="train_model", help="训练输出根目录")
    ap.add_argument("--epochs", type=int, default=120, help="训练轮数")
    ap.add_argument("--imgsz", type=int, default=1024, help="图像尺寸")
    ap.add_argument("--batch", type=int, default=8, help="batch size")
    ap.add_argument("--device", type=str, default="0", help="设备，如 0 或 cpu")
    ap.add_argument("--workers", type=int, default=8, help="数据加载 workers")
    ap.add_argument("--patience", type=int, default=50, help="早停 patience")
    ap.add_argument("--close_mosaic", type=int, default=15, help="训练末期关闭 mosaic 的 epoch 数")
    ap.add_argument("--cos_lr", action=argparse.BooleanOptionalAction, default=True, help="是否使用余弦学习率")
    ap.add_argument("--multi_scale", action=argparse.BooleanOptionalAction, default=True, help="是否启用多尺度训练")
    ap.add_argument("--no_copy_best", action="store_true", help="不复制 best.pt 到 train_model/best.pt")
    args = ap.parse_args()

    model_name_or_path = resolve_model_arg(args.model)

    data_path = resolve_data_path(args.data)

    project_dir = Path(args.project)
    if not project_dir.is_absolute():
        project_dir = repo_root() / project_dir
    project_dir.mkdir(parents=True, exist_ok=True)
    run_name = next_train_name(project_dir)

    model = YOLO(model_name_or_path)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        close_mosaic=args.close_mosaic,
        cos_lr=args.cos_lr,
        multi_scale=args.multi_scale,
        project=str(project_dir),
        name=run_name,
        exist_ok=False,
        save=True,
        plots=True,
    )

    run_dir = project_dir / run_name
    best_weight = run_dir / "weights" / "best.pt"
    print(f"[OK] 训练输出目录: {run_dir}")

    if (not args.no_copy_best) and best_weight.exists():
        latest_best = project_dir / "best.pt"
        shutil.copy2(best_weight, latest_best)
        print(f"[OK] 已复制最新 best 权重: {latest_best}")


if __name__ == "__main__":
    main()
