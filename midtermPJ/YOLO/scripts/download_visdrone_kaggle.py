import argparse
import shutil
import subprocess
from pathlib import Path
from zipfile import ZipFile


def run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(
            "命令执行失败：\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stdout:\n{p.stdout}\n"
            f"  stderr:\n{p.stderr}\n"
            "请先确保：\n"
            "1) 已安装 kaggle CLI：pip install kaggle\n"
            "2) 已放置 kaggle.json（见 README）\n"
        )


def unzip_archive(zip_path: Path, output_dir: Path) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(f"zip 文件不存在：{zip_path}")
    with ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)


def locate_visdrone_zip(output_dir: Path) -> Path:
    preferred = output_dir / "visdrone.zip"
    if preferred.exists():
        return preferred

    candidates = sorted(output_dir.glob("*.zip"))
    if not candidates:
        raise FileNotFoundError(f"下载目录中未找到 zip 文件：{output_dir}")
    return candidates[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data/raw", help="下载与解压输出目录")
    ap.add_argument("--dataset", type=str, default="evilspirit05/visdrone", help="Kaggle dataset slug")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    kaggle = shutil.which("kaggle")
    if kaggle is None:
        raise SystemExit("未找到 kaggle CLI，请先运行：pip install kaggle")

    # Kaggle datasets download 默认输出为 `<ref_short>.zip`，避免强行指定 -f。
    if not any(out_dir.glob("*.zip")):
        run([kaggle, "datasets", "download", "-d", args.dataset, "-p", str(out_dir)])

    zip_path = locate_visdrone_zip(out_dir)
    unzip_archive(zip_path, out_dir)

    print(f"[OK] 下载并解压完成：{out_dir}")
    print("接下来运行：python scripts/convert_visdrone_det_to_yolo.py --raw_root data/raw --out_root data/visdrone_yolo")


if __name__ == "__main__":
    main()

