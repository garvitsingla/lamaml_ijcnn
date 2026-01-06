import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt


def safe_load_json(path: str):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, required=True, help="e.g., PutNextLocal, GoToLocal, PickupDist")
    p.add_argument("--metrics_root", type=str, default="metrics", help="root folder that contains env subfolders")
    p.add_argument("--out_dir", type=str, default="plots", help="where to save plots")
    p.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"])
    p.add_argument("--smooth", type=int, default=0, help="moving average window (0 = no smoothing)")
    args = p.parse_args()

    env = args.env
    base_dir = os.path.join(args.metrics_root, env)   # ✅ this is the key change
    os.makedirs(args.out_dir, exist_ok=True)

    paths = {
        "maml": (
            os.path.join(base_dir, "maml_avg_steps.npy"),
            os.path.join(base_dir, "maml_meta.json"),
        ),
        "lang_conditioned": (
            os.path.join(base_dir, "lang_conditioned_avg_steps.npy"),
            os.path.join(base_dir, "lang_conditioned_meta.json"),
        ),
        "ld_maml": (
            os.path.join(base_dir, "lang_avg_steps.npy"),
            os.path.join(base_dir, "lang_meta.json"),
        ),
    }

    # load
    series = {}
    meta = {}
    for key, (npy_path, json_path) in paths.items():
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"Missing file: {npy_path}")
        series[key] = np.load(npy_path)
        meta[key] = safe_load_json(json_path)

    def maybe_smooth(y: np.ndarray, w: int):
        if w <= 1:
            return y
        kernel = np.ones(w) / w
        y_pad = np.pad(y, (w - 1, 0), mode="edge")
        return np.convolve(y_pad, kernel, mode="valid")

    labels = {
        "maml": meta["maml"].get("label", "MAML"),
        "lang_conditioned": meta["lang_conditioned"].get("label", "Language-conditioned policy"),
        "ld_maml": meta["ld_maml"].get("label", "Language-adapted (LA/LD-MAML)"),
    }

    plt.figure(figsize=(9, 5))
    for key in ["maml", "lang_conditioned", "ld_maml"]:
        y = maybe_smooth(series[key].astype(float), args.smooth)
        x = np.arange(1, len(y) + 1)
        plt.plot(x, y, label=labels[key])

    plt.xlabel("Meta-iteration / Batch")
    plt.ylabel("Average steps per episode")
    plt.title(f"Training curve comparison ({env})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(args.out_dir, f"avg_steps_comparison_{env}.{args.format}")
    plt.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()