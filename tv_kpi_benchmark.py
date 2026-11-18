"""
TV-shows ML workload + deployment KPI benchmark (OpenStack vs non-OpenStack).

Usage
  pip install pandas numpy scikit-learn psutil kagglehub matplotlib
  python tv_kpi_benchmark.py run --env openstack   # on the OpenStack VM
  python tv_kpi_benchmark.py run --env baseline    # on a normal machine/VM
  python tv_kpi_benchmark.py compare               # merges the two JSON files

Two KPI families are produced:
  1. Model KPIs      - RMSE, MAE, R2, top features (is the app useful?)
  2. Deployment KPIs - train time, inference latency p50/p95/p99, throughput,
                       CPU %, peak RAM, dataset load time (is the platform good?)
Task 3 of the brief compares the SECOND family across environments.
"""
import argparse, json, os, re, time, platform
import numpy as np, pandas as pd, psutil
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_data():
    import kagglehub
    t = time.perf_counter()
    path = kagglehub.dataset_download("akrishtverma/tv-shows-dataset")
    csv = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".csv")][0]
    df = pd.read_csv(csv)
    return df, time.perf_counter() - t


def to_num(s):
    """'8.8/10' -> 8.8, '96/100' -> 96, '18+' -> 18, otherwise coerce."""
    return pd.to_numeric(s.astype(str).str.extract(r"(\d+\.?\d*)")[0], errors="coerce")


def prepare(df, target=None):
    df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")
                          or c.lower() in ("id", "title")], errors="ignore")
    target = target or next((c for c in df.columns if "imdb" in c.lower()), None)
    if target is None:
        raise SystemExit(f"No IMDb-like column; pass --target. Columns: {list(df.columns)}")
    y = to_num(df[target])
    X = pd.DataFrame(index=df.index)
    for c in df.columns:
        if c == target:
            continue
        num = to_num(df[c])
        if num.notna().mean() > 0.6:          # mostly numeric-looking column
            X[c] = num.fillna(num.median())
        elif df[c].nunique() <= 30:           # low-cardinality categorical
            X = X.join(pd.get_dummies(df[c].astype(str), prefix=c))
    mask = y.notna()
    return X[mask].astype(float), y[mask], target


def pct(a, q):
    return float(np.percentile(a, q))


def run(env, target, n_requests):
    proc = psutil.Process()
    df, load_s = load_data()
    X, y, target = prepare(df, target)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    psutil.cpu_percent(None)
    t = time.perf_counter()
    model = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42).fit(Xtr, ytr)
    train_s = time.perf_counter() - t
    train_cpu = psutil.cpu_percent(None)

    pred = model.predict(Xte)
    model_kpis = {
        "target": target,
        "rmse": float(np.sqrt(mean_squared_error(yte, pred))),
        "mae": float(mean_absolute_error(yte, pred)),
        "r2": float(r2_score(yte, pred)),
        "top_features": dict(sorted(zip(X.columns, model.feature_importances_),
                                    key=lambda kv: -kv[1])[:5]),
    }

    # single-row inference latency (simulates API requests)
    rows = Xte.sample(n_requests, replace=True, random_state=1).values
    lat = []
    psutil.cpu_percent(None)
    t0 = time.perf_counter()
    for r in rows:
        s = time.perf_counter()
        model.predict(r.reshape(1, -1))
        lat.append((time.perf_counter() - s) * 1000)
    wall = time.perf_counter() - t0
    infer_cpu = psutil.cpu_percent(None)

    out = {
        "env": env,
        "host": {"cpu_count": psutil.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
                 "os": platform.platform()},
        "model_kpis": model_kpis,
        "deployment_kpis": {
            "dataset_load_s": load_s,
            "train_time_s": train_s,
            "train_cpu_pct": train_cpu,
            "latency_ms_p50": pct(lat, 50),
            "latency_ms_p95": pct(lat, 95),
            "latency_ms_p99": pct(lat, 99),
            "throughput_req_s": n_requests / wall,
            "infer_cpu_pct": infer_cpu,
            "peak_rss_mb": proc.memory_info().rss / 2**20,
        },
    }
    json.dump(out, open(f"kpis_{env}.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


def compare():
    import matplotlib.pyplot as plt
    a = json.load(open("kpis_openstack.json"))["deployment_kpis"]
    b = json.load(open("kpis_baseline.json"))["deployment_kpis"]
    tab = pd.DataFrame({"openstack": a, "baseline": b})
    tab["overhead_%"] = (tab.openstack - tab.baseline) / tab.baseline * 100
    print(tab.round(3))
    tab.round(3).to_csv("kpi_comparison.csv")
    tab[["openstack", "baseline"]].plot.barh(figsize=(8, 6), logx=True)
    plt.tight_layout(); plt.savefig("kpi_comparison.png", dpi=150)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run", "compare"])
    p.add_argument("--env", default="baseline")
    p.add_argument("--target")
    p.add_argument("--requests", type=int, default=2000)
    a = p.parse_args()
    run(a.env, a.target, a.requests) if a.cmd == "run" else compare()
