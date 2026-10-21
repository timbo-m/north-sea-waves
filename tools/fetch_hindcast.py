"""Download the missing 1-degree ERA5 hindcast points (lat 51-56, lon -3..9) from Open-Meteo.

Writes data/hindcast_<lat>_<lon>.csv.gz in the same format as the existing files:
2000-01-01..2025-12-31, every 3 h (00, 03, ... 21 UTC), columns
time,hs,dir,tm,tp,swell_h,swell_dir,swell_t,windsea_h,windsea_t,wind_kn,wind_dir.
era5_ocean has no swell / wind-sea partitions, so those columns stay empty, as before.
Only points whose own cell is sea (data/sea_points.json, no snapping) are fetched.
Existing files are skipped, so it can be stopped and restarted.

  py tools/fetch_hindcast.py [--until 13:00]      (UTC)
"""
import argparse, datetime as dt, gzip, io, json, os, sys, time
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
LOG = os.path.join(DATA, "_download_log.txt")
MARINE = "https://marine-api.open-meteo.com/v1/marine"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
START, END = "2000-01-01", "2025-12-31"
WAVE_VARS = ["wave_height", "wave_direction", "wave_period", "wave_peak_period"]
HEADER = "time,hs,dir,tm,tp,swell_h,swell_dir,swell_t,windsea_h,windsea_t,wind_kn,wind_dir\n"
PAUSE = 8.0                      # seconds between requests (Open-Meteo free tier is per-minute limited)

session = requests.Session()
session.headers["User-Agent"] = "north-sea-waves research script"


def log(*a):
    msg = dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%SZ ") + " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def fetch(url, params, deadline):
    """GET with backoff on rate limits, 5xx and network errors. Returns JSON or None at the deadline."""
    wait = 30
    while time.time() < deadline:
        try:
            r = session.get(url, params=params, timeout=300)
        except requests.RequestException as e:
            log("network error:", e, f"retry in {wait}s")
            time.sleep(wait); wait = min(wait * 2, 900); continue
        if r.status_code == 200:
            time.sleep(PAUSE)
            return r.json()
        text = r.text[:200]
        if r.status_code == 429 or "limit" in text.lower():
            pause = 3600 if "daily" in text.lower() else 600 if "hour" in text.lower() else 70
            log("rate limited:", text, f"sleep {pause}s")
            time.sleep(min(pause, max(deadline - time.time(), 0))); continue
        if r.status_code >= 500:
            log("server", r.status_code, text, f"retry in {wait}s")
            time.sleep(wait); wait = min(wait * 2, 900); continue
        raise RuntimeError(f"{r.status_code}: {text}")
    return None


def num(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.2f}".rstrip("0").rstrip(".")
    return str(v)


def get_point(lat, lon, deadline):
    path = os.path.join(DATA, f"hindcast_{lat}_{lon}.csv.gz")
    if os.path.exists(path):
        return "exists"
    m = fetch(MARINE, dict(latitude=lat, longitude=lon, start_date=START, end_date=END,
                           models="era5_ocean", hourly=",".join(WAVE_VARS)), deadline)
    if m is None:
        return "deadline"
    w = fetch(ARCHIVE, dict(latitude=lat, longitude=lon, start_date=START, end_date=END, models="era5",
                            hourly="wind_speed_10m,wind_direction_10m", wind_speed_unit="kn"), deadline)
    if w is None:
        return "deadline"
    mh, wh = m["hourly"], w["hourly"]
    wind = dict(zip(wh["time"], zip(wh["wind_speed_10m"], wh["wind_direction_10m"])))
    buf = io.StringIO(); buf.write(HEADER); n = 0
    for i, t in enumerate(mh["time"]):
        if int(t[11:13]) % 3:
            continue
        hs, d, tm, tp = (mh[v][i] for v in WAVE_VARS)
        ws, wd = wind.get(t, (None, None))
        buf.write(",".join([t, num(hs), num(d), num(tm), num(tp), "", "", "", "", "", num(ws), num(wd)]) + "\n")
        n += 1
    with gzip.open(path + ".part", "wt", encoding="utf-8", newline="") as f:
        f.write(buf.getvalue())
    os.replace(path + ".part", path)
    return f"ok {n} rows {os.path.getsize(path) // 1024} KB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--until", default="13:00", help="stop at this UTC time today (HH:MM)")
    a = ap.parse_args()
    now = dt.datetime.now(dt.timezone.utc)
    hh, mm = map(int, a.until.split(":"))
    deadline = now.replace(hour=hh, minute=mm, second=0, microsecond=0).timestamp()

    sea = json.load(open(os.path.join(DATA, "sea_points.json")))
    pts = [p for p in sea if abs(p["api_lat"] - p["lat"]) < 0.01 and abs(p["api_lon"] - p["lon"]) < 0.01
           and 51 <= p["lat"] <= 56 and -3 <= p["lon"] <= 9 and p["lat"] % 1 == 0 and p["lon"] % 1 == 0]
    todo = [p for p in pts if not os.path.exists(os.path.join(DATA, f"hindcast_{p['lat']}_{p['lon']}.csv.gz"))]
    log(f"fetch_hindcast: {len(pts)} sea points at 1 deg, {len(todo)} missing, deadline {a.until} UTC")
    for k, p in enumerate(todo, 1):
        if time.time() >= deadline:
            log("deadline reached"); break
        try:
            r = get_point(p["lat"], p["lon"], deadline)
        except Exception as e:                    # one bad point shouldn't stop the rest
            r = f"error {e}"
        log(f"[{k}/{len(todo)}]", p["lat"], p["lon"], r)
        if r == "deadline":
            break
    left = sum(not os.path.exists(os.path.join(DATA, f"hindcast_{p['lat']}_{p['lon']}.csv.gz")) for p in pts)
    log(f"fetch_hindcast done: {len(pts) - left}/{len(pts)} present")
    return 0 if left == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
