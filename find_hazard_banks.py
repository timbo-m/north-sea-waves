"""Find North Sea banks and shoals where big seas break or steepen dangerously
for ships, and the forecast conditions that set them off.

Pipeline
  1. Seabed: EMODnet bathymetry tiles (depth to Lowest Astronomical Tide),
     Natural Earth land mask, and for every shallow cell the compass sectors
     that are open to the sea (no land or drying bank within 25 km up-wave).
  2. Physics: for each shallow cell and each peak period, the offshore Hs that
     makes 1 in 200, 1 in 20 and 1 in 5 waves break on that depth
     (wave_physics.py: shoaling + Battjes-Janssen random-wave breaking).
  3. Climate: 25 years of ERA5 wave hindcast at 0.5 degree points. Counting how
     often the offshore sea (Hs, Tp, direction) beats each cell's trigger gives
     hours per year of steep / breaking / heavy broken water.
  4. Banks: cells that break at least once in the record are grouped into banks,
     named, scored and ranked. Each bank gets its trigger table, the storm
     directions and months that set it off, and its worst storm in the record.
  5. Refraction: the surf-finder ray tracer (point_breaks.trace) is run over the
     top banks for their worst storm direction to show where waves focus.
  6. Forecast: the latest ECMWF wave forecast is checked against every trigger.

  python find_hazard_banks.py --bathy "data/bathy_*.tif" --hindcast data/ \
      --land ne_10m_land.shp --forecast data/forecast_ecmwf.npz --out results/
"""
import argparse, glob, gzip, json, math, os, sys
import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wave_physics as wp
from gazetteer import name_for

TP_BINS = np.arange(3.0, 21.0, 1.0)             # bin edges (s)
TP_MID = 0.5 * (TP_BINS[:-1] + TP_BINS[1:])
NSEC = 16                                        # 22.5 degree compass sectors
SECT = np.arange(NSEC) * 22.5
COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
SHALLOW_MAX = 45.0                               # deeper cells never break in North Sea seas
SHALLOW_MIN = 3.0                                # shallower ground (drying flats) breaks in any sea: charted, not a forecast question
NORTH_SEA = dict(lat_min=51.0, nor_lat=57.8, nor_lon=4.5)   # leave out the Channel and the Norwegian skerries


# ---------------------------------------------------------------- seabed
def load_bathy(pattern, step=0.005):
    import rasterio
    from rasterio.merge import merge
    srcs = [rasterio.open(p) for p in sorted(glob.glob(pattern))]
    if not srcs:
        raise SystemExit(f"no bathymetry matching {pattern}")
    arr, tr = merge(srcs, res=(step, step), nodata=np.nan)
    elev = arr[0].astype(np.float32)
    nd = srcs[0].nodata
    if nd is not None and np.isfinite(nd):
        elev[elev == nd] = np.nan
    elev[np.abs(elev) > 1e4] = np.nan
    H, W = elev.shape
    lon = tr.c + tr.a * (np.arange(W) + 0.5)
    lat = tr.f + tr.e * (np.arange(H) + 0.5)
    return elev, lat, lon, tr


def land_mask(shp, tr, shape):
    from rasterio import features
    import shapefile
    from shapely.geometry import shape as shp_shape, box
    r = shapefile.Reader(shp)
    H, W = shape
    bb = box(tr.c, tr.f + tr.e * H, tr.c + tr.a * W, tr.f)
    geoms = [g for g in (shp_shape(s.__geo_interface__) for s in r.shapes()) if g.intersects(bb)]
    return features.rasterize(((g, 1) for g in geoms), out_shape=shape, transform=tr, dtype="uint8").astype(bool)


def open_sectors(blocked, lat, step, reach_km=25.0, coarse=4):
    """Bit mask per cell of the compass sectors with no blocking cell within reach_km up-wave."""
    H, W = blocked.shape
    bc = ndimage.maximum_filter(blocked, size=coarse)[::coarse, ::coarse]
    h, w = bc.shape
    dy = step * coarse * 110.57
    dx = step * coarse * 111.32 * np.cos(np.deg2rad(np.mean(lat)))
    out = np.zeros((h, w), np.int32)
    for s, ang in enumerate(SECT):
        hit = np.zeros((h, w), bool)
        a = np.deg2rad(ang)                       # waves FROM this bearing: look that way
        for dist in np.arange(dy, reach_km + 1e-6, min(dx, dy)):
            r = int(round(-dist * np.cos(a) / dy)); c = int(round(dist * np.sin(a) / dx))
            sh = np.zeros_like(bc)
            ys, yd = (slice(max(r, 0), h + min(r, 0)), slice(max(-r, 0), h + min(-r, 0)))
            xs, xd = (slice(max(c, 0), w + min(c, 0)), slice(max(-c, 0), w + min(-c, 0)))
            sh[yd, xd] = bc[ys, xs]
            hit |= sh
        out |= (~hit).astype(np.int32) << s
    full = np.repeat(np.repeat(out, coarse, 0), coarse, 1)[:H, :W]
    return full


def sector_of(dirdeg):
    return (np.round(np.asarray(dirdeg) / 22.5).astype(int)) % NSEC


# ---------------------------------------------------------------- hindcast
def load_hindcast(folder):
    import pandas as pd
    pts = []
    for p in sorted(glob.glob(os.path.join(folder, "hindcast_*.csv*"))):
        base = os.path.basename(p).replace(".csv.gz", "").replace(".csv", "")
        try:
            _, la, lo = base.split("_")
            la, lo = float(la), float(lo)
        except ValueError:
            continue
        df = pd.read_csv(p)
        df = df.dropna(subset=["hs", "tp", "dir"])
        if len(df) < 1000:
            continue
        t = pd.to_datetime(df["time"])
        pts.append(dict(lat=la, lon=lo, hs=df["hs"].to_numpy(np.float32), tp=df["tp"].to_numpy(np.float32),
                        dir=df["dir"].to_numpy(np.float32), time=t.to_numpy(),
                        wind=df["wind_kn"].to_numpy(np.float32) if "wind_kn" in df else None,
                        dt_h=float(np.median(np.diff(t.to_numpy()).astype("timedelta64[m]").astype(float))) / 60,
                        years=(t.iloc[-1] - t.iloc[0]).days / 365.25))
    if not pts:
        raise SystemExit(f"no hindcast files in {folder}")
    return pts


def nearest_point(pts, la, lo):
    P = np.array([[p["lat"], p["lon"]] for p in pts])
    cl = np.cos(np.deg2rad(55.0))
    from scipy.spatial import cKDTree
    t = cKDTree(np.c_[P[:, 0], P[:, 1] * cl])
    dist, idx = t.query(np.c_[la, lo * cl])
    return idx, dist * 111.0


# ---------------------------------------------------------------- core
def trigger_table(depth, d0):
    """[level, tp_bin, cell] offshore Hs needed (NaN = can't happen)."""
    out = np.empty((len(wp.LEVELS), len(TP_MID), depth.size), np.float32)
    for i, (_, q, _) in enumerate(wp.LEVELS):
        for j, T in enumerate(TP_MID):
            out[i, j] = wp.hs_trigger(q, T, depth, d0)
    return out


def exceed_hours(pts, cell_pt, trig, open_bits):
    """Hours per year each cell spends at or above each level (direction-aware)."""
    nl, nb, nc = trig.shape
    hours = np.zeros((nl, nc), np.float32)
    for pi, p in enumerate(pts):
        cells = np.nonzero(cell_pt == pi)[0]
        if not len(cells):
            continue
        tb = np.digitize(p["tp"], TP_BINS) - 1
        sec = sector_of(p["dir"])
        scale = p["dt_h"] / max(p["years"], 1e-6)
        for b in range(nb):
            for s in range(NSEC):
                hs = np.sort(p["hs"][(tb == b) & (sec == s)])
                if not len(hs):
                    continue
                ok = ((open_bits[cells] >> s) & 1).astype(bool)
                if not ok.any():
                    continue
                cc = cells[ok]
                for l in range(nl):
                    th = trig[l, b, cc]
                    n = len(hs) - np.searchsorted(hs, np.nan_to_num(th, nan=np.inf))
                    hours[l, cc] += n * scale
    return hours


def vessel_sampler(path):
    """Max EMODnet vessel density within ~5 km of a point (any CRS)."""
    import rasterio
    from rasterio.warp import transform
    src = rasterio.open(path)
    arr = src.read(1).astype(float)
    arr[(arr < 0) | (arr > 1e30)] = np.nan
    if src.nodata is not None:
        arr[arr == src.nodata] = np.nan

    def f(la, lo):
        x, y = transform("EPSG:4326", src.crs, [lo], [la])
        r, c = src.index(x[0], y[0])
        px = max(1, int(round(5000 / abs(src.transform.a)))) if src.crs.is_projected else 3
        w = arr[max(r - px, 0):r + px + 1, max(c - px, 0):c + px + 1]
        return float(np.nanmax(w)) if w.size and np.isfinite(w).any() else None
    return f


def score_bank(b):
    return (math.log1p(b["h_breaking"]) + 0.5 * math.log1p(b["h_heavy"]) +
            0.4 * math.log10(1 + b["area_km2"]) + 0.5 * min(b["offshore_km"] / 20.0, 1.5))


def bank_details(b, p, d, d0, open_bit):
    """Event statistics at a bank's critical cell from its hindcast point."""
    q = wp.qb(p["hs"], p["tp"], d, d0)
    q = np.where(((open_bit >> sector_of(p["dir"])) & 1).astype(bool), q, 0)
    lv = np.zeros(len(q), int)
    for i, (_, thr, _) in enumerate(wp.LEVELS):
        lv[q >= thr] = i + 1
    brk = lv >= 2
    import pandas as pd
    t = pd.to_datetime(p["time"])
    rose = np.bincount(sector_of(p["dir"][brk]), minlength=NSEC) if brk.any() else np.zeros(NSEC, int)
    months = np.bincount(t.month[brk] - 1, minlength=12) if brk.any() else np.zeros(12, int)
    i = int(np.lexsort((p["hs"], q))[-1])         # most breaking, then biggest sea
    # separate storms: breaking records more than 24 h apart
    if brk.any():
        tt = t[brk].values.astype("datetime64[h]").astype(np.int64)
        storms = 1 + int((np.diff(tt) > 24).sum())
    else:
        storms = 0
    b.update(storms_per_year=storms / max(p["years"], 1e-6), rose=rose.tolist(), months=months.tolist(),
             worst=dict(date=str(t[i].date()), hs=float(p["hs"][i]), tp=float(p["tp"][i]),
                        dir=float(p["dir"][i]), qb=float(q[i])),
             event_tp=float(np.median(p["tp"][brk])) if brk.any() else float(p["tp"][i]),
             event_dir=float(SECT[int(np.argmax(rose))]) if brk.any() else float(p["dir"][i]),
             event_hs=float(np.median(p["hs"][brk])) if brk.any() else float(p["hs"][i]),
             record_years=p["years"])
    return b


def load_forecast(path):
    """Forecast as point series: ECMWF grid (.npz from fetch_forecast.py) or the
    Open-Meteo table (csv.gz with lat,lon,time,hs,dir,tp). Returns lat, lon [p],
    time [t], swh/pp1d/mwd [t, p] and run."""
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        LA, LO = np.meshgrid(z["lat"], z["lon"], indexing="ij")
        ok = np.isfinite(z["swh"][0]).ravel()
        f = lambda k: z[k].reshape(len(z["time"]), -1)[:, ok]
        return dict(lat=LA.ravel()[ok], lon=LO.ravel()[ok], time=z["time"], swh=f("swh"), pp1d=f("pp1d"),
                    mwd=f("mwd"), run="ECMWF " + str(z["run"]))
    import pandas as pd
    df = pd.read_csv(path).dropna(subset=["hs"])
    df["time"] = pd.to_datetime(df["time"])
    df = df[df["time"].dt.hour % 3 == 0]
    tp_note = ""
    if df["tp"].isna().all() and "tm" in df:
        # Open-Meteo forecast gives no peak period: Tp = 1.13 Tm (median Tp/Tm in the ERA5 hindcast, Hs > 2 m)
        df["tp"] = df["tm"] * 1.13
        tp_note = " (Tp = 1.13 x mean period)"
    piv = {k: df.pivot_table(index="time", columns=["lat", "lon"], values=k) for k in ("hs", "tp", "dir")}
    cols = piv["hs"].columns
    return dict(lat=np.array([c[0] for c in cols]), lon=np.array([c[1] for c in cols]),
                time=np.array([t.isoformat() for t in piv["hs"].index]),
                swh=piv["hs"].to_numpy(), pp1d=piv["tp"][cols].to_numpy(), mwd=piv["dir"][cols].to_numpy(),
                run="Open-Meteo " + str(piv["hs"].index[0])[:16] + tp_note)


def forecast_levels(fc, la, lo, d, d0, open_bit):
    """Forecast hazard level per time step for one cell (nearest forecast point)."""
    dd = (fc["lat"] - la) ** 2 + ((fc["lon"] - lo) * np.cos(np.deg2rad(la))) ** 2
    p = int(np.argmin(dd))
    H, T, D = fc["swh"][:, p], fc["pp1d"][:, p], fc["mwd"][:, p]
    q = wp.qb(np.nan_to_num(H), np.nan_to_num(T, nan=8.0), d, d0)
    q = np.where(((open_bit >> sector_of(np.nan_to_num(D))) & 1).astype(bool), q, 0)
    lv = np.zeros(len(q), int)
    for k, (_, thr, _) in enumerate(wp.LEVELS):
        lv[q >= thr] = k + 1
    return dict(hs=H, tp=T, dir=D, qb=q, level=lv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bathy", required=True)
    ap.add_argument("--hindcast", required=True)
    ap.add_argument("--land", required=True)
    ap.add_argument("--forecast")
    ap.add_argument("--out", required=True)
    ap.add_argument("--coast-buffer-km", type=float, default=2.0)
    ap.add_argument("--top-rays", type=int, default=8)
    ap.add_argument("--max-point-km", type=float, default=110.0,
                    help="cells further than this from a hindcast point get no climate statistics")
    ap.add_argument("--design-hs", type=float, default=9.0,
                    help="away from the hindcast, a bank counts if this Hs at Tp 12 s makes it break")
    ap.add_argument("--vessels", help="EMODnet vessel density GeoTIFF (optional)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    print("seabed ...", flush=True)
    elev, lat, lon, tr = load_bathy(a.bathy)
    step = abs(tr.a)
    land = land_mask(a.land, tr, elev.shape) | ~np.isfinite(elev)
    depth = np.where(land, np.nan, np.maximum(-elev, 0.3))      # drying banks -> 0.3 m
    drying = (~land) & (elev >= 0)
    km_per_cell = step * 110.57
    dist_land = ndimage.distance_transform_edt(~land) * km_per_cell * 0.8
    blocked = land | (elev > 1.0)
    opn = open_sectors(blocked, lat, step)

    LAT, LON = np.meshgrid(lat, lon, indexing="ij", sparse=True)
    in_sea = ((LAT >= NORTH_SEA["lat_min"]) & ~((LAT > NORTH_SEA["nor_lat"]) & (LON > NORTH_SEA["nor_lon"]))
              & ~((LAT < 55.9) & (LON < -2.2)) & ~(LON < -4.5)      # Irish Sea, Solway, Bristol Channel
              & ~((LAT < 57.4) & (LON > 9.0)))                      # Kattegat, Limfjord, Danish Belts
    shallow = (np.isfinite(depth) & (depth < SHALLOW_MAX) & (depth >= SHALLOW_MIN)
               & (dist_land >= a.coast_buffer_km) & in_sea)
    rr, cc = np.nonzero(shallow)
    print(f"  grid {elev.shape}, shallow cells {len(rr)}", flush=True)

    print("hindcast ...", flush=True)
    pts = load_hindcast(a.hindcast)
    print(f"  {len(pts)} points, {pts[0]['years']:.1f} years, every {pts[0]['dt_h']:.0f} h", flush=True)
    cell_pt, cell_dist = nearest_point(pts, lat[rr], lon[cc])
    near = cell_dist <= a.max_point_km
    cell_pt = np.where(near, cell_pt, -1)
    # The wave-model value stands for the sea around a cell, not the bank top:
    # use the mean depth over ~25 km (wave models run on that kind of grid).
    big = ndimage.uniform_filter(np.where(np.isfinite(depth), np.minimum(depth, 150), 60.0), size=int(0.25 / step))
    dc = depth[rr, cc]
    d0 = np.maximum(big[rr, cc], dc)

    print("triggers + climate ...", flush=True)
    trig = trigger_table(dc, d0)
    hours = exceed_hours(pts, cell_pt, trig, opn[rr, cc])
    hours[:, ~near] = np.nan
    i12 = int(np.argmin(np.abs(TP_MID - 12)))
    phys = np.zeros(depth.shape, bool)
    # Away from the hindcast, skip sheltered inshore water (fjords, island gaps) the 25 km test can miss.
    phys[rr, cc] = (~near) & (np.nan_to_num(trig[1, i12], nan=99) <= a.design_hs) & (dist_land[rr, cc] >= 8.0)
    has_clim = np.zeros(depth.shape, bool); has_clim[rr, cc] = near
    print(f"  {near.mean() * 100:.0f}% of shallow cells have hindcast within {a.max_point_km:.0f} km", flush=True)

    # ------------------------------------------------ banks
    hz = np.zeros(depth.shape, np.float32); hz[rr, cc] = hours[1]
    hv = np.zeros(depth.shape, np.float32); hv[rr, cc] = hours[2]
    st = np.zeros(depth.shape, np.float32); st[rr, cc] = hours[0]
    hz = np.nan_to_num(hz); hv = np.nan_to_num(hv); st = np.nan_to_num(st)
    mask = (hz > 0) | phys
    lab, n = ndimage.label(ndimage.binary_closing(mask, np.ones((3, 3))) & shallow, np.ones((3, 3)))
    print(f"  {n} breaking areas", flush=True)
    cell_area = np.broadcast_to((step * 110.57) * (step * 111.32 * np.cos(np.deg2rad(lat)))[:, None], depth.shape)
    idx_of = -np.ones(depth.shape, np.int64); idx_of[rr, cc] = np.arange(len(rr))
    banks = []
    sl = ndimage.find_objects(lab)
    for k, s in enumerate(sl, start=1):
        m = lab[s] == k
        if m.sum() < 4:
            continue
        r0, c0 = np.unravel_index(np.argmax(np.where(m, depth[s] * -1, -np.inf)), m.shape)   # shallowest
        R, C = s[0].start + r0, s[1].start + c0
        ci = idx_of[R, C]
        brk_area = float(((m & (hz[s] >= 1.0)) * cell_area[s]).sum())
        b = dict(id=len(banks) + 1, label=k, lat=float(lat[R]), lon=float(lon[C]),
                 min_depth=float(depth[R, C]), dries=bool(drying[s][m].any()),
                 area_km2=float((m * cell_area[s]).sum()), area_break_1h_km2=brk_area,
                 offshore_km=float(dist_land[R, C]),
                 climate=bool(has_clim[R, C]),
                 h_steep=float(st[s][m].max()), h_breaking=float(hz[s][m].max()), h_heavy=float(hv[s][m].max()),
                 d0=float(d0[ci]),
                 open_sectors=[COMPASS[i] for i in range(NSEC) if (opn[R, C] >> i) & 1],
                 open_bit=int(opn[R, C]), point=int(cell_pt[ci]),
                 trig={lvl[0]: [None if not np.isfinite(v) else round(float(v), 1) for v in trig[li, :, ci]]
                       for li, lvl in enumerate(wp.LEVELS)},
                 slices=[s[0].start, s[0].stop, s[1].start, s[1].stop])
        b["name"] = name_for(b["lat"], b["lon"])
        banks.append(b)
    vd = vessel_sampler(a.vessels) if a.vessels else None
    for b in banks:
        b["vessels"] = vd(b["lat"], b["lon"]) if vd else None
        if b["climate"]:
            bank_details(b, pts[b["point"]], b["min_depth"], b["d0"], b["open_bit"])
            b["score"] = score_bank(b)
        else:
            # No hindcast nearby: physics only. Rank by how small a sea breaks it.
            b.update(h_steep=None, h_breaking=None, h_heavy=None, storms_per_year=None, rose=[0] * NSEC,
                     months=[0] * 12, worst=None, record_years=None, score=None,
                     event_tp=12.0, event_hs=b["trig"]["breaking"][i12] or a.design_hs,
                     event_dir=next((d for d in (315.0, 337.5, 0.0, 292.5, 22.5, 270.0) if (b["open_bit"] >> int(d / 22.5)) & 1),
                                    float(SECT[int(np.log2(b["open_bit"] & -b["open_bit"]))]) if b["open_bit"] else 315.0))
        if b["vessels"] is not None and b["score"] is not None:
            b["score"] += 0.3 * math.log10(1 + b["vessels"])
    # Navigable banks with wave history first (they get the ray tracing), then shallow ones, then physics-only.
    banks.sort(key=lambda b: (0 if b["min_depth"] >= 8.0 else 1, -b["score"]) if b["score"] is not None
               else (2, b["trig"]["breaking"][i12] or 99.0))
    for i, b in enumerate(banks, 1):
        b["rank"] = i
    seen = {}
    for b in banks:                                    # the highest-ranked piece keeps the plain name
        seen[b["name"]] = seen.get(b["name"], 0) + 1
        if seen[b["name"]] > 1:
            b["name"] += f" ({seen[b['name']]})"

    # ------------------------------------------------ forecast
    fc = None
    if a.forecast and os.path.exists(a.forecast):
        fc = load_forecast(a.forecast)
        for b in banks:
            f = forecast_levels(fc, b["lat"], b["lon"], b["min_depth"], b["d0"], b["open_bit"])
            b["fc_max_level"] = int(f["level"].max())
            k = int(np.argmax(f["qb"]))
            b["fc_peak"] = dict(time=str(fc["time"][k]), hs=float(f["hs"][k]), tp=float(f["tp"][k]),
                                dir=float(f["dir"][k]), qb=float(f["qb"][k]))
            b["fc_series"] = dict(level=f["level"].tolist(), hs=np.nan_to_num(f["hs"]).round(2).tolist(),
                                  tp=np.nan_to_num(f["tp"]).round(1).tolist())
        fc_meta = dict(run=str(fc["run"]), times=[str(t) for t in fc["time"]])
    else:
        fc_meta = None

    np.savez_compressed(os.path.join(a.out, "grids.npz"), lat=lat, lon=lon, depth=depth.astype(np.float32),
                        steep=st, breaking=hz, heavy=hv, label=lab.astype(np.int32), land=land,
                        has_climate=has_clim, points=np.array([[p["lat"], p["lon"]] for p in pts]))
    with open(os.path.join(a.out, "banks.json"), "w") as f:
        json.dump(dict(banks=banks, forecast=fc_meta, tp_mid=TP_MID.tolist(),
                       levels=[dict(key=k, qb=q, text=t) for k, q, t in wp.LEVELS],
                       hindcast=dict(points=len(pts), years=pts[0]["years"], start=str(pts[0]["time"][0])[:10],
                                     end=str(pts[0]["time"][-1])[:10], dt_h=pts[0]["dt_h"])), f, indent=1)
    print(f"{len(banks)} banks written", flush=True)

    # ------------------------------------------------ refraction on top banks
    try:
        rays_for_top(banks[:a.top_rays], depth, lat, lon, step, a.out)
    except Exception as e:                      # rays are a bonus figure, never fatal
        print("ray tracing skipped:", e)


def rays_for_top(banks, depth, lat, lon, step, out):
    """Run the surf-finder ray tracer over each top bank for its worst storm."""
    sys.path.insert(0, os.path.join(HERE, "..", "surf-finder"))
    from point_breaks import trace
    res = 300.0
    for b in banks:
        r0, r1, c0, c1 = b["slices"]
        pad = int(25 / (step * 110.57))
        r0, r1 = max(r0 - pad, 0), min(r1 + pad, depth.shape[0])
        c0, c1 = max(c0 - pad, 0), min(c1 + pad, depth.shape[1])
        sub = depth[r0:r1, c0:c1]
        la, lo = lat[r0:r1], lon[c0:c1]
        ky = step * 110.57e3 / res
        kx = step * 111.32e3 * np.cos(np.deg2rad(b["lat"])) / res
        if sub.shape[0] * ky * sub.shape[1] * kx > 700 * 700:
            f = math.sqrt(700 * 700 / (sub.shape[0] * ky * sub.shape[1] * kx)); ky *= f; kx *= f
        g = ndimage.zoom(np.nan_to_num(sub, nan=-1.0), (ky, kx), order=1)
        rres = step * 110.57e3 / ky
        g = np.where(g <= 0, np.nan, g)
        T = max(b["event_tp"], 7.0)
        tr_ = trace(g, rres, b["event_dir"], T, b["event_hs"], spacing=4 * rres, gamma=0.55)
        np.savez_compressed(os.path.join(out, f"rays_{b['id']}.npz"), depth=g, res=rres,
                            lat0=la.max(), lat1=la.min(), lon0=lo.min(), lon1=lo.max(),
                            row=tr_["row"], col=tr_["col"], Hb=tr_["Hb"], db=tr_["db"],
                            paths=tr_["paths"][:, ::2], T=T, H=b["event_hs"], dir=b["event_dir"])
        print(f"  rays over {b['name']}: {len(tr_['row'])} breaking", flush=True)


if __name__ == "__main__":
    main()
