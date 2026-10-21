"""Turn results/banks.json + grids.npz into the ranked CSV, KML map and PDF report.

  python make_report.py --results results/
"""
import argparse, csv, datetime as dt, html, json, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
LEVEL_NAMES = ["none", "steep", "breaking", "heavy"]
LEVEL_COL = ["#9aa5b1", "#e8b100", "#e8590c", "#c92a2a"]
NAV_DEPTH = 8.0          # banks shallower than this at LAT are charted no-go areas anyway


def r(v, n=1):
    return None if v is None else round(v, n)


def compass(d):
    return COMPASS[int(round(d / 22.5)) % 16]


def trig_at(b, level, T, tp_mid):
    v = b["trig"][level]
    i = int(np.argmin(np.abs(np.array(tp_mid) - T)))
    return v[i]


def sectors_text(rose, frac=0.12):
    tot = sum(rose)
    if not tot:
        return "-"
    order = np.argsort(rose)[::-1]
    keep = [COMPASS[i] for i in order[:4] if rose[i] >= frac * tot] or [COMPASS[i] for i in order[:3]]
    return ", ".join(keep)


def month_text(months):
    names = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
    tot = sum(months)
    if not tot:
        return "-"
    return ", ".join(names[i] for i in np.argsort(months)[::-1][:3] if months[i] > 0.1 * tot)


def trigger_sentence(b, tp_mid):
    """Plain-language forecast trigger for the 'breaking' level."""
    parts = []
    for T in (8, 10, 12, 14):
        h = trig_at(b, "breaking", T, tp_mid)
        if h is not None:
            parts.append(f"Hs {h:.1f} m at Tp {T} s")
    dirs = sectors_text(b["rose"]) if sum(b["rose"]) else ", ".join(b["open_sectors"][:6])
    if not parts:
        return "No realistic sea state makes it break."
    return f"Waves break when the offshore sea reaches {'; '.join(parts)}, from {dirs}."


def write_csv(path, banks, tp_mid):
    cols = ["rank", "name", "lat", "lon", "type", "min_depth_m_LAT", "dries", "offshore_km", "area_km2",
            "hours_per_year_steep", "hours_per_year_breaking", "hours_per_year_heavy", "breaking_storms_per_year",
            "hs_break_tp8", "hs_break_tp10", "hs_break_tp12", "hs_break_tp14",
            "hs_heavy_tp10", "hs_heavy_tp12", "hs_steep_tp10", "hs_steep_tp12",
            "trigger_directions", "worst_months", "worst_storm_date", "worst_storm_hs", "worst_storm_tp",
            "worst_storm_dir", "vessel_density", "forecast_max_level", "forecast_peak_time", "forecast_peak_hs", "forecast_peak_tp",
            "score"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for b in banks:
            wv = b["worst"] or dict(date="", hs=None, tp=None, dir=None)
            w.writerow([b["rank"], b["name"], round(b["lat"], 4), round(b["lon"], 4), b["type"],
                        round(b["min_depth"], 1), b["dries"], round(b["offshore_km"], 1), round(b["area_km2"], 1),
                        r(b["h_steep"]), r(b["h_breaking"]), r(b["h_heavy"]), r(b["storms_per_year"], 2),
                        *[trig_at(b, "breaking", T, tp_mid) for T in (8, 10, 12, 14)],
                        *[trig_at(b, "heavy", T, tp_mid) for T in (10, 12)],
                        *[trig_at(b, "steep", T, tp_mid) for T in (10, 12)],
                        sectors_text(b["rose"]) if sum(b["rose"]) else "open to " + " ".join(b["open_sectors"]),
                        month_text(b["months"]), wv["date"], r(wv["hs"]),
                        r(wv["tp"]), "" if wv["dir"] is None else compass(wv["dir"]), b.get("vessels"),
                        LEVEL_NAMES[b.get("fc_max_level", 0)], b.get("fc_peak", {}).get("time", ""),
                        round(b.get("fc_peak", {}).get("hs", float("nan")), 1),
                        round(b.get("fc_peak", {}).get("tp", float("nan")), 1), r(b["score"], 2)])


def outline(mask, lat, lon):
    """Outer contour(s) of a boolean mask as lon/lat rings."""
    fig = plt.figure()
    cs = plt.contour(lon, lat, mask.astype(float), levels=[0.5])
    rings = [p for p in cs.allsegs[0] if len(p) > 3]
    plt.close(fig)
    return rings


def write_kml(path, banks, grids, tp_mid, top=60):
    lab, lat, lon = grids["label"], grids["lat"], grids["lon"]
    style = "".join(
        f'<Style id="l{i}"><IconStyle><color>ff{c[5:7]}{c[3:5]}{c[1:3]}</color><scale>{0.9 + 0.2 * i}</scale>'
        f'<Icon><href>http://maps.google.com/mapfiles/kml/shapes/caution.png</href></Icon></IconStyle>'
        f'<LineStyle><color>ff{c[5:7]}{c[3:5]}{c[1:3]}</color><width>2</width></LineStyle>'
        f'<PolyStyle><color>55{c[5:7]}{c[3:5]}{c[1:3]}</color></PolyStyle></Style>'
        for i, c in enumerate(LEVEL_COL))
    pm = []
    for b in banks[:top]:
        lvl = (3 if b["h_heavy"] >= 1 else 2 if b["h_breaking"] >= 1 else 1) if b["climate"] else 1
        r0, r1, c0, c1 = b["slices"]
        sub = lab[r0:r1, c0:c1] == b["label"]
        rings = outline(np.pad(sub, 1), np.r_[lat[r0] + 0.005, lat[r0:r1], lat[r1 - 1] - 0.005],
                        np.r_[lon[c0] - 0.005, lon[c0:c1], lon[c1 - 1] + 0.005])
        desc = (f"<b>Rank {b['rank']} - {html.escape(b['type'])}</b><br/>"
                f"Shallowest {b['min_depth']:.1f} m below LAT{' (dries)' if b['dries'] else ''}, "
                f"{b['offshore_km']:.0f} km offshore, {b['area_km2']:.0f} km2<br/>"
                + (f"Breaking seas about {b['h_breaking']:.0f} h/yr, heavy broken water {b['h_heavy']:.0f} h/yr, "
                   f"{b['storms_per_year']:.1f} breaking storms/yr<br/>" if b["climate"] else
                   "No wave history downloaded for this area yet: trigger from physics only<br/>")
                + f"<b>Trigger:</b> {html.escape(trigger_sentence(b, tp_mid))}<br/>"
                + (f"Worst in record: {b['worst']['date']}, Hs {b['worst']['hs']:.1f} m, Tp {b['worst']['tp']:.0f} s "
                   f"from {compass(b['worst']['dir'])}<br/>" if b["worst"] else ""))
        if "fc_peak" in b:
            fp = b["fc_peak"]
            desc += (f"<b>Next 6 days:</b> {LEVEL_NAMES[b['fc_max_level']]} (peak {fp['time'][:16]} UTC, "
                     f"Hs {fp['hs']:.1f} m, Tp {fp['tp']:.0f} s)<br/>")
        desc += "Names and depths are indicative only. Use official charts for navigation."
        geo = f"<Point><coordinates>{b['lon']:.5f},{b['lat']:.5f},0</coordinates></Point>"
        polys = "".join(
            "<Polygon><outerBoundaryIs><LinearRing><coordinates>" +
            " ".join(f"{x:.5f},{y:.5f},0" for x, y in r) +
            "</coordinates></LinearRing></outerBoundaryIs></Polygon>" for r in rings)
        pm.append(f"<Placemark><name>{b['rank']}. {html.escape(b['name'])}</name><styleUrl>#l{lvl}</styleUrl>"
                  f"<description><![CDATA[{desc}]]></description><MultiGeometry>{geo}{polys}</MultiGeometry></Placemark>")
    with open(path, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
                f"<name>North Sea breaking-wave hazard banks</name>{style}"
                "<description>Banks and shoals where big North Sea seas break or steepen. "
                "Yellow = steep seas, orange = breaking seas, red = heavy broken water. "
                "Research output, not for navigation.</description>" + "".join(pm) + "</Document></kml>")


# ---------------------------------------------------------------- figures
DEPTH_CMAP = LinearSegmentedColormap.from_list("sea", ["#e9f4fb", "#9cc9e3", "#3d7fb5", "#173f6b", "#0b1f3a"])


def overview_map(path, grids, banks, field="breaking", title=None, top=12):
    lat, lon, depth, land = grids["lat"], grids["lon"], grids["depth"], grids["land"]
    s = 2
    fig, ax = plt.subplots(figsize=(8.2, 8.6))
    ext = [lon[0], lon[-1], lat[-1], lat[0]]
    ax.imshow(np.where(land[::s, ::s], np.nan, np.clip(depth[::s, ::s], 0, 150)), extent=ext, cmap=DEPTH_CMAP,
              vmin=0, vmax=150, aspect=1 / math.cos(math.radians(56)), interpolation="nearest")
    ax.imshow(np.where(land[::s, ::s], 1, np.nan), extent=ext, cmap=LinearSegmentedColormap.from_list("l", ["#d9d4c7", "#d9d4c7"]),
              aspect=1 / math.cos(math.radians(56)), interpolation="nearest")
    hz = grids[field][::s, ::s].astype(float)
    hz[hz <= 0] = np.nan
    im = ax.imshow(hz, extent=ext, cmap="YlOrRd", norm=LogNorm(0.3, 500), aspect=1 / math.cos(math.radians(56)),
                   interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, shrink=0.55, pad=0.02)
    cb.set_label("hours per year with breaking seas")
    far = [b for b in banks if not b["climate"]]
    ax.plot([b["lon"] for b in far], [b["lat"] for b in far], "^", ms=3, color="#7048e8", alpha=0.8,
            label="breaks in a severe storm (no wave history yet)")
    if far:
        ax.legend(loc="upper left", fontsize=7)
    placed = []
    for b in [b for b in banks if b["climate"]][:top]:
        ax.plot(b["lon"], b["lat"], "o", ms=4, mfc="none", mec="k", mew=0.8)
        if any(abs(b["lon"] - x) < 0.25 and abs(b["lat"] - y) < 0.12 for x, y in placed):
            continue                                   # a higher-ranked label already sits here
        placed.append((b["lon"], b["lat"]))
        ax.annotate(str(b["rank"]), (b["lon"], b["lat"]), xytext=(3, 3), textcoords="offset points", fontsize=7,
                    fontweight="bold")
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title(title or "Where North Sea seas break over the seabed (numbers = rank)", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def forecast_map(path, grids, banks, fc_meta):
    lat, lon, depth, land = grids["lat"], grids["lon"], grids["depth"], grids["land"]
    fig, ax = plt.subplots(figsize=(8.2, 8.0))
    s = 3
    ext = [lon[0], lon[-1], lat[-1], lat[0]]
    ax.imshow(np.where(land[::s, ::s], 0.6, np.nan), extent=ext, cmap="Greys", vmin=0, vmax=1,
              aspect=1 / math.cos(math.radians(56)))
    ax.imshow(np.where(land[::s, ::s], np.nan, np.clip(depth[::s, ::s], 0, 150)), extent=ext, cmap=DEPTH_CMAP,
              vmin=0, vmax=150, aspect=1 / math.cos(math.radians(56)), alpha=0.6)
    for lvl in (1, 2, 3):
        bs = [b for b in banks if b.get("fc_max_level", 0) == lvl]
        ax.scatter([b["lon"] for b in bs], [b["lat"] for b in bs], s=10 + 12 * lvl, c=LEVEL_COL[lvl], edgecolor="k",
                   lw=0.4, label=f"{LEVEL_NAMES[lvl]} ({len(bs)})", zorder=3)
    ax.legend(loc="upper left", fontsize=8, title="worst in forecast")
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_title(f"Forecast ({fc_meta['run']}): banks that will steepen or break", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def bank_figure(path, b, grids, tp_mid, results, fc_meta):
    fig = plt.figure(figsize=(8.2, 6.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.38, wspace=0.35)
    # map with rays
    ax = fig.add_subplot(gs[0, :2])
    rp = os.path.join(results, f"rays_{b['id']}.npz")
    if os.path.exists(rp):
        z = np.load(rp)
        g = z["depth"]
        ext = [float(z["lon0"]), float(z["lon1"]), float(z["lat1"]), float(z["lat0"])]
        H, W = g.shape
        ax.imshow(np.clip(g, 0, 60), extent=ext, cmap=DEPTH_CMAP, vmin=0, vmax=60,
                  aspect=1 / math.cos(math.radians(b["lat"])))
        ax.imshow(np.where(np.isnan(g), 1, np.nan), extent=ext, cmap="Greys", vmin=0, vmax=2,
                  aspect=1 / math.cos(math.radians(b["lat"])))
        tolon = lambda c: ext[0] + (c / (W - 1)) * (ext[1] - ext[0])
        tolat = lambda r: ext[3] - (r / (H - 1)) * (ext[3] - ext[2])
        P = z["paths"] / float(z["res"])
        for k in range(0, P.shape[1], 2):
            ax.plot(tolon(P[:, k, 1]), tolat(P[:, k, 0]), color="w", lw=0.35, alpha=0.6)
        if len(z["row"]):
            sc = ax.scatter(tolon(z["col"]), tolat(z["row"]), c=z["Hb"], cmap="autumn_r", s=6, zorder=3)
            fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.01).set_label("breaking wave height (m)", fontsize=7)
        cs = ax.contour(np.linspace(ext[0], ext[1], W), np.linspace(ext[3], ext[2], H), np.nan_to_num(g, nan=0),
                        levels=[10, 20, 30], colors="k", linewidths=0.4, alpha=0.6)
        ax.clabel(cs, fontsize=6, fmt="%d m")
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_title(f"Ray tracing: Hs {float(z['H']):.1f} m, Tp {float(z['T']):.0f} s from {compass(float(z['dir']))}",
                     fontsize=9)
    else:
        lat, lon, depth = grids["lat"], grids["lon"], grids["depth"]
        r0, r1, c0, c1 = b["slices"]
        pad = 30
        r0, r1, c0, c1 = max(r0 - pad, 0), r1 + pad, max(c0 - pad, 0), c1 + pad
        sub = depth[r0:r1, c0:c1]
        ext = [lon[c0], lon[min(c1, len(lon)) - 1], lat[min(r1, len(lat)) - 1], lat[r0]]
        ax.imshow(np.clip(sub, 0, 60), extent=ext, cmap=DEPTH_CMAP, vmin=0, vmax=60,
                  aspect=1 / math.cos(math.radians(b["lat"])))
        hz = grids["breaking"][r0:r1, c0:c1].astype(float); hz[hz <= 0] = np.nan
        ax.imshow(hz, extent=ext, cmap="YlOrRd", norm=LogNorm(0.3, 500), aspect=1 / math.cos(math.radians(b["lat"])))
        ax.set_title("Breaking hours per year over the bank", fontsize=9)
    ax.plot(b["lon"], b["lat"], "k+", ms=10)
    ax.tick_params(labelsize=7)
    # direction rose
    axr = fig.add_subplot(gs[0, 2], projection="polar")
    rose = np.array(b["rose"], float)
    if rose.sum():
        th = np.deg2rad(np.arange(16) * 22.5)
        axr.bar(th, rose / rose.sum() * 100, width=np.deg2rad(20), color="#e8590c", edgecolor="k", lw=0.3)
    axr.set_theta_zero_location("N"); axr.set_theta_direction(-1)
    axr.set_xticks(np.deg2rad(np.arange(0, 360, 45))); axr.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"], fontsize=7)
    axr.tick_params(labelsize=6)
    axr.set_title("Seas come from (% of\nbreaking hours)", fontsize=8)
    # trigger curves
    axt = fig.add_subplot(gs[1, :2])
    for (k, col, lab) in [("steep", LEVEL_COL[1], "steep seas"), ("breaking", LEVEL_COL[2], "breaking"),
                          ("heavy", LEVEL_COL[3], "heavy broken water")]:
        v = np.array([np.nan if x is None else x for x in b["trig"][k]], float)
        axt.plot(tp_mid, v, color=col, lw=2, label=lab)
    T = np.array(tp_mid)
    axt.plot(T, 9.81 * T ** 2 / (2 * np.pi) / 15, "k:", lw=1, label="steepest sea possible")
    axt.set_xlim(5, 17); axt.set_ylim(0, 16)
    axt.set_xlabel("peak period Tp (s)", fontsize=8); axt.set_ylabel("offshore Hs needed (m)", fontsize=8)
    axt.grid(alpha=0.3); axt.legend(fontsize=7, loc="upper right"); axt.tick_params(labelsize=7)
    axt.set_title("Forecast trigger: sea state above a line sets it off", fontsize=9)
    # forecast
    axf = fig.add_subplot(gs[1, 2])
    if "fc_series" in b:
        t = np.arange(len(b["fc_series"]["hs"])) * 3 / 24
        axf.plot(t, b["fc_series"]["hs"], color="#173f6b", lw=1.5)
        lv = np.array(b["fc_series"]["level"])
        for l in (1, 2, 3):
            axf.scatter(t[lv == l], np.array(b["fc_series"]["hs"])[lv == l], color=LEVEL_COL[l], s=12, zorder=3)
        axf.set_xlabel("days from " + fc_meta["run"], fontsize=7); axf.set_ylabel("forecast Hs (m)", fontsize=7)
        axf.grid(alpha=0.3); axf.tick_params(labelsize=7)
        axf.set_title("Forecast", fontsize=8)
    else:
        axf.axis("off")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- PDF
def build_pdf(path, data, banks, figs, results):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak,
                                    KeepTogether)
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=13)
    small = ParagraphStyle("s", parent=body, fontSize=7.5, leading=9.5)
    h1, h2 = ss["Title"], ParagraphStyle("h2", parent=ss["Heading2"], spaceBefore=6, spaceAfter=4)
    tp = data["tp_mid"]
    hc = data["hindcast"]
    fc = data.get("forecast")
    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.4 * cm,
                            bottomMargin=1.4 * cm, title="North Sea breaking-wave hazard banks")
    W = A4[0] - 3.2 * cm
    S = []
    S.append(Paragraph("North Sea banks where big seas break", h1))
    S.append(Paragraph(
        f"Seabed: EMODnet Bathymetry (about 500 m grid, depths below Lowest Astronomical Tide). "
        f"Wave history: ERA5 at {hc['points']} points in the southern North Sea, {hc['start']} to {hc['end']}. "
        + (f"Forecast: {fc['run']}." if fc else "No forecast loaded."), small))
    S.append(Spacer(1, 6))
    nav = [b for b in banks if b["climate"] and b["type"] == "navigable bank"]
    shal = [b for b in banks if b["climate"] and b["type"] != "navigable bank"]
    far = [b for b in banks if not b["climate"]]
    fc_hot = [b for b in banks if b.get("fc_max_level", 0) >= 2]
    S.append(Paragraph("<b>What this shows</b>", h2))
    bullets = [
        f"In the southern North Sea, where 26 years of wave history were downloaded, {len(nav) + len(shal)} banks and "
        f"shoals break in big seas at least once. {len(nav)} of them are deeper than {NAV_DEPTH:.0f} m at low tide, "
        f"so ships cross them routinely, and those are where breaking seas catch vessels out.",
        f"Across the rest of the North Sea a further {len(far)} banks would break in a severe storm (Hs 9 m at 12 s). "
        "They are ranked by how small a sea sets them off, since their wave history isn't downloaded yet.",
        "The sea starts to break over a bank when the significant wave height reaches roughly half to six tenths of "
        "the water depth. Longer-period seas feel the bottom sooner and break at a lower height, so a 12 to 14 s "
        "northerly swell after a storm is more dangerous over a bank than a short, steep 8 s wind sea of the same height.",
        f"The most dangerous navigable banks are {', '.join(b['name'] for b in nav[:3])}." if nav else "",
        (f"In the current forecast ({fc['run']}), {len(fc_hot)} banks reach breaking level"
         + (f", worst {fc_hot[0]['name']} around {fc_hot[0]['fc_peak']['time'][:16].replace('T', ' ')} UTC." if fc_hot else ".")) if fc else "",
    ]
    for t in bullets:
        if t:
            S.append(Paragraph("&bull; " + t, body))
    S.append(Spacer(1, 4))
    S.append(Image(figs["overview"], width=W * 0.92, height=W * 0.92 * 8.6 / 8.2))
    S.append(PageBreak())

    def table(rows, title, n):
        S.append(Paragraph(title, h2))
        hdr = ["#", "Bank", "Min depth\n(m LAT)", "Breaking\nh/yr", "Storms\n/yr", "Hs to break\nTp 10 s", "Hs to break\nTp 13 s",
               "Seas from", "Forecast"]
        data_ = [hdr]
        for b in rows[:n]:
            h10 = trig_at(b, "breaking", 10, tp); h13 = trig_at(b, "breaking", 13, tp)
            data_.append([str(b["rank"]), Paragraph(html.escape(b["name"]), small), f"{b['min_depth']:.0f}",
                          "-" if b["h_breaking"] is None else f"{b['h_breaking']:.0f}",
                          "-" if b["storms_per_year"] is None else f"{b['storms_per_year']:.1f}",
                          "-" if h10 is None else f"{h10:.1f} m", "-" if h13 is None else f"{h13:.1f} m",
                          Paragraph(sectors_text(b["rose"]) if sum(b["rose"]) else "-", small), LEVEL_NAMES[b.get("fc_max_level", 0)]])
        t = Table(data_, colWidths=[0.8 * cm, 4.6 * cm, 1.6 * cm, 1.5 * cm, 1.3 * cm, 1.9 * cm, 1.9 * cm, 2.0 * cm, 1.8 * cm],
                  repeatRows=1)
        st = [("FONTSIZE", (0, 0), (-1, -1), 7.5), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173f6b")),
              ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
              ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f4f8")]),
              ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c8d0da"))]
        for i, b in enumerate(rows[:n], start=1):
            l = b.get("fc_max_level", 0)
            if l:
                st.append(("BACKGROUND", (8, i), (8, i), colors.HexColor(LEVEL_COL[l])))
        t.setStyle(TableStyle(st))
        S.append(t)

    table(nav, f"Southern North Sea, ranked: navigable banks (deeper than {NAV_DEPTH:.0f} m at low tide)", 30)
    S.append(Spacer(1, 4))
    S.append(Paragraph(
        "Hs is the significant wave height of the open sea near the bank (what a forecast gives you); Tp is the peak "
        "period. '-' means no realistic sea at that period is big enough. Breaking h/yr is the average number of hours "
        "per year that at least 1 in 20 waves breaks over the shallowest part. Depths are at lowest astronomical tide: at "
        "high water or in a positive storm surge the triggers rise by about half the extra depth.", small))
    S.append(PageBreak())
    table(shal, f"Southern North Sea: shallow sandbanks (3 to {NAV_DEPTH:.0f} m at low tide)", 25)
    S.append(PageBreak())
    table(far, "Rest of the North Sea: banks ranked by the sea state that breaks them (physics only)", 35)
    if fc:
        S.append(PageBreak())
        S.append(Paragraph("Next six days", h2))
        S.append(Image(figs["forecast"], width=W * 0.85, height=W * 0.85 * 8.0 / 8.2))
        rows = sorted([b for b in banks if b.get("fc_max_level", 0) >= 1], key=lambda b: -b["fc_peak"]["qb"])[:15]
        if rows:
            d = [["Bank", "Worst level", "When (UTC)", "Hs", "Tp", "From"]]
            for b in rows:
                fp = b["fc_peak"]
                d.append([Paragraph(html.escape(b["name"]), small), LEVEL_NAMES[b["fc_max_level"]],
                          fp["time"][:16].replace("T", " "), f"{fp['hs']:.1f} m", f"{fp['tp']:.0f} s", compass(fp["dir"])])
            t = Table(d, colWidths=[6 * cm, 2.4 * cm, 3.2 * cm, 1.6 * cm, 1.4 * cm, 1.4 * cm])
            t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173f6b")),
                                   ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                                   ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c8d0da"))]))
            S.append(t)
        else:
            S.append(Paragraph("No bank reaches even the steep-sea level in this forecast.", body))
    S.append(PageBreak())
    S.append(Paragraph("The top banks, one by one", h2))
    for b in nav[:8] + shal[:2]:
        if b["id"] not in figs["banks"]:
            continue
        wv = b["worst"]
        if not b["climate"]:
            continue
        txt = (f"<b>{b['rank']}. {html.escape(b['name'])}</b> ({b['lat']:.3f} N, {b['lon']:.3f} E). "
               f"Shallowest {b['min_depth']:.1f} m at low tide, {b['offshore_km']:.0f} km offshore, "
               f"{b['area_km2']:.0f} km2 of breaking ground. About {b['h_breaking']:.0f} hours a year of breaking "
               f"seas in {b['storms_per_year']:.1f} storms, mostly in {month_text(b['months'])}. "
               f"{html.escape(trigger_sentence(b, tp))} Worst in the record: {wv['date']}, Hs {wv['hs']:.1f} m, "
               f"Tp {wv['tp']:.0f} s from {compass(wv['dir'])}.")
        S.append(KeepTogether([Paragraph(txt, body), Spacer(1, 3),
                               Image(figs["banks"][b["id"]], width=W, height=W * 0.8), Spacer(1, 8)]))
    S.append(PageBreak())
    S.append(Paragraph("How it works", h2))
    for t in [
        "<b>Seabed.</b> EMODnet's mean depth grid was cut into shallow cells (under 45 m and at least 2 km from the "
        "coast). For each cell the compass directions open to the sea were found: a direction is blocked if land or a "
        "drying bank lies within 25 km up-wave, which keeps sheltered water such as the Wadden Sea from scoring. "
        "Ground shallower than 3 m at low tide is left out: drying flats break in almost any sea and are charted "
        "no-go areas, so they are not a forecasting question. The Channel and the Norwegian skerry coast are left out too.",
        "<b>Breaking.</b> The surf-finder broke a single swell at 0.78 times the depth. Ships meet a random sea, so this "
        "version uses the Battjes-Janssen model: waves in a sea state follow a Rayleigh spread of heights, and the depth "
        "and wavelength cap the largest wave the water can hold. The fraction above that cap breaks. Before reaching the "
        "bank the sea shoals (group-velocity shoaling from the depth the wave model sees to the bank). Three levels: "
        "steep (1 in 200 waves break), breaking (1 in 20) and heavy broken water (1 in 5).",
        "<b>Triggers.</b> Inverting the model gives, for every cell and peak period, the offshore Hs that reaches each "
        "level. Periods where that Hs would be steeper than real seas get (Hs above a fifteenth of the deep-water "
        "wavelength) are marked as impossible.",
        f"<b>Climate.</b> {hc['years']:.0f} years of ERA5 waves every {hc['dt_h']:.0f} hours were compared with each "
        "cell's triggers, counting only seas from its open directions. Cells that break at least once were grouped "
        "into banks. Ranking favours hours of breaking seas, heavy broken water, size, distance offshore and ship "
        f"traffic (EMODnet vessel density). Wave history was only downloaded at {hc['points']} points this time, so "
        "these statistics cover banks within 110 km of them. Elsewhere a bank is listed if a severe storm sea "
        "(Hs 9 m, Tp 12 s) would break on it, and it is ranked by the Hs that sets it off.",
        "<b>Refraction.</b> The surf-finder ray tracer was run over the top banks for the storm direction and period that "
        "most often sets them off, showing where the wave energy focuses and where it breaks first.",
        "<b>Limits.</b> This is a screening study, not a navigational product. Tidal currents are not included, and "
        "wind against tide over a bank makes seas steeper and break earlier than shown. The 500 m grid misses small "
        "pinnacles and wrecks, sandbanks move, and ERA5 at 0.5 degrees underestimates the very biggest storm peaks. "
        "Bank names come from an approximate gazetteer and should be checked on an Admiralty chart.",
    ]:
        S.append(Paragraph(t, body)); S.append(Spacer(1, 4))
    doc.build(S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    R = a.results
    data = json.load(open(os.path.join(R, "banks.json")))
    g = dict(np.load(os.path.join(R, "grids.npz")))
    banks = data["banks"]
    for b in banks:
        b["type"] = "navigable bank" if b["min_depth"] >= NAV_DEPTH else ("drying sandbank" if b["dries"] else "shallow sandbank")
    nav = [b for b in banks if b["climate"] and b["type"] == "navigable bank"]
    others = [b for b in banks if b["climate"] and b["type"] != "navigable bank"]
    far = [b for b in banks if not b["climate"]]
    ordered = nav + others + far
    for i, b in enumerate(ordered, 1):
        b["rank"] = i
    figdir = os.path.join(R, "figures"); os.makedirs(figdir, exist_ok=True)
    figs = {"overview": os.path.join(figdir, "overview.png"), "banks": {}}
    overview_map(figs["overview"], g, ordered)
    if data.get("forecast"):
        figs["forecast"] = os.path.join(figdir, "forecast.png")
        forecast_map(figs["forecast"], g, ordered, data["forecast"])
    for b in nav[:8] + others[:2]:
        p = os.path.join(figdir, f"bank_{b['rank']:02d}.png")
        bank_figure(p, b, g, data["tp_mid"], R, data.get("forecast"))
        figs["banks"][b["id"]] = p
    write_csv(os.path.join(R, "north_sea_hazard_banks_ranked.csv"), ordered, data["tp_mid"])
    write_kml(os.path.join(R, "north_sea_hazard_banks.kml"), ordered, g, data["tp_mid"])
    build_pdf(os.path.join(R, "North_Sea_breaking_wave_hazard_report.pdf"), data, ordered, figs, R)
    print("wrote CSV, KML, PDF to", R)


if __name__ == "__main__":
    main()
