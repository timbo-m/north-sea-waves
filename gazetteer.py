"""Approximate positions of named North Sea banks and coastal reference points.

Used only to put a recognisable name on a shoal the analysis finds. Positions
are rounded and from general knowledge, not from a chart: check any name on
an Admiralty chart before relying on it.
"""
# name, lat, lon
BANKS = [
    ("Goodwin Sands", 51.27, 1.55), ("Sandettie Bank", 51.15, 1.80),
    ("Kentish Knock", 51.65, 1.62), ("Long Sand", 51.60, 1.33),
    ("Gunfleet Sand", 51.75, 1.20), ("Sunk Sand", 51.80, 1.32),
    ("Galloper Bank", 51.73, 1.95), ("Inner Gabbard", 51.92, 1.85),
    ("Outer Gabbard", 51.98, 2.05), ("Shipwash", 51.98, 1.60),
    ("Flemish Banks", 51.25, 2.60), ("Thornton Bank", 51.55, 2.95),
    ("Bligh Bank", 51.65, 2.80),
    ("Scroby Sands", 52.65, 1.78), ("Newcome Sand", 52.47, 1.80),
    ("Haisborough Sand", 52.93, 1.70), ("Hammond Knoll", 52.88, 1.93),
    ("Winterton Ridge", 52.80, 1.85), ("Smith's Knoll", 52.75, 2.20),
    ("Leman Bank", 53.10, 2.20), ("Ower Bank", 53.15, 2.35),
    ("Swarte Bank", 53.40, 2.10), ("Indefatigable Banks", 53.55, 2.25),
    ("Inner Dowsing", 53.25, 0.55), ("Race Bank", 53.13, 0.78),
    ("Docking Shoal", 53.15, 0.65), ("Brown Ridge", 52.65, 3.30),
    ("Cleaver Bank", 53.95, 3.25), ("Horns Rev", 55.50, 7.85),
    ("Wee Bankie", 56.30, -2.10),
]

# Dogger Bank is too big for one point; anything inside this box takes its name.
DOGGER = (54.15, 55.60, 0.80, 4.30)      # lat0, lat1, lon0, lon1

TOWNS = [
    ("Lerwick", 60.15, -1.14), ("Kirkwall", 58.98, -2.96), ("Wick", 58.44, -3.09),
    ("Fraserburgh", 57.69, -2.00), ("Peterhead", 57.50, -1.78), ("Aberdeen", 57.15, -2.08),
    ("Montrose", 56.71, -2.47), ("St Andrews", 56.34, -2.80), ("Dunbar", 56.00, -2.52),
    ("Berwick", 55.77, -2.00), ("Tynemouth", 55.02, -1.42), ("Hartlepool", 54.69, -1.20),
    ("Whitby", 54.49, -0.61), ("Flamborough Head", 54.12, -0.08), ("Spurn Head", 53.58, 0.11),
    ("Skegness", 53.14, 0.34), ("Cromer", 52.93, 1.30), ("Great Yarmouth", 52.60, 1.73),
    ("Lowestoft", 52.48, 1.75), ("Harwich", 51.95, 1.29), ("Margate", 51.39, 1.38),
    ("Dover", 51.12, 1.32), ("Dunkirk", 51.05, 2.37), ("Ostend", 51.23, 2.92),
    ("Vlissingen", 51.44, 3.57), ("Scheveningen", 52.10, 4.27), ("IJmuiden", 52.46, 4.60),
    ("Den Helder", 52.96, 4.76), ("Terschelling", 53.40, 5.30), ("Borkum", 53.60, 6.70),
    ("Helgoland", 54.18, 7.89), ("Sylt", 54.90, 8.30), ("Esbjerg", 55.47, 8.45),
    ("Hvide Sande", 56.00, 8.13), ("Thyboron", 56.70, 8.22), ("Hanstholm", 57.12, 8.60),
    ("Hirtshals", 57.60, 9.96), ("Lista", 58.10, 6.60), ("Egersund", 58.45, 6.00),
    ("Stavanger", 58.97, 5.73), ("Bergen", 60.39, 5.32),
]


def _km(lat1, lon1, lat2, lon2):
    import math
    x = (lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2)) * 111.32
    y = (lat2 - lat1) * 110.57
    return math.hypot(x, y)


def name_for(lat, lon, within_km=14):
    """Best name for a shoal at lat/lon: a named bank, Dogger Bank, or 'off <town>'."""
    best = min(BANKS, key=lambda b: _km(lat, lon, b[1], b[2]))
    if _km(lat, lon, best[1], best[2]) <= within_km:
        return best[0]
    la0, la1, lo0, lo1 = DOGGER
    if la0 <= lat <= la1 and lo0 <= lon <= lo1:
        part = ("west" if lon < 2.0 else "central" if lon < 3.0 else "east (Tail End)")
        return f"Dogger Bank, {part}"
    t = min(TOWNS, key=lambda b: _km(lat, lon, b[1], b[2]))
    return f"Shoal {_km(lat, lon, t[1], t[2]):.0f} km off {t[0]}"
