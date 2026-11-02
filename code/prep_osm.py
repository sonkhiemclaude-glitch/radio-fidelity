"""Turn the OpenStreetMap extract into the two geometric objects the simulator
needs: a building occupancy raster, and a set of reference points on the road
network.

Why a raster.  The fidelity rungs that matter for an urban radio simulator are
"how many buildings does the line of sight cross" and "how many metres of
building material does it cross".  Both are integrals along a segment, so a
regular occupancy raster is the natural representation: sampling the segment at
the raster pitch turns each into a sum, and the whole link set can then be
evaluated with array indexing instead of polygon clipping.

Why road points.  The simulated training set has to live somewhere.  Using the
measurement positions themselves would leak the real covariate distribution
into the simulator; a deployment that has not yet measured anything would place
reference points on the streets.  We therefore sample the OSM road network at a
fixed pitch, which is both realistic and independent of the measurement
campaign.

Source: Geofabrik OSM extract of Belgium (ODbL), layers
`gis_osm_buildings_a_free_1` and `gis_osm_roads_free_1`.
"""
from __future__ import annotations
import os, sys, time, zipfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RAW = os.path.join(HERE, "..", "raw")
OUT = os.environ.get("REV_DATA_DIR", os.path.join(HERE, "..", "data"))
ZIP = os.path.join(RAW, "belgium-260801-free.shp.zip")
EXT = os.path.join(RAW, "osm_be")

sys.path.insert(0, HERE)
import shpfast

RES = 5.0            # raster pitch, metres
PAD = 600.0          # padding around the area of interest, metres
MAX_GW_KM = 15.0     # gateways further away than this are excluded as outliers
ROAD_PITCH = 50.0    # spacing of reference points along the road network

DRIVABLE = {
    "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
    "residential", "living_street", "motorway_link", "trunk_link",
    "primary_link", "secondary_link", "tertiary_link", "service",
}

# buildings need geometry only; roads need one attribute column as well
WANT = {"gis_osm_buildings_a_free_1": (".shp", ".shx"),
        "gis_osm_roads_free_1": (".shp", ".shx", ".dbf")}


def extract_layers():
    os.makedirs(EXT, exist_ok=True)
    with zipfile.ZipFile(ZIP) as z:
        for n in z.namelist():
            b = os.path.basename(n)
            for stem, exts in WANT.items():
                if b.startswith(stem) and b.endswith(exts):
                    dst = os.path.join(EXT, b)
                    if os.path.exists(dst) and os.path.getsize(dst) == z.getinfo(n).file_size:
                        continue
                    print(f"  extracting {b} "
                          f"({z.getinfo(n).file_size/1e6:.0f} MB) ...", flush=True)
                    with z.open(n) as src, open(dst, "wb") as out:
                        while True:
                            c = src.read(1 << 22)
                            if not c:
                                break
                            out.write(c)


def enu(lat, lon, lat0, lon0):
    R = 6_371_000.0
    x = np.deg2rad(np.asarray(lon, float) - lon0) * R * np.cos(np.deg2rad(lat0))
    y = np.deg2rad(np.asarray(lat, float) - lat0) * R
    return x, y


def main():
    t0 = time.time()
    d = np.load(os.path.join(OUT, "lorawan.npz"), allow_pickle=True)
    lat0, lon0 = float(d["lat0"]), float(d["lon0"])
    gx, gy = d["gw_x"], d["gw_y"]
    keep_gw = np.hypot(gx, gy) <= MAX_GW_KM * 1000
    px, py = d["x"], d["y"]

    xs = np.concatenate([px, gx[keep_gw]]); ys = np.concatenate([py, gy[keep_gw]])
    x_lo, x_hi = xs.min() - PAD, xs.max() + PAD
    y_lo, y_hi = ys.min() - PAD, ys.max() + PAD
    nx = int(np.ceil((x_hi - x_lo) / RES)); ny = int(np.ceil((y_hi - y_lo) / RES))
    print(f"AOI {x_hi-x_lo:.0f} x {y_hi-y_lo:.0f} m -> raster {nx} x {ny} "
          f"= {nx*ny/1e6:.1f} Mcells at {RES:.0f} m")

    R = 6_371_000.0
    lat_lo = lat0 + np.rad2deg(y_lo / R); lat_hi = lat0 + np.rad2deg(y_hi / R)
    lon_lo = lon0 + np.rad2deg(x_lo / (R * np.cos(np.deg2rad(lat0))))
    lon_hi = lon0 + np.rad2deg(x_hi / (R * np.cos(np.deg2rad(lat0))))
    print(f"  lat [{lat_lo:.4f},{lat_hi:.4f}] lon [{lon_lo:.4f},{lon_hi:.4f}]")

    extract_layers()
    from matplotlib.path import Path

    # ---------------- buildings -> occupancy raster ------------------------
    stem = os.path.join(EXT, "gis_osm_buildings_a_free_1")
    off = shpfast.read_shx(stem + ".shx")
    print(f"buildings: {len(off)} polygons nationally ({time.time()-t0:.0f}s)",
          flush=True)
    bb = shpfast.bboxes(stem + ".shp", off)
    sel = np.flatnonzero((bb[:, 2] >= lon_lo) & (bb[:, 0] <= lon_hi)
                         & (bb[:, 3] >= lat_lo) & (bb[:, 1] <= lat_hi))
    print(f"  {len(sel)} inside the AOI bounding box ({time.time()-t0:.0f}s)",
          flush=True)
    sh = shpfast.Shapes(stem + ".shp", off)
    occ = np.zeros((ny, nx), dtype=bool)
    for c, i in enumerate(sel):
        parts, pts = sh.parts_points(i)
        bx, by = enu(pts[:, 1], pts[:, 0], lat0, lon0)
        bounds = list(parts) + [len(pts)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            ring = np.c_[bx[a:b], by[a:b]]
            if len(ring) < 3:
                continue
            i0 = max(0, int(np.floor((ring[:, 0].min() - x_lo) / RES)))
            i1 = min(nx - 1, int(np.ceil((ring[:, 0].max() - x_lo) / RES)))
            j0 = max(0, int(np.floor((ring[:, 1].min() - y_lo) / RES)))
            j1 = min(ny - 1, int(np.ceil((ring[:, 1].max() - y_lo) / RES)))
            if i1 < i0 or j1 < j0:
                continue
            ii, jj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1))
            cx = x_lo + (ii + 0.5) * RES; cy = y_lo + (jj + 0.5) * RES
            inside = Path(ring).contains_points(np.c_[cx.ravel(), cy.ravel()])
            if inside.any():
                occ[jj.ravel()[inside], ii.ravel()[inside]] = True
        if (c + 1) % 100_000 == 0:
            print(f"    {c+1}/{len(sel)} rasterised, occupancy "
                  f"{occ.mean()*100:.2f}%  ({time.time()-t0:.0f}s)", flush=True)
    n_b = len(sel)
    print(f"  {n_b} polygons rasterised; occupancy {occ.mean()*100:.2f}% "
          f"({time.time()-t0:.0f}s)")

    # ---------------- roads -> reference points ----------------------------
    stem = os.path.join(EXT, "gis_osm_roads_free_1")
    roff = shpfast.read_shx(stem + ".shx")
    rbb = shpfast.bboxes(stem + ".shp", roff)
    rx_lo, rx_hi = px.min() - 200, px.max() + 200
    ry_lo, ry_hi = py.min() - 200, py.max() + 200
    rlat_lo = lat0 + np.rad2deg(ry_lo / R); rlat_hi = lat0 + np.rad2deg(ry_hi / R)
    rlon_lo = lon0 + np.rad2deg(rx_lo / (R * np.cos(np.deg2rad(lat0))))
    rlon_hi = lon0 + np.rad2deg(rx_hi / (R * np.cos(np.deg2rad(lat0))))
    rsel = np.flatnonzero((rbb[:, 2] >= rlon_lo) & (rbb[:, 0] <= rlon_hi)
                          & (rbb[:, 3] >= rlat_lo) & (rbb[:, 1] <= rlat_hi))
    print(f"roads: {len(roff)} nationally, {len(rsel)} in the campaign area")
    fclass = shpfast.dbf_field(stem + ".dbf", "fclass", rsel)
    rsh = shpfast.Shapes(stem + ".shp", roff)
    pts_x, pts_y = [], []
    n_drivable = 0
    for i, fc in zip(rsel, fclass):
        if fc not in DRIVABLE:
            continue
        n_drivable += 1
        _, p = rsh.parts_points(i)
        if len(p) < 2:
            continue
        ax, ay = enu(p[:, 1], p[:, 0], lat0, lon0)
        seg = np.hypot(np.diff(ax), np.diff(ay))
        s = np.concatenate([[0.0], np.cumsum(seg)])
        if s[-1] < 1e-6:
            continue
        q = np.arange(0.0, s[-1], ROAD_PITCH)
        pts_x.append(np.interp(q, s, ax)); pts_y.append(np.interp(q, s, ay))
    rxs = np.concatenate(pts_x); rys = np.concatenate(pts_y)
    m = (rxs >= rx_lo) & (rxs <= rx_hi) & (rys >= ry_lo) & (rys <= ry_hi)
    rxs, rys = rxs[m], rys[m]
    key = (np.round(rxs / ROAD_PITCH).astype(np.int64) * 1_000_000
           + np.round(rys / ROAD_PITCH).astype(np.int64))
    _, first = np.unique(key, return_index=True)
    rxs, rys = rxs[first], rys[first]
    print(f"  {n_drivable} drivable ways -> {len(rxs)} reference points "
          f"at {ROAD_PITCH:.0f} m pitch")

    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "geometry.npz"),
                        occ=np.packbits(occ, axis=None),
                        occ_shape=np.array([ny, nx]),
                        x_lo=x_lo, y_lo=y_lo, res=RES,
                        road_x=rxs, road_y=rys, n_buildings=n_b,
                        keep_gw=keep_gw)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
