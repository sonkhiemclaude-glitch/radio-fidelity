"""Parse the Antwerp LoRaWAN 2019 corpus into a compact, geometry-carrying form.

Source: Aernouts, Berkvens, Van Vlaenderen, Weyn -- "Sigfox and LoRaWAN Datasets
for Fingerprint Localization in Large Urban and Rural Areas", Zenodo 3904158
(CC-BY-4.0).  We read the *JSON* rendering rather than the CSV because only the
JSON carries the receiving gateway's identifier, and without the identifier the
gateway coordinates in the companion file cannot be joined -- i.e. without it
there is no geometry, and hence no physics, in the dataset at all.

Outputs `data/lorawan.npz`:
    dev        (M,)   device index
    t          (M,)   unix seconds
    lat, lon   (M,)   ground-truth position of the transmitter
    x, y       (M,)   the same position in local ENU metres
    sf, hdop   (M,)
    obs_msg    (N,)   message index of each gateway observation
    obs_gw     (N,)   gateway index
    obs_rssi   (N,)   RSSI in dBm
    obs_snr    (N,)   SNR in dB
    obs_esp    (N,)   estimated signal power in dBm
    gw_lat/lon (G,)   gateway coordinates
    gw_x/gw_y  (G,)   gateway coordinates in local ENU metres
"""
from __future__ import annotations
import json, os, re, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RAW = os.path.join(HERE, "..", "raw")
OUT = os.environ.get("REV_DATA_DIR", os.path.join(HERE, "..", "data"))

# The measurement campaign is confined to the city of Antwerp; the companion
# gateway file lists every gateway in the Belgian Proximus network, most of
# which are hundreds of kilometres away and never appear.
BBOX = (51.15, 51.28, 4.28, 4.52)          # lat_lo, lat_hi, lon_lo, lon_hi
HDOP_MAX = 1.0                              # GPS quality gate (median is 0.66)
MIN_RX = 200                                # drop gateways with too few receptions
R_EARTH = 6_371_000.0


def enu(lat, lon, lat0, lon0):
    """Equirectangular local tangent-plane projection, metres.  Over a 6 km
    city the distortion against a proper geodesic is below 0.1 m."""
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    x = np.deg2rad(lon - lon0) * R_EARTH * np.cos(np.deg2rad(lat0))
    y = np.deg2rad(lat - lat0) * R_EARTH
    return x, y


_TS = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?)"
                 r"(Z|[+-]\d{2}:\d{2})?$")


def _to_utc_ms(ts: str) -> np.int64:
    """Parse an ISO-8601 instant with an explicit offset into UTC milliseconds."""
    m = _TS.match(ts)
    if m is None:
        raise ValueError(ts)
    base, off = m.group(1), m.group(2)
    if "." in base:                      # numpy tolerates at most nanoseconds
        head, frac = base.split(".")
        base = head + "." + frac[:6]
    v = np.datetime64(base, "ms").astype("int64")
    if off and off != "Z":
        sign = 1 if off[0] == "+" else -1
        v -= sign * (int(off[1:3]) * 3600 + int(off[4:6]) * 60) * 1000
    return np.int64(v)


def main():
    t0 = time.time()
    gwfile = os.path.join(RAW, "lorawan_antwerp_gateway_locations.json.txt")
    msgfile = os.path.join(RAW, "lorawan_antwerp_2019_dataset.json.txt")
    gwloc = json.load(open(gwfile))
    print(f"gateway file: {len(gwloc)} gateways in the national network")

    print("loading messages ...", flush=True)
    msgs = json.load(open(msgfile))
    print(f"  {len(msgs)} messages, {time.time()-t0:.0f}s")

    # ---- pass 1: which gateways are actually used, and how often -----------
    from collections import Counter
    cnt = Counter()
    for m in msgs:
        for g in m["gateways"]:
            cnt[g["id"]] += 1
    usable = sorted([g for g, c in cnt.items() if c >= MIN_RX and g in gwloc])
    print(f"  gateways seen: {len(cnt)}; with coordinates and >={MIN_RX} rx: {len(usable)}")
    gidx = {g: i for i, g in enumerate(usable)}

    gw_lat = np.array([gwloc[g]["latitude"] for g in usable])
    gw_lon = np.array([gwloc[g]["longitude"] for g in usable])

    # ---- pass 2: materialise ---------------------------------------------
    devs: dict[str, int] = {}
    dev, T, LAT, LON, SF, HDOP = [], [], [], [], [], []
    o_msg, o_gw, o_rssi, o_snr, o_esp = [], [], [], [], []
    kept = 0
    for m in msgs:
        la, lo = m["latitude"], m["longitude"]
        if not (BBOX[0] <= la <= BBOX[1] and BBOX[2] <= lo <= BBOX[3]):
            continue
        hd = m.get("hdop")
        if hd is None or not (0 < hd <= HDOP_MAX):
            continue
        obs = [g for g in m["gateways"] if g["id"] in gidx]
        if not obs:
            continue
        # every gateway observation in the record carries the same rx timestamp
        # up to sub-second jitter; take the first as the message time
        ts = obs[0]["rx_time"]["time"]
        # timestamps carry an explicit UTC offset ("...+01:00"); strip it to UTC
        # by hand so numpy never has to guess a timezone
        try:
            tt = _to_utc_ms(ts)
        except Exception:
            continue
        d = m["dev_eui"]
        if d not in devs:
            devs[d] = len(devs)
        i = kept
        for g in obs:
            o_msg.append(i); o_gw.append(gidx[g["id"]])
            o_rssi.append(g["rssi"]); o_snr.append(g["snr"]); o_esp.append(g["esp"])
        dev.append(devs[d]); T.append(float(tt) / 1000.0)
        LAT.append(la); LON.append(lo); SF.append(m["sf"]); HDOP.append(hd)
        kept += 1

    lat = np.array(LAT); lon = np.array(LON)
    lat0, lon0 = float(lat.mean()), float(lon.mean())
    x, y = enu(lat, lon, lat0, lon0)
    gx, gy = enu(gw_lat, gw_lon, lat0, lon0)

    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(
        os.path.join(OUT, "lorawan.npz"),
        dev=np.array(dev, np.int32), t=np.array(T),
        lat=lat, lon=lon, x=x, y=y,
        sf=np.array(SF, np.int8), hdop=np.array(HDOP),
        obs_msg=np.array(o_msg, np.int32), obs_gw=np.array(o_gw, np.int16),
        obs_rssi=np.array(o_rssi, np.float32), obs_snr=np.array(o_snr, np.float32),
        obs_esp=np.array(o_esp, np.float32),
        gw_lat=gw_lat, gw_lon=gw_lon, gw_x=gx, gw_y=gy,
        gw_id=np.array(usable), lat0=lat0, lon0=lon0,
    )
    print(f"kept {kept} messages, {len(o_msg)} gateway observations, "
          f"{len(devs)} devices, {len(usable)} gateways")
    print(f"  extent x [{x.min():.0f},{x.max():.0f}] m, y [{y.min():.0f},{y.max():.0f}] m")
    print(f"  gateway extent x [{gx.min():.0f},{gx.max():.0f}] m, "
          f"y [{gy.min():.0f},{gy.max():.0f}] m")
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
