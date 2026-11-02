"""Download public revision inputs and record immutable SHA256 provenance."""
import concurrent.futures, hashlib, json, pathlib, urllib.request, zipfile, datetime
ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / 'raw'
URLS = {
 'lorawan_antwerp_gateway_locations.json.txt': 'https://zenodo.org/records/3904158/files/lorawan_antwerp_gateway_locations.json.txt?download=1',
 'lorawan_antwerp_2019_dataset.json.txt': 'https://zenodo.org/records/3904158/files/lorawan_antwerp_2019_dataset.json.txt?download=1',
 'uji.zip': 'https://archive.ics.uci.edu/static/public/310/ujiindoorloc.zip',
 'belgium-260801-free.shp.zip': 'https://download.geofabrik.de/europe/belgium-260801-free.shp.zip',
}
EXPECTED_SHA256 = {'lorawan_antwerp_gateway_locations.json.txt': '507f9bb266d23f59fdc2b743447cecf9c909290536e24de3f9483aaa178d374d', 'lorawan_antwerp_2019_dataset.json.txt': 'f2f1fbd478cdef2b76fb8e3aae33751d332b50fc8546684d8e73fdb1505451cb', 'uji.zip': '893512b82dfd7a7c345d84195b1c8019fbca0fa0d7820ce491ce5aa45ec3782f', 'belgium-260801-free.shp.zip': 'f18b5ce432c68a60717b2f4b6c0b54a45bbe517f92c79df66efc9b254a8eb0da'}

def download(item):
 name,url=item; dst=RAW/name
 if not dst.exists():
  tmp=dst.with_suffix(dst.suffix+'.part')
  urllib.request.urlretrieve(url,tmp); tmp.rename(dst)
 h=hashlib.sha256()
 with dst.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 if h.hexdigest()!=EXPECTED_SHA256[name]:raise ValueError(f'Checksum mismatch for {name}; refusing to replace the pinned provenance')
 print(name,dst.stat().st_size,flush=True)
 return dict(file=name,url=url,bytes=dst.stat().st_size,sha256=h.hexdigest())
if __name__=='__main__':
 RAW.mkdir(exist_ok=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: rows=list(pool.map(download,URLS.items()))
 (RAW/'manifest.json').write_text(json.dumps(dict(downloaded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=rows),indent=2))
 with zipfile.ZipFile(RAW/'uji.zip') as z:z.extractall(RAW/'uji')
