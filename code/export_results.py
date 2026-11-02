"""Export every saved result field to CSV without rounding or changing values.

Each JSON file gets a path/value CSV containing every scalar leaf, and each
top-level list of records gets a rectangular table. Nested values in record
tables remain JSON strings. Original JSON remains the authoritative source.
"""
import csv, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def leaves(value, path='$'):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from leaves(item, path+'.'+key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from leaves(item, f'{path}[{index}]')
    else:
        yield path, value

def main():
    count=0
    for source in sorted((ROOT/'results').rglob('*.json')):
        if 'intermediate' in source.parts:
            continue
        value=json.loads(source.read_text())
        dest=ROOT/'reports/tables'/source.relative_to(ROOT/'results').with_suffix('')
        dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.with_suffix('.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['json_path','value'])
            writer.writerows(leaves(value))
        count+=1
        if not isinstance(value,dict):
            continue
        for name, records in value.items():
            if not isinstance(records,list) or not records or not all(isinstance(r,dict) for r in records):
                continue
            fields=list(dict.fromkeys(k for record in records for k in record))
            with dest.with_name(dest.name+'__'+name+'.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
                for record in records:
                    writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in record.items()})
            count+=1
    print('Exported',count,'CSV result tables')

if __name__=='__main__':
    main()
