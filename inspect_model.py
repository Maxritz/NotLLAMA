import json, sys

with open(sys.argv[1]) as f:
    j = json.load(f)

m = j.get('model', {})
print(f"Model: {m.get('architecture', '?')} blocks={m.get('block_count', '?')} dim={m.get('embedding_length', '?')}")
tensors = j.get('tensors', [])
print(f"Total tensors: {len(tensors)}")

# Largest by size_bytes
sorted_tensors = sorted(tensors, key=lambda x: x.get('size_bytes', 0), reverse=True)
for t in sorted_tensors[:5]:
    print(f"  {t['name']}: size_bytes={t['size_bytes']/1024/1024:.1f} MB  dtype={t['dtype']}")

dtypes = {}
for t in tensors:
    d = t.get('dtype', '?')
    dtypes[d] = dtypes.get(d, 0) + 1
print(f"\nDtype counts: {dtypes}")

# Largest by bin_size (raw binary on disk)
sorted_by_bin = sorted(tensors, key=lambda x: x.get('bin_size', 0), reverse=True)
print("\n--- Largest by bin_size ---")
for t in sorted_by_bin[:5]:
    print(f"  {t['name']}: bin_size={t['bin_size']/1024/1024:.1f} MB  dtype={t['dtype']}")

bin_total = sum(v.get('bin_size', 0) for v in tensors)
size_total = sum(v.get('size_bytes', 0) for v in tensors)
print(f"\nTotal bin_size: {bin_total/1024/1024/1024:.2f} GB")
print(f"Total size_bytes: {size_total/1024/1024/1024:.2f} GB")
