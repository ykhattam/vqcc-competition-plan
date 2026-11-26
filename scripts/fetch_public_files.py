import csv, io, os, subprocess, sys

DATASET = "manwithacat/vqcc-apache-public-circuits"
DEST = os.path.expanduser("~/vqcc/public_circuits")
os.makedirs(DEST, exist_ok=True)

token = None
names = []
while True:
    cmd = ["kaggle", "datasets", "files", "-d", DATASET, "--page-size", "200", "--format", "csv"]
    if token:
        cmd += ["--page-token", token]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    lines = [ln for ln in out.splitlines() if ln.strip()]
    next_token = None
    body = []
    for ln in lines:
        if ln.startswith("Next Page Token"):
            next_token = ln.split("=", 1)[1].strip()
        else:
            body.append(ln)
    rows = list(csv.DictReader(io.StringIO("\n".join(body))))
    names += [r["name"] for r in rows if r.get("name")]
    print(f"page: {len(rows)} files", flush=True)
    if not next_token or not rows:
        break
    token = next_token

print(f"total files: {len(names)}", flush=True)
for i, name in enumerate(names, 1):
    target = os.path.join(DEST, name)
    if os.path.exists(target):
        continue
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-f", name,
         "-p", os.path.dirname(target), "-o", "-q"],
        check=True,
    )
    if i % 25 == 0:
        print(f"downloaded {i}/{len(names)}", flush=True)
print("done", flush=True)
