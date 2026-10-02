import subprocess, sys
for sid in sys.argv[1:]:
    print("=" * 20, sid, flush=True)
    for cmd in (["python", "-u", "fetch_landsat.py", sid, "swir22,nir08,lwir11"], ["python", "-u", "-W", "ignore", "screen_scene.py", sid]):
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout[-1800:], r.stderr[-400:] if r.returncode else "", flush=True)
print("ALL DONE", flush=True)
