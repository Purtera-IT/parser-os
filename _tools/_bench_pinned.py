"""Worker-width benchmark pinned to N CPUs, so it reflects the 4-CPU container
rather than this 22-CPU workstation."""
import os, sys
import psutil
n = int(os.environ.get("PIN_CPUS", "4"))
p = psutil.Process()
p.cpu_affinity(list(range(n)))
print("pinned to %d CPUs: %s" % (n, p.cpu_affinity()))
sys.argv = [sys.argv[0]] + sys.argv[1:]
exec(open("_bench_workers.py", encoding="utf-8").read())
