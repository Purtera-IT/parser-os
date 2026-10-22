"""keep_heads.py <refit.pt> <base.pt> <out.pt> <key> [key ...]
Copy the named free heads back from the base checkpoint (refit regressed them)."""
import sys, torch
refit, base, out, keys = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
R = torch.load(refit, map_location="cpu", weights_only=False)
B = torch.load(base, map_location="cpu", weights_only=False)["state"]
for k in keys:
    for part in ("weight", "bias"):
        name = f"free_heads.{k}.{part}"
        R["state"][name] = B[name].clone()
        print("kept base", name)
torch.save(R, out)
print("wrote", out)
