"""claude_run helper: animate two recorded rollouts of the SAME reference clip side by side.
Each run_frames process writes its own rotated motion copy, whose digest differs between runs, so
render_rollouts' matched-motion check rejects them. This copies both reports with a shared digest.
Use only for pairs that rotate the same clip by the same heading."""
import json, sys, tempfile
from pathlib import Path
from course.render_rollouts import render

a, b, la, lb, asset, out = sys.argv[1:7]
tmp = Path(tempfile.mkdtemp())
paths = []
for i, p in enumerate((a, b)):
    d = json.loads(Path(p).read_text())
    d["motion_sha256"] = "shared-for-rendering"
    q = tmp / f"{i}.json"; q.write_text(json.dumps(d)); paths.append(str(q))
render(paths, [la, lb], asset, out)
print("wrote", out)
