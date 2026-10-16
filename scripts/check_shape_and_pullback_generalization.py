import json, os, sys, pathlib
root = pathlib.Path(r"c:\Users\x\Documents\antigravity\darksnail")
sys.path.insert(0, str(root))
from darksnail.shell import infer_arc_output_shape_beltrami, extract_beltrami_pullback_canvas

base = root.parent / "brain-ai" / "data" / "arc_extracted" / "ARC-AGI-master" / "data"
out = {}
for split in ("training", "evaluation"):
    d = base / split
    tot = ok = 0
    pb_fire = pb_exact_test = 0
    changing = 0
    for fn in sorted(os.listdir(d)):
        t = json.load(open(d / fn))
        demos = t["train"]
        same = all(len(x["input"]) == len(x["output"]) and len(x["input"][0]) == len(x["output"][0]) for x in demos)
        for p in t["test"]:
            tot += 1
            gt = (len(p["output"]), len(p["output"][0]))
            if tuple(infer_arc_output_shape_beltrami(demos, p["input"])) == gt:
                ok += 1
            if not same:
                changing += 1
                pb = extract_beltrami_pullback_canvas(demos, test_input=p["input"], target_shape=gt)
                if (not pb["is_same_size"]) and pb["demo_exact_match"]:
                    pb_fire += 1
                    if pb["test_grid_pullback"] == p["output"]:
                        pb_exact_test += 1
    out[split] = dict(test_pairs=tot, shape_exact=ok, shape_acc_pct=round(100 * ok / tot, 2),
                      shape_changing_pairs=changing, generic_pullback_demo_exact=pb_fire,
                      generic_pullback_test_exact=pb_exact_test)
    print(split, out[split], flush=True)
json.dump(out, open(root / "results" / "shape_and_pullback_generalization.json", "w"), indent=2)
