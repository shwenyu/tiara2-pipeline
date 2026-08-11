#!/usr/bin/env python3
"""Extract non-deployed head/length calibration candidates from M1-M5 audit."""
import argparse,csv,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument("--audit",required=True);p.add_argument("--out",required=True);a=p.parse_args();m=json.load(open(a.audit));rows=[]
 for cell,v in sorted(m["cells"].items()):
  head,bin_name=cell.split("/");thresholds=v["one_vs_rest_thresholds"]
  rows.append({"head":head,"length_bin":bin_name,"N":v["N"],"T_star":v["temperature"]["T_star"],"empirical_thresholds_json":json.dumps({k:x["empirical_threshold"] for k,x in thresholds.items()},sort_keys=True),"f1_half_diagnostic_json":json.dumps({k:x["f1_half"] for k,x in thresholds.items()},sort_keys=True),"status":"validation_candidate_not_deployed"})
 out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
 with out.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=rows[0].keys(),delimiter="\t");w.writeheader();w.writerows(rows)
 print(json.dumps({"out":str(out),"rows":len(rows),"status":"not_deployed"},indent=2))
if __name__=="__main__":main()
