"""Step 9.7A: analyze an existing frozen localization results directory without NCLT or a rerun."""
from __future__ import annotations
import argparse
from pathlib import Path
from benchmark.orientation_diagnostics import analyze


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',required=True,type=Path,help='Completed Step 9.6 or unified localization results directory')
    p.add_argument('--output',default='results/step9_7a_orientation',type=Path)
    a=p.parse_args()
    rows=analyze(a.results,a.output)
    for row in rows:
        if row['query_session']=='ALL':
            print(f"{row['map_geometry']}: fine {row['final_fine']}/{row['queries']}, "
                  f"XY {row['final_xy']}/{row['queries']}, yaw {row['final_yaw']}/{row['queries']}, "
                  f"yaw-only fails {row['yaw_only_failures']}, "
                  f"map losses/rescues {row['front_fine_lost']}/{row['front_not_fine_recovered']}")
    print(f'Wrote diagnostic reports: {a.output}')


if __name__=='__main__':
    main()
