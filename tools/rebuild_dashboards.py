"""Regenerate offline dashboards from existing logs; no training, GIF re-encoding or FEM solve."""
from pathlib import Path
import argparse,subprocess,sys
from restore_results import restore

ROOT=Path(__file__).resolve().parents[1]
SUITES={
 'heat1d':('pinn_optimizer_lab','demo'),
 'heat_contact2d':('pinn_2d_optimizer_lab','baseline'),
 'hertz':('hertz_cylinder_lab','kkt_all11'),
 'dcgd_config':('hertz_cylinder_lab','dcgd_config_control'),
 'scheduler':('pinn_scheduler_lab','scheduler60'),
}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--suite',choices=list(SUITES)+['all'],default='all');args=parser.parse_args()
    print('Restore/verify aggregates:',restore(),flush=True)
    for key in SUITES if args.suite=='all' else [args.suite]:
        project,suite=SUITES[key];folder=ROOT/project
        if project=='pinn_scheduler_lab':
            command=[sys.executable,'-B',str(folder/'report.py'),'--suite',suite,'--skip-media']
        else:command=[sys.executable,'-B',str(folder/'report.py'),str(folder/'runs'/suite/'results.json'),'--no-gif']
        subprocess.run(command,cwd=folder,check=True)
        print('Open locally:',folder/'runs'/suite/'dashboard.html',flush=True)


if __name__=='__main__':main()
