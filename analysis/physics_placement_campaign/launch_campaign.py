"""Single-controller entry point for preparation, validation, and large runs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import campaign
import pilot


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--out',type=Path,default=campaign.DEFAULT)
    args,_=parser.parse_known_args()
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/'.campaign.lock').open('a+') as lock:
        try:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('This campaign already has a controller running; use its checkpoints after it finishes.',file=sys.stderr)
            return 2
        record={'pid':os.getpid(),'start_unix':time.time(),'argv':sys.argv[1:],
                'launcher_sha256':pilot.digest(__file__)}
        lock.seek(0);lock.truncate();lock.write(json.dumps(record));lock.flush()
        launches=args.out/'launches';launches.mkdir(exist_ok=True)
        pilot.write(launches/f'{time.time_ns()}.json',record)
        campaign.main()
    return 0

if __name__=='__main__':sys.exit(main())
