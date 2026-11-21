"""One crash-isolated growth run. All control inputs and outputs are JSON files."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time
import traceback


def execute(spec_file):
    spec=json.loads(Path(spec_file).read_text())
    started=time.monotonic()
    result_path=Path(spec['result_file'])
    try:
        from corner_growth_controller import GrowthPolicy, run
        state=run(spec['trial'],spec['seed_directory'],spec['output'],GrowthPolicy(**spec['policy']))
        result=dict(outcome='completed',state=state)
    except BaseException as exc:
        result=dict(outcome='crashed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        print(result['traceback'],file=sys.stderr,flush=True)
    usage=resource.getrusage(resource.RUSAGE_SELF)
    children=resource.getrusage(resource.RUSAGE_CHILDREN)
    result.update(elapsed_seconds=time.monotonic()-started,
        cpu_user_seconds=usage.ru_utime+children.ru_utime,
        cpu_system_seconds=usage.ru_stime+children.ru_stime,
        peak_rss_platform_units=max(usage.ru_maxrss,children.ru_maxrss))
    # Result publication does not depend on successful numerical imports.
    temporary=result_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,indent=2)+'\n')
    temporary.replace(result_path)
    if result['outcome']!='completed':raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',required=True)
    execute(parser.parse_args().spec)
