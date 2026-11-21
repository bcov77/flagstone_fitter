#!/usr/bin/env python3
"""Build a source-only handoff ZIP without private inputs or run artifacts."""
import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'elliot_patio_source/'


def source_files():
    paths = set()
    for pattern in (
        'src/flagstone/*.py', 'tests/*.py', 'scripts/*.py', 'scripts/*.sh',
        'scripts/*.java', 'scripts/*.html', 'physics_patio/patio_physics/*.py',
        'physics_patio/patio_physics/*.html', 'physics_patio/patio_physics/static/*.html',
        'physics_patio/patio_physics/static/*.css', 'physics_patio/patio_physics/static/*.js',
        'physics_patio/tests/*.py', 'physics_patio/*.sh', 'analysis/*/*.py',
    ):
        paths.update(p for p in ROOT.glob(pattern) if p.is_file() and not p.is_symlink())
    for name in (
        'pyproject.toml', 'uv.lock', 'PROJECT_LAYOUT.md',
        'physics_patio/requirements.txt', 'physics_patio/README.md',
        'analysis/aggressive_trajectories/config.json',
        'analysis/allocation_campaigns/README.md', 'analysis/full_trajectories/README.md',
        'analysis/physics_trajectories/README.md', 'analysis/physics_trajectories/requirements.txt',
    ):
        path = ROOT/name
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f'Required source missing: {name}')
        paths.add(path)
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'exports/elliot_patio_source.zip')
    args = parser.parse_args()
    paths = source_files()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, PREFIX+str(path.relative_to(ROOT)))
        archive.writestr(PREFIX+'README.md', (ROOT/'PROJECT_LAYOUT.md').read_text())
        archive.writestr(PREFIX+'.gitignore', '''.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.DS_Store
examples/
exports/
data/
real_world_pics/
generate_flagstones/
generate_patio_outline/
physics_patio/sessions/
*.json
*.json.gz
!analysis/aggressive_trajectories/config.json
*.png
*.jpg
*.jpeg
*.pdf
*.zip
*.log
''')
    print(f'{args.out.resolve()} ({len(paths)+2} files, {args.out.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
