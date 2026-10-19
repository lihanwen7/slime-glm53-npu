#!/usr/bin/env python3
"""Check or apply the exact dependency patch sequence without resetting user trees."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def git(root, *args):
    result = subprocess.run(['git', '-C', str(root), *args], text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_matches(root, item):
    files = item.get('expected_files', {})
    return bool(files) and all((root / rel).is_file() and digest(root / rel) == value
                               for rel, value in files.items())


def apply_and_verify(root, source_root, item):
    for entry in item['patches']:
        patch = source_root / entry['file']
        if digest(patch) != entry['sha256']:
            raise RuntimeError(f'Patch checksum mismatch: {patch}')
        git(root, 'apply', '--check', str(patch))
        git(root, 'apply', str(patch))
    for rel, value in item.get('expected_files', {}).items():
        if digest(root / rel) != value:
            raise RuntimeError(f'Reconstructed file differs from source snapshot: {root / rel}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True,
                        help='Parent directory containing slime, sglang, Megatron-LM, etc.')
    parser.add_argument('--dependency', action='append', help='Limit to these dependency names.')
    parser.add_argument('--clone', action='store_true', help='Clone missing repositories at pinned commits.')
    parser.add_argument('--apply', action='store_true', help='Apply patches to clean matching working trees.')
    args = parser.parse_args()
    source_root = Path(__file__).resolve().parents[1]
    manifest = json.loads((source_root / 'patches/sources.lock.json').read_text())
    available = {item['name'] for item in manifest['repositories']}
    if args.dependency and not set(args.dependency) <= available:
        parser.error('Unknown dependency: ' + ', '.join(set(args.dependency) - available))
    for item in manifest['repositories']:
        if args.dependency and item['name'] not in args.dependency:
            continue
        target = args.workspace.resolve() / item['directory']
        if not target.exists():
            if not args.clone:
                raise RuntimeError(f'Missing {target}; use --clone or clone it first.')
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['git', 'clone', '--no-checkout', item['url'], str(target)], check=True)
            try:
                git(target, 'cat-file', '-e', item['commit'] + '^{commit}')
            except RuntimeError:
                git(target, 'fetch', 'origin', item['commit'])
            git(target, 'checkout', '--detach', item['commit'])
        head = git(target, 'rev-parse', 'HEAD')
        if head != item['commit']:
            raise RuntimeError(f'{target}: expected HEAD {item["commit"]}, got {head}; no reset performed.')
        if expected_matches(target, item):
            print(f'{item["name"]}: already matches the patched snapshot')
            continue
        status = git(target, 'status', '--porcelain', '--untracked-files=normal')
        if status:
            raise RuntimeError(f'{target}: working tree is not clean; no reset performed.\n{status}')
        if args.apply:
            apply_and_verify(target, source_root, item)
            print(f'{item["name"]}: applied {len(item["patches"])} patches; hashes match')
        else:
            with tempfile.TemporaryDirectory(prefix='slime-patch-check-') as temporary:
                check = Path(temporary) / 'checkout'
                subprocess.run(['git', 'clone', '--quiet', '--shared', '--no-checkout',
                                str(target), str(check)], check=True)
                git(check, 'checkout', '--quiet', '--detach', item['commit'])
                apply_and_verify(check, source_root, item)
            print(f'{item["name"]}: replay check passed; target working tree unchanged')


if __name__ == '__main__':
    main()
