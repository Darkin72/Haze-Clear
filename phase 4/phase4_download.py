"""Resolve existing datasets or download the external synthetic benchmarks."""
from pathlib import Path
import json


def pair_root(root):
    root = Path(root)
    dirs = {p.name.lower(): p for p in root.iterdir() if p.is_dir()} if root.is_dir() else {}
    clear = next((dirs[k] for k in ('clear', 'gt', 'clean', 'groundtruth', 'ground_truth') if k in dirs), None)
    hazy = next((dirs[k] for k in ('hazy', 'haze', 'input', 'inputs', 'hazy_images') if k in dirs), None)
    return clear is not None and hazy is not None and any(p.is_file() for p in clear.iterdir()) and any(p.is_file() for p in hazy.iterdir())


def find_dataset(base, name, test_only=False):
    base = Path(base).expanduser().resolve()
    candidates = [base, *sorted((p for p in base.rglob('*') if p.is_dir()), key=lambda p: (len(p.parts), str(p)))]
    if name.startswith('sots-'):
        domain = 'indoor' if name == 'sots-its' else 'outdoor'
        candidates = [p for p in candidates if domain in {x.lower() for x in p.parts}]
    if name == 'reside6k':
        if not test_only:
            for p in candidates:
                children = {d.name.lower() for d in p.iterdir() if d.is_dir()}
                if {'train', 'test'} <= children:
                    return p
        tests = [p for p in candidates if 'test' in {x.lower() for x in p.parts} and pair_root(p)]
        if tests:
            return tests[0]
        # Never silently evaluate a training directory when the test layout is missing.
        raise FileNotFoundError(f'Không tìm thấy RESIDE-6K test/clear và test/hazy dưới {base}.')
    for p in candidates:
        if pair_root(p):
            return p
    raise FileNotFoundError(f'Không tìm thấy cặp clear/hazy cho {name} dưới {base}.')


def resolve_datasets(names, specs, project_root, overrides, cache_root, *,
                     allow_download=True, test_only=False, download=None,
                     reside_handle='kmljts/reside-6k',
                     sots_handle='balraj98/synthetic-objective-testing-set-sots-reside',
                     continue_on_error=False):
    """Return actual roots plus errors; an explicit local override never triggers a download."""
    project_root, cache_root = Path(project_root), Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest = cache_root/'downloads.json'
    cached = json.loads(manifest.read_text(encoding='utf-8')) if manifest.is_file() else {}
    roots, errors, downloads = {}, {}, {}
    for name in names:
        try:
            explicit = overrides.get(name)
            local = Path(explicit).expanduser().resolve() if explicit else project_root/specs[name]['relative_root']
            if explicit:
                if not local.is_dir():
                    raise FileNotFoundError(f'Đường dẫn local của {name} không tồn tại: {local}')
                roots[name] = str(local)
                continue
            if name in {'o-hazy', 'i-haze'}:
                if not local.is_dir():
                    raise FileNotFoundError(f'{name} phải lấy từ checkout VDT_GD2: {local}')
                roots[name] = str(local.resolve())
                continue
            handle = reside_handle if name == 'reside6k' else sots_handle
            if local.is_dir():
                try:
                    roots[name] = str(find_dataset(local, name, test_only))
                    continue
                except FileNotFoundError:
                    pass
            if handle in cached and Path(cached[handle]).is_dir():
                try:
                    roots[name] = str(find_dataset(cached[handle], name, test_only))
                    continue
                except FileNotFoundError:
                    pass
            if not allow_download:
                raise FileNotFoundError(f'Thiếu {name}; bật AUTO_DOWNLOAD_DATASETS hoặc đặt DATA_ROOT_OVERRIDE(S).')
            if handle not in downloads:
                print(f'Đang tải/tái dùng cache KaggleHub: {handle}', flush=True)
                if download is None:
                    import kagglehub
                    downloads[handle] = Path(kagglehub.dataset_download(handle))
                else:
                    downloads[handle] = Path(download(handle))
                cached[handle] = str(downloads[handle].resolve())
                manifest.write_text(json.dumps(cached, indent=2), encoding='utf-8')
            roots[name] = str(find_dataset(downloads[handle], name, test_only))
        except Exception as exc:
            if not continue_on_error:
                raise
            errors[name] = str(exc)
            print(f'{name}: chuẩn bị dataset lỗi: {exc}', flush=True)
    return roots, errors


def resolve_training_roots(names, specs, project_root, train_roots, overrides, cache_root,
                           *, enabled=True, allow_download=True, download=None,
                           reside_handle='kmljts/reside-6k'):
    """Resolve separate training data; never substitute a SOTS benchmark directory."""
    roots = dict(train_roots)
    missing = [n for n in names if n.startswith('sots-') and not roots.get(n)]
    if not enabled or not missing:
        return roots
    resolved, _ = resolve_datasets(['reside6k'], specs, project_root, overrides, cache_root,
        allow_download=allow_download, test_only=False, download=download,
        reside_handle=reside_handle)
    base = Path(resolved['reside6k'])
    train = next((p for p in base.iterdir() if p.is_dir() and p.name.lower() == 'train'), None)
    if train is None or not pair_root(train):
        raise FileNotFoundError(f'Cần RESIDE-6K/train với các cặp GT/hazy, không dùng test: {base}')
    for name in missing:
        roots[name] = str(train.resolve())
    return roots
