"""Shared checkpoint bootstrap. Does not modify model/training sources or artifacts."""
from pathlib import Path
import json
import subprocess
import sys
import zipfile


WEIGHT_NAMES = {'o-hazy': 'Ohaze_1200_1600_27.22.pth',
                'reside6k': 'reside6k_32.15.pth',
                'sots-its': 'reside6k_32.15.pth', 'sots-ots': 'reside6k_32.15.pth'}


def download_archive(target):
    try:
        import gdown
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gdown'], check=True)
        import gdown
    result = gdown.download(id='1Vm-1mtMSc9YIUICq4nWKu-4AHCtYnPpF', output=str(target),
                            quiet=False, use_cookies=False, resume=True)
    if result is None:
        raise RuntimeError('Google Drive không trả về ZIP trọng số. Kiểm tra quota hoặc đặt CHECKPOINT_ZIP local.')


def find_weight(roots, filename):
    for root in roots:
        root = Path(root).expanduser()
        if root.is_dir():
            found = sorted(root.rglob(filename))
            found = [p for p in found if p.is_file() and p.stat().st_size]
            if found:
                return found[0].resolve()
    return None


def ensure_shared_weights(configs, cache_root, download=download_archive):
    """Populate shared cache once; keep failures per dataset and leave explicit paths alone."""
    cache = Path(cache_root).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    missing, errors = [], {}
    for cfg in configs:
        name = cfg['dataset_name']
        if name not in WEIGHT_NAMES:
            continue  # I-HAZE never needs pretrained weights.
        roots = cfg.setdefault('checkpoint_search_roots', [])
        if str(cache) not in roots:
            roots.append(str(cache))
        if cfg.get('repo_root') and cfg['repo_root'] not in roots:
            roots.append(cfg['repo_root'])
        if cfg.get('checkpoint') or (name != 'o-hazy' and cfg.get('reside_checkpoint')):
            continue  # Explicit path validation remains in the runner.
        search = [cfg.get('project_root', ''), cfg.get('repo_root', ''),
                  Path(cfg['work_root'])/'weights', *roots]
        if not find_weight([p for p in search if p], WEIGHT_NAMES[name]):
            missing.append(cfg)
    if not missing:
        return errors
    # A supplied ZIP works even when network downloads are disabled.
    provided = next((c['checkpoint_zip'] for c in missing if c.get('checkpoint_zip')), '')
    downloadable = [c for c in missing if c.get('allow_weight_download')]
    if not provided and not downloadable:
        return {c['dataset_name']: 'Thiếu checkpoint và ALLOW_WEIGHT_DOWNLOAD=False.' for c in missing}
    archive = Path(provided).expanduser() if provided else cache/'WDMamba_ckpts.zip'
    try:
        if not provided and not zipfile.is_zipfile(archive):
            partial = archive.with_suffix('.zip.partial')
            print('Đang tải bộ checkpoint dùng chung cho SOTS và O-HAZY...', flush=True)
            download(partial)
            if not zipfile.is_zipfile(partial):
                raise ValueError('File tải về không phải ZIP trọng số hợp lệ.')
            partial.replace(archive)
        target = cache/'unzipped'; target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:
                raise ValueError('ZIP trọng số bị lỗi CRC.')
            for item in z.infolist():
                if not (target/item.filename).resolve().is_relative_to(target):
                    raise ValueError('ZIP chứa đường dẫn ngoài thư mục giải nén.')
            z.extractall(target)
        for cfg in missing:
            name = cfg['dataset_name']
            if not find_weight([cache], WEIGHT_NAMES[name]):
                errors[name] = f'ZIP không chứa {WEIGHT_NAMES[name]}.'
    except Exception as exc:
        errors = {c['dataset_name']: str(exc) for c in missing}
    for name, message in errors.items():
        print(f'{name}: chuẩn bị checkpoint lỗi: {message}', flush=True)
    return errors


if __name__ == '__main__':
    path = Path(sys.argv[1])
    bundle = json.loads(path.read_text(encoding='utf-8'))
    bundle['errors'] = ensure_shared_weights(bundle['configs'], bundle['cache_root'])
    path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding='utf-8')
