"""Paired-data and metric utilities for Phase 3.2; no pretrained weights."""
from pathlib import Path
import csv
import hashlib
import json
import random
import re

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity

EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


DATASET_SPECS = {'o-hazy': {'label': 'O-HAZY',
            'relative_root': 'dataset/O-HAZY',
            'kind': 'paired',
            'can_train': True,
            'val_ids': [5, 15, 25, 35]},
 'i-haze': {'label': 'I-HAZE',
            'relative_root': 'dataset/I-HAZE',
            'kind': 'paired',
            'can_train': True,
            'val_ids': []},
 'reside6k': {'label': 'RESIDE-6K',
              'relative_root': 'dataset/RESIDE-6K',
              'kind': 'paired',
              'can_train': True,
              'val_ids': []},
 'sots-its': {'label': 'SOTS Indoor / ITS',
              'relative_root': 'dataset/Synthetic Objective Testing Set (SOTS) [RESIDE]/indoor',
              'kind': 'evaluation-only',
              'can_train': False,
              'val_ids': []},
 'sots-ots': {'label': 'SOTS Outdoor / OTS',
              'relative_root': 'dataset/Synthetic Objective Testing Set (SOTS) [RESIDE]/outdoor',
              'kind': 'evaluation-only',
              'can_train': False,
              'val_ids': []}}


def normalize_dataset_name(name):
    key = str(name or 'o-hazy').strip().lower().replace('_', '-').replace('/', '-').replace(' ', '-')
    key = re.sub(r'-+', '-', key).strip('-')
    aliases = {'o-haze': 'o-hazy', 'ohazy': 'o-hazy', 'ihaze': 'i-haze', 'sots-indoor': 'sots-its', 'sots-indoor-its': 'sots-its',
               'its': 'sots-its', 'sots-outdoor': 'sots-ots', 'sots-outdoor-ots': 'sots-ots', 'ots': 'sots-ots', 'reside-6k': 'reside6k'}
    key = aliases.get(key, key)
    if key not in DATASET_SPECS:
        raise ValueError(f'DATASET_NAME không hợp lệ: {name}. Chọn một trong {list(DATASET_SPECS)}')
    return key


def _kind(path, root):
    parents = {a.lower() for a in Path(path).relative_to(root).parts[:-1]}
    stem = Path(path).stem.lower()
    if parents & {'gt', 'clear', 'clean', 'groundtruth', 'ground_truth'} or stem.endswith(('_gt', '_clear', '_clean')):
        return 'clear'
    if parents & {'hazy', 'haze', 'input', 'inputs', 'hazy_images'} or stem.endswith(('_hazy', '_haze', '_input')):
        return 'hazy'
    return None


def _base_stem(path):
    stem = Path(path).stem.lower()
    stem = re.sub(r'(?i)([_-]?hazy|[_-]?haze|[_-]?gt|[_-]?clear|[_-]?clean)$', '', stem)
    return re.sub(r'\d+', lambda m: str(int(m[0])), stem)


def collect_pairs(root, dataset_name='o-hazy'):
    """Pair clear/hazy images across explicit split folders and common suffix conventions."""
    dataset_name = normalize_dataset_name(dataset_name)
    root = Path(root).resolve()
    groups = {}
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.suffix.lower() not in EXT:
            continue
        kind = _kind(p, root)
        if kind is None:
            continue
        rel_parts = p.relative_to(root).parts
        split = next((x.lower() for x in rel_parts if x.lower() in {'train', 'val', 'validation', 'test'}), None)
        if split == 'validation': split = 'val'
        groups.setdefault(split or 'all', {'clear': [], 'hazy': []})[kind].append(p.resolve())
    pairs = []
    for split, sides in sorted(groups.items()):
        exact, normalized = {}, {}
        for clear in sides['clear']:
            for table, key in ((exact, clear.stem.lower()), (normalized, _base_stem(clear))):
                if key in table:
                    raise ValueError(f'Trùng ảnh clear trong split {split}: {table[key]} và {clear}')
                table[key] = clear
        used, ids = set(), set()
        for hazy in sides['hazy']:
            clear = exact.get(hazy.stem.lower()) or normalized.get(_base_stem(hazy))
            # RESIDE: one clear scene can have several atmospheric variants.
            if clear is None and dataset_name in {'reside6k', 'sots-its', 'sots-ots'}:
                clear = normalized.get(_base_stem(Path(hazy.stem.split('_')[0])))
            if clear is None:
                raise ValueError(f'Không ghép được ảnh hazy với clear trong split {split}: {hazy}')
            sid = hazy.stem.lower()
            if sid in ids:
                raise ValueError(f'Trùng ID hazy trong split {split}: {sid}')
            ids.add(sid); used.add(clear)
            pairs.append({'id': sid, 'scene': _base_stem(clear), 'hazy': str(hazy), 'clear': str(clear),
                          'source_split': None if split == 'all' else split})
        if set(sides['clear']) != used:
            raise ValueError(f'Có ảnh clear thiếu hazy trong split {split}: {set(sides["clear"])-used}')
    if not pairs:
        raise ValueError(f'Không tìm thấy cặp clear/hazy trong {root}')
    if dataset_name == 'o-hazy':
        # Phase 3 protocol: IDs 1-40 for tuning, 41-45 held out for test.
        def numeric(row):
            m = re.match(r'0*(\d+)', row['id'])
            return int(m.group(1)) if m else None
        by_id = {}
        for row in pairs:
            sid = numeric(row)
            if sid in by_id or sid is None:
                raise ValueError(f'Trùng hoặc thiếu ID scene O-HAZY: {row}')
            by_id[sid] = row
        if set(by_id) != set(range(1, 46)):
            raise ValueError('O-HAZY cần đủ 45 scene 01–45 để giữ protocol Phase 3.')
        pairs = [by_id[i] | {'id': i} for i in sorted(by_id)]
    return pairs


def _holdout_train(rows, seed, fraction):
    scenes = sorted({r['scene'] for r in rows})
    if len(scenes) < 2:
        raise ValueError('Cần ít nhất hai scene train để tách validation.')
    random.Random(seed).shuffle(scenes)
    val = set(scenes[:min(len(scenes)-1, max(1, round(len(scenes)*fraction)))])
    return [r for r in rows if r['scene'] not in val], [r for r in rows if r['scene'] in val]


def partition(pairs, cfg):
    dataset_name = normalize_dataset_name(cfg.get('dataset_name', 'o-hazy'))
    if dataset_name == 'o-hazy' and all(isinstance(p['id'], int) for p in pairs):
        val = set(cfg.get('val_ids', [5, 15, 25, 35]))
        if not val or not val <= set(range(1, 41)):
            raise ValueError('val_ids O-HAZY phải thuộc 1–40 và không rỗng.')
        result = {'train': [p for p in pairs if p['id'] <= 40 and p['id'] not in val],
                  'val': [p for p in pairs if p['id'] in val],
                  'test': [p for p in pairs if 41 <= p['id'] <= 45]}
    else:
        explicit = {p.get('source_split') for p in pairs}
        if None in explicit and len(explicit) > 1:
            raise ValueError('Không trộn dữ liệu phẳng với thư mục train/val/test.')
        if dataset_name.startswith('sots-'):
            # Some repackaged SOTS have train/val folders. They are still benchmark test images.
            result = {'train': [], 'val': [], 'test': list(pairs)}
        elif explicit == {None}:
            raise ValueError('Dataset train cần thư mục train và test; không tự lấy test làm validation.')
        else:
            result = {k: [p for p in pairs if p.get('source_split') == k] for k in ('train', 'val', 'test')}
            if result['train'] and not result['val']:
                result['train'], result['val'] = _holdout_train(result['train'], cfg.get('seed', 2026), .1)
    assert_disjoint(result)
    return result


def assert_disjoint(splits):
    seen = {}
    for split, rows in splits.items():
        ids = set()
        for row in rows:
            if row['id'] in ids:
                raise ValueError(f'Trùng ID trong split {split}: {row["id"]}')
            ids.add(row['id'])
            keys = [('scene', row['scene']), ('path', row['clear']), ('path', row['hazy'])]
            keys += [('clear_sha256', row['clear_sha256'])] if row.get('clear_sha256') else []
            keys += [('hazy_sha256', row['hazy_sha256'])] if row.get('hazy_sha256') else []
            for key in keys:
                if key in seen and seen[key] != split:
                    raise ValueError(f'Rò rỉ dữ liệu giữa {seen[key]} và {split}: {key}')
                seen[key] = split


def dataset_splits(cfg):
    name = normalize_dataset_name(cfg.get('dataset_name', 'o-hazy'))
    data = Path(cfg.get('data_root') or Path(cfg['project_root']) / DATASET_SPECS[name]['relative_root']).expanduser().resolve()
    if not data.is_dir():
        raise FileNotFoundError(f'Không có dataset {name} tại {data}. Đặt DATA_ROOT_OVERRIDE tới dữ liệu có sẵn.')
    splits = partition(collect_pairs(data, name), {**cfg, 'dataset_name': name})
    if cfg.get('train_root'):
        if not name.startswith('sots-'):
            raise ValueError('TRAIN_ROOT chỉ dành cho train ITS/OTS riêng khi đánh giá trên SOTS.')
        train_root = Path(cfg['train_root']).expanduser().resolve()
        if train_root == data or train_root.is_relative_to(data) or data.is_relative_to(train_root):
            raise ValueError('TRAIN_ROOT phải tách biệt với thư mục benchmark SOTS.')
        training = collect_pairs(train_root, name)
        explicit = {r.get('source_split') for r in training}
        if explicit == {None}:
            splits['train'], splits['val'] = _holdout_train(training, cfg.get('seed', 2026), .1)
        elif explicit <= {'train', 'val'} and 'train' in explicit:
            splits['train'] = [r for r in training if r['source_split'] == 'train']
            splits['val'] = [r for r in training if r['source_split'] == 'val']
            if not splits['val']:
                splits['train'], splits['val'] = _holdout_train(splits['train'], cfg.get('seed', 2026), .1)
        else:
            raise ValueError('TRAIN_ROOT chỉ chứa train/val hoặc các cặp ảnh phẳng, không chứa test.')
    requested = cfg.get('eval_only')
    eval_only = (not splits['train']) if requested is None else bool(requested)
    if not splits['test'] or (not eval_only and (not splits['train'] or not splits['val'])):
        raise ValueError('Cần test để đánh giá; fine-tune còn cần train và validation riêng. SOTS cần TRAIN_ROOT để train.')
    assert_disjoint(splits)
    return data, splits, eval_only


def manifest_digest(splits):
    rows = [{k: row[k] for k in ('split', 'id', 'scene', 'hazy_sha256', 'clear_sha256')}
            for split in ('train', 'val', 'test') for row in splits[split]]
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def prepare(cfg):
    work = Path(cfg['work_root'])
    work.mkdir(parents=True, exist_ok=True)
    root, splits, evaluation_only = dataset_splits(cfg)
    name = normalize_dataset_name(cfg.get('dataset_name', 'o-hazy'))
    # Hash shared clear images once, including synthetic sets with many atmospheric variants.
    hashes = {}
    for split, rows in splits.items():
        for row in rows:
            row.update(dataset_name=name, split=split)
            for key in ('hazy', 'clear'):
                if row[key] not in hashes:
                    hashes[row[key]] = sha256(row[key])
                row[key+'_sha256'] = hashes[row[key]]
    assert_disjoint(splits)
    value = {'dataset_name': name, 'data_root': str(root), 'evaluation_only': evaluation_only,
             'splits': splits, 'manifest_sha256': manifest_digest(splits)}
    dump_json(work/'inputs.json', value)
    write_csv(work/'split_manifest.csv', [r for rows in splits.values() for r in rows])
    print('Dataset:', name, {k: len(v) for k, v in splits.items()}, 'eval_only:', evaluation_only, flush=True)
    return value


def validate(cfg, inputs):
    root, splits, only = dataset_splits(cfg)
    if inputs['dataset_name'] != normalize_dataset_name(cfg['dataset_name']) or str(root) != inputs['data_root']:
        raise ValueError('Dataset/root đã đổi; chạy prepare lại.')
    if only != inputs['evaluation_only']:
        raise ValueError('Chế độ dataset đã đổi; chạy prepare lại.')
    for split in splits:
        keys = lambda rows: [(r['id'], r['hazy'], r['clear']) for r in rows]
        if keys(splits[split]) != keys(inputs['splits'][split]):
            raise ValueError('Split đã đổi; chạy prepare lại.')
    checked = {}
    for rows in inputs['splits'].values():
        for row in rows:
            for key in ('hazy', 'clear'):
                if row[key] not in checked:
                    checked[row[key]] = sha256(row[key])
                if checked[row[key]] != row[key+'_sha256']:
                    raise ValueError(f'Dữ liệu đã đổi: {row[key]}')
    assert_disjoint(inputs['splits'])
    if manifest_digest(inputs['splits']) != inputs['manifest_sha256']:
        raise ValueError('Manifest không còn khớp checksum.')


def load_pair(pair, max_side):
    images = []
    for key in ('hazy', 'clear'):
        image = cv2.imread(pair[key], cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f'Không đọc được {pair[key]}')
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    x, y = images
    if (pair.get('dataset_name') == 'sots-its' and y.shape[:2] == (x.shape[0]+20, x.shape[1]+20)):
        # Standard SOTS indoor GT includes a 10-pixel frame absent from the hazy image.
        y = y[10:-10, 10:-10]
    if x.shape != y.shape:
        raise ValueError(f'Cặp ảnh khác shape: scene {pair["id"]}; {x.shape}, {y.shape}')
    h, w = x.shape[:2]
    if max_side and max(h, w) > max_side:
        scale = max_side / max(h, w)
        size = (round(w * scale), round(h * scale))
        x, y = [cv2.resize(a, size, interpolation=cv2.INTER_AREA) for a in (x, y)]
    return [torch.from_numpy(a.copy()).permute(2, 0, 1).float().div_(255) for a in (x, y)]


def pad32(x):
    h, w = x.shape[-2:]
    dh, dw = (-h) % 32, (-w) % 32
    return F.pad(x, (0, dw, 0, dh), mode='reflect' if dh < h and dw < w else 'replicate')


def final_output(y):
    return y[-1] if isinstance(y, (tuple, list)) else y


def metrics(pred, gt, border=4):
    a, b = [t.detach().float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0) for t in (pred, gt)]
    if border:
        a, b = [v[border:-border, border:-border] for v in (a, b)]
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    psnr = float(-10 * np.log10(max(mse, 1e-12)))
    # Same 11x11 Gaussian window/sigma and valid-region averaging as BasicSR SSIM.
    ssim = float(structural_similarity(a, b, channel_axis=2, data_range=1.,
                                      gaussian_weights=True, sigma=1.5, use_sample_covariance=False))
    return psnr, ssim
