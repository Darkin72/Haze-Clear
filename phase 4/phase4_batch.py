"""Standard-library orchestration for the notebook's all-dataset option."""
from pathlib import Path
import copy
import csv
import html
import json

ALL_DATASETS = ('sots-its', 'sots-ots', 'o-hazy', 'i-haze')


def make_configs(base, specs, root, data_roots=None, train_roots=None, checkpoints=None,
                 students=None):
    root = Path(root)
    configs = []
    for name in ALL_DATASETS:
        cfg = copy.deepcopy(base)
        work = root/'datasets'/name
        cfg.update(dataset_name=name, work_root=str(work), output_root=str(work/'results'),
                   data_root=str((data_roots or {}).get(name) or
                                 Path(base['project_root'])/specs[name]['relative_root']),
                   train_root=(train_roots or {}).get(name, ''),
                   checkpoint=(checkpoints or {}).get(name, ''),
                   student_checkpoint=(students or {}).get(name, ''))
        if name == 'i-haze':
            cfg['checkpoint'] = cfg['student_checkpoint'] = ''
            cfg['eval_only'] = False
        else:
            cfg['eval_only'] = True if base.get('preset') == 'quick' and name.startswith('sots-') else base.get('eval_only')
        configs.append(cfg)
    return configs


def save_status(output, states):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    (output/'suite_status.json').write_text(json.dumps(states, ensure_ascii=False, indent=2), encoding='utf-8')


def prepare_jobs(configs, invoke, output):
    states = []
    for cfg in configs:
        state = {'dataset': cfg['dataset_name'], 'status': 'ready', 'error': ''}
        try:
            invoke(cfg, 'prepare')
        except Exception as exc:
            state.update(status='prepare_failed', error=str(exc))
            print(f'{cfg["dataset_name"]}: chuẩn bị lỗi: {exc}', flush=True)
        states.append(state); save_status(output, states)
    return states


def run_jobs(configs, states, invoke, mode, output):
    for cfg, state in zip(configs, states):
        if state['status'] != 'ready': continue
        try:
            invoke(cfg, mode)
            state['status'] = 'completed'
        except Exception as exc:
            state.update(status='run_failed', error=str(exc))
            print(f'{cfg["dataset_name"]}: chạy lỗi: {exc}', flush=True)
        save_status(output, states)
        collect_reports(configs, states, output)
    return collect_reports(configs, states, output)


def collect_reports(configs, states, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    summaries, final_rows, statuses = [], [], []
    for cfg, state in zip(configs, states):
        name, directory = cfg['dataset_name'], Path(cfg['output_root'])
        if state['status'] != 'completed':
            statuses.append(f'| {name} | {state["status"]} |')
            continue
        with (directory/'summary.csv').open(encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        meta = json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
        summaries.extend(rows)
        if rows:
            final_rows.append((name, rows[-1], meta['status'], directory))
        statuses.append(f'| {name} | {meta["status"]} |')
    keys = list(dict.fromkeys(k for row in summaries for k in row)) or [
        'dataset', 'model', 'PSNR', 'SSIM', 'MSE', 'FPS', 'FPS_over_MSE', 'GFLOPs_256_estimated_total']
    with (output/'summary_all.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(summaries)
    lines = ['# Phase 4 — Tổng hợp bốn bộ dữ liệu', '',
             '| Dataset | PSNR | SSIM | FPS | FPS/MSE | GFLOPs 256* | Trạng thái |',
             '|---|---:|---:|---:|---:|---:|---|']
    for name, row, status, _ in final_rows:
        lines.append(f'| {name} | {float(row["PSNR"]):.3f} | {float(row["SSIM"]):.4f} | '
                     f'{float(row["FPS"]):.1f} | {float(row["FPS_over_MSE"]):,.0f} | '
                     f'{float(row["GFLOPs_256_estimated_total"]):.2f} | {status} |')
    lines += ['', '*GFLOPs gồm các thành phần ước tính; xem protocol từng dataset. '
              'Không gộp trung bình PSNR/FPS giữa các dataset.', '',
              '| Dataset | Tình trạng lượt chạy |', '|---|---|', *statuses,
              '', 'Lỗi chi tiết: suite_status.json. Mỗi dataset có report và metadata riêng.']
    (output/'final_report_all.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    escape = lambda s: html.escape(str(s))
    table = ''.join('<tr>'+''.join(f'<td>{escape(v)}</td>' for v in (
        name, f'{float(row["PSNR"]):.3f}', f'{float(row["SSIM"]):.4f}', f'{float(row["FPS"]):.1f}',
        f'{float(row["FPS_over_MSE"]):,.0f}', f'{float(row["GFLOPs_256_estimated_total"]):.2f}', status))+'</tr>'
        for name,row,status,_ in final_rows)
    document = '''<!doctype html><html lang="vi"><meta charset="utf-8"><title>Phase 4 — All</title>
<style>body{background:#f3f6fb;color:#162238;font:16px/1.6 system-ui;margin:48px auto;max-width:1100px;padding:24px}
table{width:100%;border-collapse:collapse;background:white}td,th{padding:18px;text-align:right;border-bottom:1px solid #e8edf4}td:first-child,th:first-child{text-align:left}small{color:#61718c}</style>
<small>KẾT QUẢ THỰC NGHIỆM · PHASE 4</small><h1>Bốn bộ dữ liệu</h1>
<table><tr><th>Dataset</th><th>PSNR</th><th>SSIM</th><th>FPS</th><th>FPS/MSE</th><th>GFLOPs 256*</th><th>Trạng thái</th></tr>'''
    document += table+'</table><p>*GFLOPs có thành phần ước tính. Không lấy trung bình giữa các dataset.</p>'
    document += '<ul>'+''.join(f'<li>{escape(s["dataset"])}: {escape(s["status"])} {escape(s["error"])}</li>' for s in states)+'</ul></html>'
    (output/'report_all.html').write_text(document, encoding='utf-8')
    return final_rows
