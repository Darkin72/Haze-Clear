"""Retry failed jobs using their original saved sources; reuse completed results."""
from pathlib import Path
import json
import subprocess
import sys
from checkpoint_setup import ensure_shared_weights


ORDER = ('sots-its', 'sots-ots', 'o-hazy', 'i-haze')


def recover_all(root, python, local_zip='', invoke=None):
    root = Path(root).expanduser().resolve()
    configs = []
    for name in ORDER:
        path = root/'datasets'/name/'config.json'
        if not path.is_file():
            raise FileNotFoundError(f'Không có {path}. Cần dùng runtime cũ còn thư mục run đã chạy.')
        cfg = json.loads(path.read_text(encoding='utf-8'))
        if cfg['dataset_name'] != name:
            raise ValueError(f'Config dataset không khớp: {path}')
        configs.append(cfg)
    output = root/'results'; output.mkdir(parents=True, exist_ok=True)
    status_path = output/'suite_status.json'
    old = json.loads(status_path.read_text(encoding='utf-8')) if status_path.is_file() else []
    completed = {s['dataset'] for s in old if s['status'] == 'completed'}
    states, pending = [], []
    for cfg in configs:
        dest = Path(cfg['output_root'])
        done = (cfg['dataset_name'] in completed and (dest/'summary.csv').is_file()
                and (dest/'metadata.json').is_file())
        states.append({'dataset': cfg['dataset_name'], 'status': 'completed' if done else 'ready', 'error': ''})
        if done:
            print(f'{cfg["dataset_name"]}: giữ kết quả đã hoàn tất, không train/benchmark lại.', flush=True)
        else:
            cfg['allow_weight_download'] = True
            if local_zip:
                cfg['checkpoint_zip'] = local_zip
            pending.append(cfg)
    errors = ensure_shared_weights(pending, root/'checkpoints')

    def run(cfg, action):
        args = [str(python), '-u', str(Path(cfg['work_root'])/'phase4_runner.py'),
                str(Path(cfg['work_root'])/'config.json'), action]
        print('$', ' '.join(args), flush=True)
        with (root/f'recover_{cfg["dataset_name"]}_{action}.log').open('a', encoding='utf-8') as log:
            with subprocess.Popen(args, cwd=cfg['repo_root'], stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace') as proc:
                try:
                    for line in proc.stdout:
                        print(line, end='', flush=True); log.write(line); log.flush()
                    code = proc.wait()
                except BaseException:
                    proc.kill(); proc.wait(); raise
            if code:
                raise subprocess.CalledProcessError(code, args)

    invoke = invoke or run
    batch_ns = {'__name__': 'phase4_batch_recovery'}
    batch_path = Path(configs[0]['work_root'])/'phase4_batch.py'
    exec(compile(batch_path.read_text(encoding='utf-8-sig'), str(batch_path), 'exec'), batch_ns)
    for cfg, state in zip(configs, states):
        if state['status'] == 'completed':
            continue
        state['status'] = 'prepare_failed'
        try:
            if cfg['dataset_name'] in errors:
                raise RuntimeError(errors[cfg['dataset_name']])
            # Download/search options are not part of the training identity.
            # Keep original runner/architecture sources and schedules untouched.
            (Path(cfg['work_root'])/'config.json').write_text(
                json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
            invoke(cfg, 'prepare')
            state['status'] = 'run_failed'
            invoke(cfg, 'auto')
            state['status'] = 'completed'
        except Exception as exc:
            state['error'] = str(exc)
            print(f'{cfg["dataset_name"]}: {exc}', flush=True)
        finally:
            batch_ns['save_status'](output, states)
            batch_ns['collect_reports'](configs, states, output)
    missing = [s['dataset'] for s in states if s['status'] != 'completed']
    if missing:
        raise RuntimeError('Lượt all chưa hoàn tất: ' + ', '.join(missing) + '. Xem suite_status.json và log khôi phục.')
    print('Hoàn tất đủ 4/4 dataset:', output/'final_report_all.md', flush=True)
    return states


if __name__ == '__main__':
    recover_all(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else '')
