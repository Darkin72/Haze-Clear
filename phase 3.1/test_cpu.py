"""CPU checks for real fusion, counter conventions, hook cleanup and notebook packaging."""
import ast
import copy
import sys
import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path

import nbformat
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refine_o_hazy as v2


class V2Tests(unittest.TestCase):
    def test_counter_conv_linear_transpose_and_no_hooks_left(self):
        # 3->4, 3x3 conv, 8x8 output: 64*4*27 MACs; each MAC is two FLOPs.
        model = nn.Conv2d(3, 4, 3, padding=1)
        result = v2.profile_gflops(model, torch.rand(1, 3, 8, 8))
        self.assertAlmostEqual(result['counted_GFLOPs']*1e9, 13824.)
        self.assertEqual(result['supplemental_GFLOPs'], 0.)
        self.assertFalse(model._forward_hooks)
        linear = nn.Linear(3, 5)
        self.assertAlmostEqual(v2.profile_gflops(linear, torch.rand(2, 3))['counted_GFLOPs']*1e9, 60.)
        up = nn.ConvTranspose2d(2, 3, 2, stride=2)
        self.assertAlmostEqual(v2.profile_gflops(up, torch.rand(1, 2, 4, 4))['counted_GFLOPs']*1e9, 768.)

    def test_counter_cleanup_on_failure_and_restores_training(self):
        class Bad(nn.Module):
            def forward(self, x):
                raise RuntimeError('intentional failure')
        model = Bad().train()
        with self.assertRaisesRegex(RuntimeError, 'intentional'):
            v2.profile_gflops(model, torch.zeros(1))
        self.assertTrue(model.training)
        self.assertFalse(model._forward_hooks)

    def test_fusion_preserves_nontrivial_batchnorm_inference(self):
        torch.manual_seed(42)
        model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU())
        model[1].running_mean.copy_(torch.linspace(-1, 1, 8))
        model[1].running_var.copy_(torch.linspace(.2, 2, 8))
        model.eval()
        x = torch.randn(2, 3, 17, 29)
        expected = model(x)
        before = copy.deepcopy(model.state_dict())
        self.assertEqual(v2.fuse_conv_bn(model), 1)
        torch.testing.assert_close(model(x), expected, atol=2e-6, rtol=2e-6)
        self.assertIsInstance(model[1], nn.Identity)
        self.assertEqual(v2.fuse_conv_bn(model), 0)
        self.assertIn('1.running_mean', before)

    def test_fusion_never_creates_inference_parameters(self):
        model = nn.Sequential(nn.Conv2d(3, 4, 1, bias=False), nn.BatchNorm2d(4)).eval()
        x = torch.randn(1, 3, 9, 13)
        with torch.no_grad():
            expected = model(x)
        # Reproduce the caller context that caused the deployment crash on CUDA.
        with torch.inference_mode():
            self.assertEqual(v2.fuse_conv_bn(model), 1)
        for p in model.parameters():
            self.assertFalse(torch.is_inference(p))
            self.assertIsInstance(p._version, int)
        with torch.no_grad():
            torch.testing.assert_close(model(x), expected, atol=1e-6, rtol=1e-6)
        with torch.inference_mode():
            torch.testing.assert_close(model(x), expected, atol=1e-6, rtol=1e-6)

    def test_graph_failure_falls_back_to_ordinary_fused_model(self):
        import contextlib
        import io
        model = nn.Sequential(nn.Conv2d(3, 4, 1, bias=False), nn.BatchNorm2d(4)).eval()
        x = torch.randn(3, 32, 32)
        expected = model(x[None]).detach()
        cfg = {'eval_max_side': 0, 'fusion_max_error': 1e-4, 'try_cuda_graph': True}
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(nn.Module, 'cuda', lambda self: self), \
             patch.object(torch.Tensor, 'cuda', lambda self: self), \
             patch.object(v2.core, 'load_pair', return_value=(x, x)), \
             patch.object(v2.core, 'latency', return_value={'latency_ms': 10.}), \
             patch.object(v2.GraphReplay, 'forward', side_effect=RuntimeError('capture failed')), \
             contextlib.redirect_stdout(io.StringIO()):
            with torch.inference_mode():
                chosen, info = v2.choose_deployment(model, cfg, {'splits': {'val': [{}]}}, Path(tmp))
        self.assertIs(chosen, model)
        self.assertEqual(info['execution'], 'fused_eager')
        self.assertIn('capture failed', info['graph_error'])
        self.assertTrue(all(not torch.is_inference(p) for p in chosen.parameters()))
        with torch.no_grad():
            torch.testing.assert_close(chosen(x[None]), expected, atol=1e-6, rtol=1e-6)

    def test_restore_v2_reuses_checkpoint_and_rejects_wrong_provenance(self):
        cfg = {'dataset_name': 'o-hazy', 'den_blocks': 1, 'eval_max_side': 768,
               'crop_border': 4, 'val_ids': [5, 15, 25, 35]}
        inputs = {'checkpoint_sha256': 'teacher', 'phase3_sha256': 'baseline'}
        source, student = nn.Linear(1, 1), nn.Linear(1, 1)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            saved = {'config': cfg, 'params_ema': source.state_dict(), 'teacher_sha256': 'teacher',
                     'phase3_sha256': 'baseline', 'step': 100, 'val_PSNR': 27.79}
            checkpoint = output/'fast_wdmamba_v2_best.pth'
            torch.save(saved, checkpoint)
            training = v2.restore_v2(cfg, inputs, student, output)
            for a, b in zip(source.parameters(), student.parameters()):
                torch.testing.assert_close(a, b)
            self.assertEqual(training['selected_step'], 100)
            self.assertEqual(training['best_val_PSNR'], 27.79)
            with self.assertRaises(ValueError):
                v2.restore_v2(cfg, {**inputs, 'checkpoint_sha256': 'wrong'}, student, output)
            with self.assertRaises(ValueError):
                v2.restore_v2({**cfg, 'dataset_name': 'i-haze'}, inputs, student, output)
            with self.assertRaises(ValueError):
                v2.restore_v2(cfg, {**inputs, 'phase3_sha256': 'wrong'}, student, output)
            with self.assertRaises(FileNotFoundError):
                v2.restore_v2({**cfg, 'v2_checkpoint': str(output/'missing.pth')}, inputs, student, output)

    def test_auto_reuses_v2_after_failure_without_training(self):
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = {'work_root': str(root), 'output_root': str(root)}
            v2.core.dump_json(root/'inputs.json', {'evaluation_only': False})
            with patch.object(v2, 'run') as train, patch.object(v2, 'benchmark') as bench, \
                 contextlib.redirect_stdout(io.StringIO()):
                v2.auto(cfg)
                train.assert_called_once_with(cfg)
                bench.assert_not_called()
                train.reset_mock()
                (root/'fast_wdmamba_v2_best.pth').write_bytes(b'checkpoint; restore validates contents')
                v2.auto(cfg)
                bench.assert_called_once_with(cfg)
                train.assert_not_called()
                bench.reset_mock()
                # A benchmark dataset must not silently adopt an unrelated cached student.
                v2.core.dump_json(root/'inputs.json', {'evaluation_only': True})
                v2.auto(cfg)
                train.assert_called_once_with(cfg)
                bench.assert_not_called()

    def test_real_dataset_split_and_phase3_checkpoint(self):
        root = HERE.parents[2]
        pairs = v2.core.collect_pairs(root/'dataset/O-HAZY')
        parts = v2.core.partition(pairs, {'val_ids': [5, 15, 25, 35]})
        self.assertEqual([len(parts[k]) for k in ('train', 'val', 'test')], [36, 4, 5])
        self.assertEqual([p['id'] for p in parts['test']], [41, 42, 43, 44, 45])
        checkpoint = torch.load(HERE.parent/'phase 3/fast_wdmamba_o_hazy_best.pth',
                                map_location='cpu', weights_only=False)
        self.assertEqual(checkpoint['config']['den_blocks'], 1)
        self.assertEqual(checkpoint['config']['val_ids'], [5, 15, 25, 35])

    def test_dataset_selectors_and_layouts(self):
        root = HERE.parents[2]
        self.assertEqual(v2.core.normalize_dataset_name('SOTS Indoor / ITS'), 'sots-its')
        self.assertEqual(v2.core.normalize_dataset_name('ots'), 'sots-ots')
        self.assertEqual(v2.core.normalize_dataset_name('RESIDE-6K'), 'reside6k')
        with self.assertRaises(ValueError):
            v2.core.normalize_dataset_name('unknown')
        ihaze = v2.core.collect_pairs(root/'dataset/I-HAZE', 'i-haze')
        splits = v2.core.partition(ihaze, {'dataset_name': 'i-haze', 'seed': 2026})
        self.assertEqual([len(splits[k]) for k in ('train', 'val', 'test')], [25, 5, 5])
        self.assertTrue(all('_hazy' in p['id'] for p in splits['test']))
        self.assertFalse(set(p['scene'] for p in splits['train']) & set(p['scene'] for p in splits['val']))

    def test_sots_is_evaluation_only(self):
        self.assertFalse(v2.core.DATASET_SPECS['sots-its']['can_train'])
        self.assertFalse(v2.core.DATASET_SPECS['sots-ots']['can_train'])
        rows = [{'id': 'a', 'scene': 'a', 'source_split': 'test', 'clear': 'c', 'hazy': 'h'}]
        parts = v2.core.partition(rows, {'dataset_name': 'sots-its'})
        self.assertEqual(len(parts['train']), 0)
        self.assertEqual(len(parts['test']), 1)

    def test_split_leakage_guard(self):
        row = {'id': 'same', 'scene': 'scene', 'clear': 'clear', 'hazy': 'hazy'}
        with self.assertRaises(ValueError):
            v2.core.assert_disjoint({'train': [row], 'val': [dict(row)]})

    @staticmethod
    def make_scene(root, scene, variants=1):
        clear = root/'GT'/f'{scene}.png'
        clear.parent.mkdir(parents=True, exist_ok=True)
        # Pairing/manifest checks use file bytes, independent of the image codec.
        clear.write_bytes(('clear-'+scene).encode())
        for i in range(variants):
            hazy = root/'hazy'/f'{scene}_{i}_0.2.png'
            hazy.parent.mkdir(parents=True, exist_ok=True)
            hazy.write_bytes(('hazy-'+scene+str(i)).encode())

    def test_reside_shared_gt_and_validation_from_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(10):
                self.make_scene(root/'train', f'{i:04d}', 2)
            self.make_scene(root/'test', '1000', 2)
            cfg = {'data_root': str(root), 'dataset_name': 'reside6k', 'seed': 2026}
            _, splits, eval_only = v2.core.dataset_splits(cfg)
            self.assertFalse(eval_only)
            self.assertEqual([len(splits[k]) for k in ('train', 'val', 'test')], [18, 2, 2])
            again = v2.core.dataset_splits(cfg)[1]
            self.assertEqual(splits, again)
            v2.core.assert_disjoint(splits)
            self.assertEqual(len({r['scene'] for r in splits['val']}), 1)

    def test_sots_prepare_manifest_and_no_test_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root/'sots'
            self.make_scene(data, '1000', 2)
            checkpoint = root/'teacher.pth'
            checkpoint.write_bytes(b'checkpoint-for-manifest-only')
            cfg = {'dataset_name': 'sots-its', 'data_root': str(data), 'work_root': str(root/'work'),
                   'checkpoint': str(checkpoint), 'phase3_checkpoint': '', 'eval_only': None}
            v2.prepare(cfg)
            import json
            inputs = json.loads((root/'work/inputs.json').read_text(encoding='utf-8'))
            self.assertTrue(inputs['evaluation_only'])
            self.assertIsNone(inputs['phase3_checkpoint'])
            self.assertEqual(len(inputs['splits']['test']), 2)
            self.assertIn('hazy_sha256', inputs['splits']['test'][0])
            with self.assertRaises(ValueError):
                v2.core.dataset_splits({**cfg, 'eval_only': False})
            v2.validate_inputs(cfg, inputs)
            # Training is skipped before any CUDA, cache or checkpoint operation.
            with patch.object(v2.core, 'load_pair', side_effect=AssertionError('test read for training')):
                self.assertEqual(v2.refine(cfg, inputs, None, None, root)['steps'], 0)
            training = root/'its'
            for i in range(3):
                self.make_scene(training, str(i), 2)
            _, splits, only = v2.core.dataset_splits({**cfg, 'train_root': str(training), 'eval_only': False})
            self.assertFalse(only)
            self.assertEqual([len(splits[k]) for k in ('train', 'val', 'test')], [4, 2, 2])
            with self.assertRaises(ValueError):
                v2.core.dataset_splits({**cfg, 'train_root': str(data)})

    def test_unmatched_pair_and_duplicate_hash_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_scene(root, '1')
            (root/'hazy/missing_0.2.png').write_bytes(b'missing')
            with self.assertRaises(ValueError):
                v2.core.collect_pairs(root, 'reside6k')
        a = {'id': 'a', 'scene': 'a', 'clear': 'a', 'hazy': 'ha', 'clear_sha256': 'same'}
        b = {'id': 'b', 'scene': 'b', 'clear': 'b', 'hazy': 'hb', 'clear_sha256': 'same'}
        with self.assertRaises(ValueError):
            v2.core.assert_disjoint({'train': [a], 'test': [b]})

    def test_runner_keeps_initial_baseline_separate_from_refinement(self):
        # Exercise orchestration/report writing without pretending CPU timings are GPU results.
        import contextlib
        import io
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = {'work_root': str(root), 'output_root': str(root/'results'), 'dataset_name': 'reside6k',
                   'min_psnr_gain_db': .1, 'min_fps_ratio': 1.03, 'psnr_tolerance_db': .3,
                   'deploy_psnr_tolerance_db': .01}
            inputs = {'dataset_name': 'reside6k', 'evaluation_only': False, 'checkpoint': 'teacher.pth',
                      'checkpoint_sha256': 'teacher', 'phase3_checkpoint': None, 'phase3_sha256': None,
                      'splits': {'test': [{'id': 'one'}]}}
            v2.core.dump_json(root/'inputs.json', inputs)
            teacher, student = nn.Linear(1, 1, bias=False), nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                teacher.weight.fill_(3)
                student.weight.fill_(1)

            def fake_refine(cfg, inputs, teacher, student, output):
                with torch.no_grad():
                    student.weight.fill_(2)
                return {'steps': 1}

            measured = {}
            def fake_evaluate(model, cfg, inputs, label, execution, output):
                weight = model.weight.item()
                measured[label] = weight
                return {'dataset': cfg['dataset_name'], 'model': label, 'execution': execution,
                        'PSNR': 27+weight*.2, 'SSIM': .9, 'FPS': 100+weight*4, 'FPS_256': 200.,
                        'GFLOPs_256_counted': 1., 'GFLOPs_256_supplemental': .2,
                        'GFLOPs_256_estimated_total': 1.2, 'params_M': .001}

            with patch.object(v2.torch.cuda, 'is_available', return_value=True), \
                 patch.object(v2.torch.cuda, 'get_device_name', return_value='CPU test stub'), \
                 patch.object(v2, 'configure'), patch.object(v2, 'validate_inputs'), \
                 patch.object(v2.core, 'teacher_and_student', return_value=(teacher, student, 6)), \
                 patch.object(v2, 'refine', side_effect=fake_refine), \
                 patch.object(v2, 'choose_deployment', side_effect=lambda m, *args: (m, {'execution': 'stub'})), \
                 patch.object(v2, 'evaluate', side_effect=fake_evaluate), contextlib.redirect_stdout(io.StringIO()):
                v2.run(cfg)
            self.assertEqual(measured['Compressed_init'], 1.)
            self.assertEqual(measured['V2_eager'], 2.)
            self.assertEqual(measured['V2_deploy'], 2.)
            meta = json.loads((root/'results/metadata.json').read_text(encoding='utf-8'))
            self.assertEqual(meta['baseline_model'], 'Compressed_init')
            self.assertEqual(meta['dataset'], 'reside6k')
            self.assertIn('GFLOPs counted', (root/'results/final_report.md').read_text(encoding='utf-8'))

    def test_embedded_sources_and_notebook_syntax(self):
        nb = nbformat.read(HERE/'phase3_1_psnr_fps_gflops.ipynb', as_version=4)
        nbformat.validate(nb)
        self.assertEqual(nb.metadata['wdmamba_runner_version'], 'v2.1-fusion-fix-auto-resume')
        self.assertIn("RUN_MODE = 'auto'", nb.cells[2].source)
        self.assertIn('TRY_CUDA_GRAPH = False', nb.cells[2].source)
        embedded = {}
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != 'code':
                continue
            compile(cell.source, f'cell-{i}', 'exec')
            for node in ast.parse(cell.source).body:
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in ('CORE_SOURCE', 'RUNNER_SOURCE'):
                            embedded[target.id] = ast.literal_eval(node.value)
        for key, filename in [('CORE_SOURCE', 'phase3_core.py'), ('RUNNER_SOURCE', 'refine_o_hazy.py')]:
            compile(embedded[key], filename, 'exec')
            self.assertEqual(embedded[key].strip(), (HERE/filename).read_text(encoding='utf-8').strip())


if __name__ == '__main__':
    unittest.main()
