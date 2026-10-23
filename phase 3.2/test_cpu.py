import ast
import json
import sys
import tempfile
import unittest
import contextlib
import io
from unittest.mock import patch
from pathlib import Path

import cv2
import numpy as np
import nbformat
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data_utils as data
from swdnet import ARCH_VERSION, HaarDown, HaarUp, SWDNet
import train_scratch as runner


class Phase32Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    @staticmethod
    def tiny_config(root):
        for i, split in enumerate(('train', 'train', 'train', 'val', 'test')):
            rng = np.random.default_rng(i)
            clear = rng.integers(20, 180, (33, 39, 3), dtype=np.uint8)
            hazy = np.clip(clear.astype(np.float32)*.7+65, 0, 255).astype(np.uint8)
            for kind, image in (('clear', clear), ('hazy', hazy)):
                folder = root/'dataset'/split/kind
                folder.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(folder/f'scene{i}.png'), image)
        return {'dataset_name': 'i-haze', 'data_root': str(root/'dataset'), 'train_root': '', 'eval_only': None,
                'work_root': str(root/'work'), 'output_root': str(root/'output'), 'seed': 2032, 'device': 'cpu',
                'model': {'width':8,'depths':[1,1,1,1,1]},
                'schedule':[{'steps':2,'crop':16,'batch_size':2},{'steps':2,'crop':0,'batch_size':1}],
                'lr':1e-3,'min_lr':1e-5,'warmup_steps':1,'weight_decay':0.,'ema_decay':.9,
                'l1_weight':.05,'edge_weight':.02,'train_precision':'fp32',
                'train_max_side':0,'eval_max_side':0,'crop_border':0,'cache_images':2,
                'val_every':2,'log_every':1,'early_stop_patience':0,'early_stop_min_steps':0,
                'warmup':1,'timing_repeats':2,'evaluation_checkpoint':''}

    def test_haar_exact_reconstruction_odd_and_even(self):
        for h, w in ((8, 12), (17, 23)):
            x = torch.randn(2, 3, h, w)
            padded = torch.nn.functional.pad(x, (0, (-w)%2, 0, (-h)%2), mode='replicate')
            y = HaarUp()(HaarDown()(padded))
            torch.testing.assert_close(y, padded, atol=1e-6, rtol=1e-6)

    def test_new_model_shape_identity_and_finite(self):
        torch.manual_seed(2032)
        model = SWDNet(width=16, depths=(1,1,2,1,1)).eval()
        x = torch.rand(2, 3, 65, 71)
        y = model(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all())
        # Zero heads make the random-start model a valid identity restoration baseline.
        torch.testing.assert_close(y, x, atol=1e-5, rtol=1e-5)
        self.assertEqual(model.spec['arch_version'], ARCH_VERSION)
        self.assertGreater(sum(p.numel() for p in model.parameters()), 10000)
        self.assertFalse(any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules()))

    def test_model_learns_and_gradients_reach_backbone(self):
        torch.manual_seed(2032)
        model = SWDNet(width=8, depths=(1,1,1,1,1)).train()
        x = torch.rand(1,3,16,24)
        gt = x*.8
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        initial = (model(x)-gt).square().mean().item()
        for _ in range(15):
            optimizer.zero_grad()
            loss = (model(x)-gt).square().mean()
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(), initial)
        self.assertGreater(model.stem.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_gflops_hooks_cleanup_and_training_restore(self):
        model = SWDNet(width=8, depths=(1,1,1,1,1)).train()
        before = model.training
        result = runner.profile_gflops(model, torch.rand(1,3,32,40))
        self.assertTrue(before and model.training)
        self.assertGreater(result['counted'], 0)
        self.assertGreater(result['supplemental'], 0)
        self.assertFalse(any(m._forward_hooks for m in model.modules()))
        conv = nn.Conv2d(3,4,3,padding=1)
        result = runner.profile_gflops(conv,torch.zeros(1,3,8,8))
        self.assertAlmostEqual(result['counted']*1e9, 13824)
        haar = runner.profile_gflops(HaarDown(),torch.rand(1,3,8,8))
        self.assertAlmostEqual(haar['supplemental']*1e9, 4*3*8*8)

    def test_selector_and_split_from_repo(self):
        root = HERE.parents[2]
        self.assertEqual(data.normalize_dataset_name('SOTS Indoor / ITS'), 'sots-its')
        self.assertEqual(data.normalize_dataset_name('reside 6k'), 'reside6k')
        pairs = data.collect_pairs(root/'dataset/O-HAZY', 'o-hazy')
        splits = data.partition(pairs, {'dataset_name':'o-hazy', 'val_ids':[5,15,25,35]})
        self.assertEqual([len(splits[k]) for k in ('train','val','test')], [36,4,5])

    def test_schedule_and_loss_finite(self):
        cfg = {'schedule':[{'steps':3,'crop':16,'batch_size':2}], 'val_every':1, 'cache_images':1}
        cfg.update({'crop_border':2, 'l1_weight':.1, 'edge_weight':.05})
        runner.validate_schedule(cfg)
        pred, gt = torch.rand(2,3,16,16), torch.rand(2,3,16,16)
        self.assertTrue(torch.isfinite(runner.restoration_loss(pred,gt,cfg)))
        with self.assertRaises(ValueError):
            runner.validate_schedule({'schedule':[], 'val_every':1, 'cache_images':1})

    def test_complete_training_evaluation_and_resume_reproduce_weights(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            cfg = self.tiny_config(root)
            inputs = data.prepare(cfg)
            result = runner.run(cfg, 'train')
            self.assertTrue(np.isfinite(result['PSNR']))
            self.assertEqual(result['images'],1)
            self.assertEqual(result['device'],'cpu')
            self.assertGreater(result['GFLOPs_256_counted'],0)
            final = torch.load(root/'output/last.pth',map_location='cpu',weights_only=False)
            self.assertEqual(final['initialization'],'random')
            self.assertEqual(final['step'],4)
            self.assertTrue(final['completed'])
            with self.assertRaises(FileExistsError):
                runner.train(cfg,inputs,root/'output')

            interrupted = {**cfg, 'output_root':str(root/'interrupted')}
            original_save = runner.atomic_save
            def stop_at_boundary(value, path):
                original_save(value, path)
                if Path(path).name == 'last.pth' and value['step'] == 2:
                    raise InterruptedError('simulated interruption after durable checkpoint')
            with patch.object(runner,'atomic_save',side_effect=stop_at_boundary):
                with self.assertRaises(InterruptedError):
                    runner.run(interrupted, 'train')
            runner.run(interrupted,'resume')
            resumed = torch.load(root/'interrupted/last.pth',map_location='cpu',weights_only=False)
            for key in final['params']:
                torch.testing.assert_close(final['params'][key],resumed['params'][key],atol=0,rtol=0)
                torch.testing.assert_close(final['params_ema'][key],resumed['params_ema'][key],atol=0,rtol=0)
            bad = {**cfg,'lr':5e-4}
            with self.assertRaises(ValueError):
                runner.train(bad,inputs,root/'output',resume=True)

    def test_manifest_tamper_and_sots_no_training(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp)
            cfg=self.tiny_config(root)
            inputs=data.prepare(cfg)
            row=inputs['splits']['train'][0]
            Path(row['hazy']).write_bytes(b'tampered')
            with self.assertRaises(ValueError): data.validate(cfg,inputs)
            with self.assertRaises(ValueError):
                runner.train(cfg,{'evaluation_only':True},root/'output')
            cfg['evaluation_checkpoint']=str(root/'missing.pth')
            with self.assertRaises(FileNotFoundError): runner.evaluate(cfg,inputs,root/'output')

    def test_notebook_embeds_exact_sources_and_no_pretrained_checkpoint(self):
        nb = nbformat.read(HERE/'phase3_2_swdnet_scratch.ipynb', as_version=4)
        nbformat.validate(nb)
        found = {}
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != 'code': continue
            compile(cell.source, f'cell-{i}', 'exec')
            for node in ast.parse(cell.source).body:
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in {'DATA_SOURCE','ARCH_SOURCE','RUNNER_SOURCE'}:
                            found[target.id] = ast.literal_eval(node.value)
        for key, filename in [('DATA_SOURCE','data_utils.py'),('ARCH_SOURCE','swdnet.py'),('RUNNER_SOURCE','train_scratch.py')]:
            self.assertEqual(found[key].strip(), (HERE/filename).read_text(encoding='utf-8').strip())
        setup = ''.join(nb.cells[2].source)
        self.assertIn("RUN_MODE = 'train'", setup)
        self.assertNotIn("'phase3_checkpoint'", setup)


if __name__ == '__main__':
    unittest.main()
