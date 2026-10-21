"""CPU regression tests. Tiny networks exercise orchestration, not Mamba speed/quality."""
import ast
import copy
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
import torch
from torch import nn
import phase4_data as data
import phase4_arch as arch
import phase4_runner as runner
from phase4_profile import profile_gflops
import phase4_batch as batch

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent


class TinyTrunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)
        with torch.no_grad():
            self.conv.weight.zero_(); self.conv.bias.fill_(.15)

    def forward(self, x):
        return self.conv(x)


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__(); self.conv1 = nn.Conv2d(3, 3, 1)

    def forward(self, x):
        return x+self.conv1(x)*.01


class TinyDEN(nn.Module):
    def __init__(self, depth):
        super().__init__()
        for name in ('g1','g2','g3'):
            group = nn.Module(); group.gp = nn.Sequential(*([TinyBlock() for _ in range(depth)]+[nn.Conv2d(3,3,1)]))
            setattr(self, name, group)

    def forward(self,x):
        for name in ('g1','g2','g3'): x = getattr(self,name).gp(x)
        return x


class DENTrunk(nn.Module):
    def __init__(self, depth):
        super().__init__(); self.restoration_network = nn.Module()
        self.restoration_network.DE = TinyDEN(depth)

    def forward(self,x): return self.restoration_network.DE(x)


def fixtures(root):
    root = Path(root); rows=[]
    for i in range(4):
        rng=np.random.default_rng(i+10)
        x=rng.integers(20,90,(32,40,3),dtype=np.uint8); y=np.clip(x.astype(int)+80,0,255).astype(np.uint8)
        hazy,clear=root/f'{i}_hazy.png',root/f'{i}_GT.png'
        cv2.imwrite(str(hazy),x); cv2.imwrite(str(clear),y)
        rows.append({'id':str(i),'scene':str(i),'dataset_name':'i-haze','hazy':str(hazy),'clear':str(clear)})
    schedule=[{'steps':6,'crop':24,'accum':1,'lr':.03}, {'steps':4,'crop':0,'accum':1,'lr':.01}]
    cfg={'dataset_name':'i-haze','seed':2040,'device':'cpu','den_blocks':1,'detail_width':16,'reference_depth':4,
         'train_max_side':0,'eval_max_side':0,'crop_border':4,'adapt_schedule':schedule,
         'scratch_schedule':schedule,'reference_schedule':schedule,'l1_weight':.02,'edge_weight':.01,
         'kd_weight':.1,'ema_decay':.9,'weight_decay':0.,'val_every':2,'patience':100,
         'cache_images':2,'log_every':100,'warmup':1,'timing_repeats':2,'fusion_max_error':1e-4,
         'psnr_tolerance_db':.3,'target_speedup':2.,'target_speedup_256':1.1,'deploy_psnr_tolerance_db':.01}
    inputs={'splits':{'train':rows[:2],'val':rows[2:3],'test':rows[3:]}, 'manifest_sha256':'toy',
            'source_sha256':'','student_sha256':'','code_sha256':{},'evaluation_only':False}
    return cfg,inputs


def tiny_model():
    return arch.PaddedModel(TinyTrunk(), {'kind':'test','depth':2,'detail_width':0})


class Phase4Tests(unittest.TestCase):
    def test_all_configs_preserve_domains_and_independent_paths(self):
        base={'project_root':'project','preset':'quick','eval_only':None,'reside_checkpoint':'shared.pth'}
        configs=batch.make_configs(base,data.DATASET_SPECS,'suite',
            data_roots={'sots-its':'custom_indoor'},train_roots={'sots-ots':'custom_train'},
            checkpoints={'o-hazy':'ohazy.pth','i-haze':'must_ignore.pth'},students={'o-hazy':'student.pth'})
        self.assertEqual(tuple(c['dataset_name'] for c in configs),batch.ALL_DATASETS)
        self.assertEqual(len({c['output_root'] for c in configs}),4)
        self.assertEqual(configs[0]['data_root'],'custom_indoor')
        self.assertEqual(configs[1]['train_root'],'custom_train')
        self.assertTrue(all(c['eval_only'] for c in configs[:2]))
        self.assertEqual(configs[2]['student_checkpoint'],'student.pth')
        self.assertEqual(configs[3]['checkpoint'],''); self.assertEqual(configs[3]['student_checkpoint'],'')
        self.assertTrue(all(c['reside_checkpoint']=='shared.pth' for c in configs))
        configs[0]['preset']='changed'; self.assertEqual(configs[1]['preset'],'quick')

    def test_all_jobs_continue_after_errors_and_aggregate_only_completed(self):
        import csv
        with tempfile.TemporaryDirectory() as tmp:
            configs=batch.make_configs({'project_root':tmp,'preset':'quick'},data.DATASET_SPECS,tmp)
            calls=[]
            def invoke(cfg,mode):
                name=cfg['dataset_name']; calls.append((name,mode))
                if name=='sots-its': raise FileNotFoundError('dataset missing')
                if mode=='prepare': return
                if name=='o-hazy': raise RuntimeError('interrupted')
                directory=Path(cfg['output_root']); directory.mkdir(parents=True,exist_ok=True)
                row={'dataset':name,'model':'Toy','PSNR':20,'SSIM':.8,'MSE':.01,'FPS':100,
                     'FPS_over_MSE':10000,'GFLOPs_256_estimated_total':1}
                data.write_csv(directory/'summary.csv',[row])
                (directory/'metadata.json').write_text(json.dumps({'status':'REVIEW'}),encoding='utf-8')
            output=Path(tmp)/'reports'
            with contextlib.redirect_stdout(io.StringIO()):
                states=batch.prepare_jobs(configs,invoke,output)
                batch.run_jobs(configs,states,invoke,'auto',output)
            self.assertEqual([s['status'] for s in states],['prepare_failed','completed','run_failed','completed'])
            self.assertEqual([name for name,mode in calls if mode=='auto'],['sots-ots','o-hazy','i-haze'])
            with (output/'summary_all.csv').open(encoding='utf-8') as f: rows=list(csv.DictReader(f))
            self.assertEqual([r['dataset'] for r in rows],['sots-ots','i-haze'])
            self.assertIn('run_failed',(output/'final_report_all.md').read_text(encoding='utf-8'))
            self.assertTrue((output/'report_all.html').is_file())
            self.assertEqual(json.loads((output/'suite_status.json').read_text(encoding='utf-8')),states)

    def test_all_failed_clears_previous_rollup(self):
        import csv
        with tempfile.TemporaryDirectory() as tmp:
            configs=batch.make_configs({'project_root':tmp,'preset':'quick'},data.DATASET_SPECS,tmp)
            data.write_csv(Path(tmp)/'summary_all.csv',[{'dataset':'stale'}])
            states=[{'dataset':c['dataset_name'],'status':'prepare_failed','error':'missing'} for c in configs]
            batch.collect_reports(configs,states,tmp)
            with (Path(tmp)/'summary_all.csv').open(encoding='utf-8') as f: self.assertEqual(list(csv.DictReader(f)),[])

    def test_metrics_and_macro_efficiency(self):
        a=torch.zeros(3,32,32); b=torch.full_like(a,.1)
        values=data.metric_values(a,b,4)
        self.assertAlmostEqual(values['MSE'],.01,places=7)
        self.assertAlmostEqual(values['PSNR'],20.,places=5)
        mean=runner.metric_mean([{'MSE':.01},{'MSE':.09}],'MSE')
        self.assertAlmostEqual(data.efficiency(100,mean),2000.)
        self.assertNotAlmostEqual(data.efficiency(100,mean),(100/.01+100/.09)/2)
        self.assertTrue(math.isinf(data.efficiency(100,0)))
        with self.assertRaises(ValueError): data.efficiency(100,-1)

    def test_nonfinite_or_small_images_rejected(self):
        with self.assertRaises(FloatingPointError):
            data.metric_values(torch.full((3,32,32),float('nan')),torch.zeros(3,32,32),4)
        with self.assertRaises(ValueError): data.metric_values(torch.zeros(3,12,12),torch.zeros(3,12,12),4)

    def test_real_local_data_splits(self):
        project=HERE.parents[2]
        for name,counts in [('o-hazy',(36,4,5)),('i-haze',(25,5,5))]:
            root=project/data.DATASET_SPECS[name]['relative_root']
            if not root.is_dir(): continue
            with tempfile.TemporaryDirectory() as tmp:
                cfg={'dataset_name':name,'project_root':str(project),'data_root':str(root),
                     'train_root':'','eval_only':None,'seed':2040,'work_root':tmp}
                value=data.prepare(cfg); data.validate(cfg,value)
                self.assertEqual(tuple(len(value['splits'][k]) for k in ('train','val','test')),counts)

    def test_sots_stays_test_and_scene_leakage_rejected(self):
        rows=[{'id':str(i),'scene':str(i),'hazy':str(i)+'h','clear':str(i)+'c','source_split':'train'} for i in range(4)]
        splits=data.partition(rows,{'dataset_name':'sots-its'})
        self.assertFalse(splits['train']); self.assertEqual(len(splits['test']),4)
        with self.assertRaises(ValueError): data.assert_disjoint({'train':[rows[0]],'test':[rows[0]]})

    def test_sots_never_uses_o_hazy_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'solution/Haze Clear/phase 3'; path.mkdir(parents=True)
            (path/'fast_wdmamba_o_hazy_best.pth').write_bytes(b'unused')
            with self.assertRaises(FileNotFoundError):
                runner.resolve_source({'dataset_name':'sots-its','project_root':tmp,'work_root':tmp})
            with self.assertRaises(ValueError):
                runner.resolve_source({'dataset_name':'i-haze','checkpoint':'anything'})

    def test_detect_checkpoint_depth_and_bad_topology(self):
        state={f'restoration_network.DE.g{g}.gp.{i}.conv1.weight':torch.zeros(1) for g in (1,2,3) for i in range(4)}
        self.assertEqual(arch.detect_den_depth(state),4)
        del state['restoration_network.DE.g1.gp.0.conv1.weight']
        with self.assertRaises(ValueError): arch.detect_den_depth(state)
        path=HERE.parent/'phase 3/fast_wdmamba_o_hazy_best.pth'
        if path.is_file(): self.assertEqual(arch.detect_den_depth(arch.checkpoint_state(path)),1)

    def test_identity_expansion_and_state_reconstruction(self):
        torch.manual_seed(12)
        reference=arch.PaddedModel(DENTrunk(1),{'kind':'reference','depth':1,'detail_width':0}).eval()
        model,selected=arch.final_from_reference(reference)
        x=torch.rand(1,3,35,43)
        torch.testing.assert_close(model.eval()(x),reference(x),rtol=1e-6,atol=1e-6)
        self.assertEqual(selected,[0]); self.assertEqual(model(x).shape,x.shape)
        with patch.object(arch,'new_trunk',side_effect=lambda root,depth:DENTrunk(depth)):
            rebuilt=arch.skeleton('unused',model.spec).eval(); rebuilt.load_state_dict(model.state_dict(),strict=True)
            torch.testing.assert_close(rebuilt(x),model(x))

    def test_four_to_one_plus_dw_preserves_terminal_and_no_alias(self):
        reference=arch.PaddedModel(DENTrunk(4),{'kind':'reference','depth':4,'detail_width':0})
        model,selected=arch.final_from_reference(reference)
        self.assertEqual(selected,[0])
        for group in ('g1','g2','g3'):
            src=getattr(reference.trunk.restoration_network.DE,group).gp
            dst=getattr(model.trunk.restoration_network.DE,group).gp
            torch.testing.assert_close(dst[-1].weight,src[-1].weight)
            self.assertNotEqual(dst[-1].weight.data_ptr(),src[-1].weight.data_ptr())
        model(torch.rand(1,3,32,40)).mean().backward()
        self.assertIsNotNone(model.adapter.body[-1].weight.grad)

    def test_fusion_inside_inference_mode(self):
        model=nn.Sequential(nn.Conv2d(3,4,3,padding=1),nn.BatchNorm2d(4),nn.ReLU(),nn.Conv2d(4,3,1)).eval()
        x=torch.rand(1,3,32,32); expected=model(x)
        with torch.inference_mode(): self.assertEqual(arch.fuse_conv_bn(model),1)
        self.assertFalse(model[0].weight.is_inference())
        torch.testing.assert_close(model(x),expected,rtol=1e-5,atol=1e-6)

    def test_profiler_does_not_pollute_model(self):
        model=tiny_model().train(); before=set(model.state_dict())
        result=profile_gflops(model,torch.rand(1,3,35,43))
        self.assertEqual(before,set(model.state_dict())); self.assertTrue(model.training)
        self.assertTrue(all(not m._forward_hooks for m in model.modules()))
        self.assertAlmostEqual(result['counted_GFLOPs'],2*1*3*64*64*3/1e9)

    def test_training_checkpoint_reload_and_resume_equivalence(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,inputs=fixtures(tmp); a,b=Path(tmp)/'a',Path(tmp)/'b'; a.mkdir(); b.mkdir()
            runner.configure(cfg); first=tiny_model()
            runner.train_model(first,cfg,inputs,a,'final',cfg['scratch_schedule'],'random')
            runner.configure(cfg); interrupted=tiny_model(); original=runner.atomic_save
            def save_and_interrupt(value,path):
                original(value,path)
                if Path(path).name=='final_last.pth' and value['step']==2: raise KeyboardInterrupt()
            with patch.object(runner,'atomic_save',side_effect=save_and_interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    runner.train_model(interrupted,cfg,inputs,b,'final',cfg['scratch_schedule'],'random')
            # Resume in a fresh model with disturbed RNG to verify checkpoint restoration.
            torch.rand(19); resumed=tiny_model()
            runner.train_model(resumed,cfg,inputs,b,'final',cfg['scratch_schedule'],'random',mode='resume')
            for key,value in first.state_dict().items(): torch.testing.assert_close(value,resumed.state_dict()[key],rtol=0,atol=0)
            score=runner.quality(first,inputs['splits']['val'],cfg,torch.device('cpu'))[0]['PSNR']
            self.assertGreater(score,10.)
            with patch.object(runner,'restoration_loss',side_effect=AssertionError('benchmark must not train')):
                loaded=tiny_model(); runner.train_model(loaded,cfg,inputs,a,'final',cfg['scratch_schedule'],'random',mode='benchmark')
                torch.testing.assert_close(loaded.trunk.conv.weight,first.trunk.conv.weight)
            bad=dict(cfg,seed=9)
            with self.assertRaises(ValueError): runner.train_model(tiny_model(),bad,inputs,a,'final',cfg['scratch_schedule'],'random',mode='benchmark')

    def test_runtime_selection_does_not_read_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,inputs=fixtures(tmp); inputs['splits']['val']=[]
            inputs['splits']['test']=[{'hazy':'must-not-open','clear':'must-not-open'}]
            with patch.object(data,'load_pair',side_effect=AssertionError('test consulted')):
                _,info=runner.choose_runtime(tiny_model(),cfg,inputs,Path(tmp))
            self.assertEqual(info['selection'],'synthetic')

    def test_evaluation_efficiency_and_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,inputs=fixtures(tmp)
            result=runner.evaluate_one(tiny_model(),'Toy','eager',cfg,inputs,Path(tmp))
            self.assertAlmostEqual(result['FPS_over_MSE'],result['FPS']/result['MSE'])
            worse=dict(result,PSNR=result['PSNR']-.5,FPS=result['FPS']*10,FPS_256=result['FPS_256']*10,
                       FPS_over_MSE=result['FPS_over_MSE']*10)
            self.assertFalse(runner.assess(result,worse,cfg)['quality'])
            self.assertTrue(runner.assess(result,worse,cfg)['speed'])

    def test_notebook_sources_and_schema(self):
        import nbformat
        nb=nbformat.read(HERE/'phase4_wavelite_final.ipynb',as_version=4); nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type=='code': ast.parse(cell.source)
        source=next(c.source for c in nb.cells if c.cell_type=='code' and 'MODULE_SOURCES = ' in c.source)
        sources=ast.literal_eval(ast.parse(source).body[0].value)
        for name,value in sources.items(): self.assertEqual(value,(HERE/name).read_text(encoding='utf-8-sig'))

    def test_end_to_end_i_haze_export_and_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,inputs=fixtures(tmp); output=Path(tmp)/'results'; output.mkdir()
            cfg.update(work_root=tmp,output_root=str(output),repo_root='unused')
            (Path(tmp)/'inputs.json').write_text(json.dumps(inputs),encoding='utf-8')
            def reference(*args,**kwargs):
                return arch.PaddedModel(TinyTrunk(),{'kind':'reference','depth':4,'detail_width':0})
            def final(root,spec): return arch.PaddedModel(TinyTrunk(),dict(spec))
            with patch.object(runner,'validate'), patch.object(arch,'reference_model',side_effect=reference), \
                 patch.object(arch,'skeleton',side_effect=final), contextlib.redirect_stdout(io.StringIO()):
                runner.run(cfg,'auto')
                saved=torch.load(output/'model_final.pth',map_location='cpu',weights_only=False)
                self.assertEqual(saved['origin']['method'],'random')
                self.assertEqual(saved['origin']['final_seed'],3040)
                restored=final('',saved['model_spec']); restored.load_state_dict(saved['params_ema'],strict=True)
                with patch.object(runner,'restoration_loss',side_effect=AssertionError('benchmark trained')):
                    runner.run(cfg,'benchmark')
            meta=json.loads((output/'metadata.json').read_text(encoding='utf-8'))
            self.assertIn('export_sha256',meta)
            self.assertIn('FPS/MSE',(output/'final_report.md').read_text(encoding='utf-8'))

    def test_runtime_only_never_trains_or_shrinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,inputs=fixtures(tmp); output=Path(tmp)/'results'; output.mkdir()
            cfg.update(dataset_name='sots-its',work_root=tmp,output_root=str(output),repo_root='unused')
            inputs.update(evaluation_only=True,source='domain.pth')
            inputs['splits']['train']=[]; inputs['splits']['val']=[]
            (Path(tmp)/'inputs.json').write_text(json.dumps(inputs),encoding='utf-8')
            reference=arch.PaddedModel(TinyTrunk(),{'kind':'reference','depth':6,'detail_width':0})
            with patch.object(runner,'validate'), patch.object(arch,'reference_model',return_value=reference), \
                 patch.object(runner,'train_model',side_effect=AssertionError('SOTS test used for training')), \
                 patch.object(arch,'final_from_reference',side_effect=AssertionError('pruned without train')), \
                 contextlib.redirect_stdout(io.StringIO()):
                runner.run(cfg,'auto')
            meta=json.loads((output/'metadata.json').read_text(encoding='utf-8'))
            self.assertEqual(meta['model_spec']['depth'],6)
            self.assertEqual(meta['origin']['method'],'runtime_only')

    def test_reside_checkpoint_is_shared_across_synthetic_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint=Path(tmp)/'reside6k_32.15.pth'; checkpoint.write_bytes(b'path resolution only')
            for name in ('sots-its','sots-ots','reside6k'):
                self.assertEqual(runner.resolve_source({'dataset_name':name,'reside_checkpoint':str(checkpoint)}),checkpoint.resolve())

    def test_shape_cache_reduces_timing_but_evaluates_every_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,inputs=fixtures(tmp); cfg['cache_shape_benchmarks']=True
            inputs['splits']['test']=inputs['splits']['val']+inputs['splits']['test']
            with patch.object(runner,'latency',wraps=runner.latency) as timing, \
                 patch.object(runner,'profile_gflops',wraps=runner.profile_gflops) as flops, \
                 contextlib.redirect_stdout(io.StringIO()):
                result=runner.evaluate_one(tiny_model(),'Cached','eager',cfg,inputs,Path(tmp))
            self.assertEqual(result['images'],2)
            self.assertEqual(timing.call_count,2)  # one image shape + 256 probe
            self.assertEqual(flops.call_count,2)
            import csv
            with (Path(tmp)/'Cached_timing.csv').open(encoding='utf-8') as f: rows=list(csv.DictReader(f))
            self.assertEqual([r['timing_source'] for r in rows],['measured','shape_cache'])

    def test_quick_runtime_path_skips_train_even_when_train_split_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,inputs=fixtures(tmp); output=Path(tmp)/'results'; output.mkdir()
            cfg.update(dataset_name='reside6k',work_root=tmp,output_root=str(output),repo_root='unused',
                       runtime_only_datasets=['reside6k'],skip_duplicate_eager=True,
                       cache_shape_benchmarks=True,preset='quick')
            inputs.update(source='reside6k.pth')  # evaluation_only remains False
            (Path(tmp)/'inputs.json').write_text(json.dumps(inputs),encoding='utf-8')
            reference=arch.PaddedModel(TinyTrunk(),{'kind':'reference','depth':6,'detail_width':0})
            with patch.object(runner,'validate'), patch.object(arch,'reference_model',return_value=reference), \
                 patch.object(runner,'train_model',side_effect=AssertionError('quick path trained')), \
                 contextlib.redirect_stdout(io.StringIO()): runner.run(cfg,'auto')
            import csv
            with (output/'summary.csv').open(encoding='utf-8') as f: rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),2)
            self.assertTrue((output/'report.html').is_file())
            self.assertTrue((output/'protocol_report.md').is_file())
            self.assertIn('FPS/MSE',(output/'final_report.md').read_text(encoding='utf-8'))


if __name__=='__main__': unittest.main()
