"""Local integration validation. Requires pre-downloaded official sample; never downloads."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from face_lab.__main__ import main
from face_lab.models import sha256


def call(*args, expected=0):
    # Prove analysis/regroup/check do not use Python networking.
    with patch("sys.argv", ["face_lab", *map(str,args)]), patch("socket.socket", side_effect=AssertionError("NETWORK DISABLED")):
        code = main()
    if code != expected:
        raise AssertionError(f"exit {code}, expected {expected}: {args}")


class IntegrationTests(unittest.TestCase):
    def test_local_pipeline(self):
        sample = ROOT/"validation_data"/"official_sample"/"lena.jpg"
        if not sample.exists():
            self.skipTest("Official sample was not downloaded; see TESTING.md")
        base = ROOT/"validation_data"/("integration_"+__import__("uuid").uuid4().hex[:8])
        data, out = base/"한글 공백 입력", base/"outputs"
        data.mkdir(parents=True)
        shutil.copyfile(sample, data/"공개 샘플 1.jpg")
        shutil.copyfile(sample, data/"공개 샘플 2.jpg")
        Image.new("RGB", (320,240), "white").save(data/"얼굴 없는 생성 이미지.jpg")
        (data/"손상 사진.jpg").write_bytes(b"broken jpeg")
        before = {p.name:sha256(p) for p in data.iterdir()}
        call("check")
        unicode_models=base/"한글 모델 폴더"
        shutil.copytree(ROOT/"models",unicode_models)
        call("check","--models",unicode_models)
        call("analyze", "--input", data, "--output", out)
        run = next(out.iterdir())
        summary = json.loads((run/"summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["total_photos"],4)
        self.assertEqual(summary["failed_photos"],1)
        self.assertGreaterEqual(summary["detected_faces"],2)
        self.assertEqual(summary["groups"],1)
        self.assertEqual(summary["zero_face_photos"],1)
        with np.load(run/"features.npz",allow_pickle=False) as archive:
            self.assertTrue(np.allclose(np.linalg.norm(archive["vectors"],axis=1),1,atol=1e-6))
        call("regroup", "--source", run, "--input", data, "--output", out, "--similarity-threshold", ".50")
        rerun = [r for r in out.iterdir() if r!=run][0]
        self.assertEqual((run/"faces.csv").read_bytes(),(rerun/"faces.csv").read_bytes())
        call("regroup", "--source",run,"--input",data,"--output",out,"--min-group-size","3")
        unknown_run=sorted(out.iterdir())[-1]
        unknown_summary=json.loads((unknown_run/"summary.json").read_text(encoding="utf-8"))
        self.assertEqual(unknown_summary["unknown_faces"],2)
        self.assertTrue((unknown_run/"montages"/"unknown_001.jpg").exists())
        # Metadata version mismatch must not reuse a stale preprocessing cache.
        original_cache=(run/"cache.json").read_bytes()
        invalid=json.loads(original_cache)
        invalid["signature"]["runtime"]["preprocessing"]="different-version"
        (run/"cache.json").write_text(json.dumps(invalid),encoding="utf-8")
        try:
            call("regroup","--source",run,"--input",data,"--output",out,expected=2)
        finally:
            (run/"cache.json").write_bytes(original_cache)
        # A second extraction must have the same face mappings and labels.
        call("analyze", "--input", data, "--output", out)
        repeated = sorted(out.iterdir())[-1]
        self.assertEqual((run/"faces.csv").read_bytes(), (repeated/"faces.csv").read_bytes())
        self.assertEqual(before,{p.name:sha256(p) for p in data.iterdir()})
        call("evaluate", "--run",run,"--ground-truth",run/"ground_truth_template.csv",expected=2)
        with (run/"ground_truth_template.csv").open(encoding="utf-8-sig",newline="") as f:
            reader=csv.DictReader(f)
            fields=reader.fieldnames
            rows=list(reader)
        for row in rows:
            row["manual_person_id"]="public_sample_person"
        truth=run/"filled_ground_truth.csv"
        with truth.open("w",encoding="utf-8-sig",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        call("evaluate","--run",run,"--ground-truth",truth)
        metric=json.loads(next((run/"evaluations").glob("*.json")).read_text(encoding="utf-8"))
        self.assertEqual(metric["f1"],1)
        # Offline annotation UI and explicit correction workflows.
        call('review','--run',run)
        self.assertTrue(list(run.glob('review_*.html')))
        call('apply-review','--source',run,'--input',data,'--output',out,'--ground-truth',truth)
        manual_run=sorted(out.iterdir())[-1]
        manual_summary=json.loads((manual_run/'summary.json').read_text(encoding='utf-8'))
        self.assertEqual(manual_summary['mode'],'apply-review')
        self.assertEqual(manual_summary['groups'],1)
        call('evaluate','--run',manual_run,'--ground-truth',truth,expected=2)
        from face_lab.review import context_hash
        source_cache=json.loads((run/'cache.json').read_text(encoding='utf-8'))
        correction_file=base/'synthetic_test_constraints.json'
        correction_file.write_text(json.dumps(dict(context_sha256=context_hash(source_cache),source_faces_sha256=sha256(run/'faces.csv'),
            same_person_sets=[],different_person_pairs=[[r['face_id'] for r in rows]])),encoding='utf-8')
        call('correct-groups','--source',run,'--input',data,'--output',out,'--corrections',correction_file)
        split_run=sorted(out.iterdir())[-1]
        checks=json.loads((split_run/'constraint_checks.json').read_text(encoding='utf-8'))
        self.assertEqual(checks['before']['mixed_pair_violations'],1)
        self.assertEqual(checks['after']['mixed_pair_violations'],0)
        # Source modification must invalidate regroup, not reuse stale features.
        with (data/"공개 샘플 1.jpg").open("ab") as f:
            f.write(b"changed")
        call("regroup","--source",run,"--input",data,"--output",out,expected=2)
        print(f"INTEGRATION_ARTIFACTS: {base}")

    def test_disjoint_dev_selection_eval_pipeline(self):
        sample = ROOT/'validation_data/official_sample/lena.jpg'
        if not sample.exists():
            self.skipTest('Local sample missing')
        import tempfile
        from face_lab.pipeline import write_csv
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);data=base/'input';data.mkdir();out=base/'out'
            for i in range(4): shutil.copy2(sample,data/f'{i}.jpg')
            call('analyze','--input',data,'--output',out,'--min-group-size','1')
            source=next(out.iterdir())
            cache=json.loads((source/'cache.json').read_text())
            truth=[]
            from face_lab.review import context_hash
            for f in cache['faces']:
                truth.append(dict(face_id=f['face_id'],manual_person_id='public_A',include='1',_context_sha256=context_hash(cache)))
            truthpath=base/'truth.csv';write_csv(truthpath,truth,list(truth[0]))
            splits=[dict(photo_id=p['photo_id'],split='dev' if i<2 else 'eval') for i,p in enumerate(cache['photos'])]
            splitpath=base/'split.csv';write_csv(splitpath,splits,['photo_id','split'])
            compareout=base/'comparison'
            call('compare','--source',source,'--input',data,'--output',compareout,
                 '--split-manifest',splitpath,'--subset','dev','--ground-truth',truthpath,'--thresholds','.5')
            dev=next(compareout.iterdir())
            report=json.loads((dev/'comparison.json').read_text())
            for result in report['experiments'].values():
                self.assertEqual(result['summary']['detected_faces'],2)
                self.assertEqual(result['evaluation']['evaluated_faces'],2)
                self.assertEqual(result['evaluation']['f1'],1)
            call('select-config','--comparison',dev,'--experiment','baseline_complete')
            selection=next(dev.glob('selected_config_*.json'))
            call('compare','--source',source,'--input',data,'--output',compareout,
                 '--split-manifest',splitpath,'--subset','eval',expected=2)
            call('compare','--source',source,'--input',data,'--output',compareout,
                 '--split-manifest',splitpath,'--subset','eval','--selected-config',selection,
                 '--ground-truth',truthpath,'--similarity-threshold','.99')
            evaluation=sorted(compareout.iterdir())[-1]
            report=json.loads((evaluation/'comparison.json').read_text())
            self.assertEqual(len(report['experiments']),1)
            frozen=report['experiments']['frozen_dev_selection']
            self.assertEqual(frozen['config']['similarity_threshold'],.45)  # eval CLI cannot tune the frozen threshold
            self.assertEqual(frozen['evaluation']['f1'],1)
            dev_ids=set(json.loads((dev/'baseline_complete/experiment.json').read_text())['face_ids'])
            eval_ids=set(json.loads((evaluation/'frozen_dev_selection/experiment.json').read_text())['face_ids'])
            self.assertFalse(dev_ids & eval_ids)
            call('evaluate','--run',evaluation/'frozen_dev_selection','--ground-truth',truthpath,
                 '--split-manifest',splitpath,'--subset','eval')
            call('review','--run',dev/'baseline_complete')
            call('select-config','--comparison',evaluation,'--experiment','frozen_dev_selection',expected=2)


if __name__=="__main__":
    unittest.main()
