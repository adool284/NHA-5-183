import hashlib,json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prepare import checked,validate_splits,read
from collect import ROOT

class IntegrityTests(unittest.TestCase):
    def test_external_path_rejected(self):
        with self.assertRaises(ValueError):checked({'image':'../outside.jpg','sha256':'x'})
    def test_capture_leakage_rejected(self):
        with self.assertRaises(ValueError):validate_splits({'reference':[{'capture_group':'same'}],'test':[{'capture_group':'same'}]})
    def test_reencoded_duplicate_leakage_rejected(self):
        with self.assertRaises(ValueError):validate_splits({'reference':[{'pixel_sha256':'same'}],'test':[{'pixel_sha256':'same'}]})
    def test_same_split_sequence_allowed(self):
        validate_splits({'reference':[{'capture_group':'same'},{'capture_group':'same'}]})
    def test_file_hash_leakage_rejected(self):
        with self.assertRaises(ValueError):validate_splits({'train':[{'sha256':'same'}],'test':[{'sha256':'same'}]})
    def test_original_source_leakage_rejected(self):
        with self.assertRaises(ValueError):validate_splits({'train':[{'original_sha256':'same'}],'validation':[{'original_sha256':'same'}]})
    def test_generated_split_and_unsupported_assignments(self):
        groups={name:read(f'manifests/{name}.json') for name in ('train','validation','test')}
        validate_splits(groups)
        reference=read('manifests/combined_reference.json')
        render_ids={row['image_id'] for row in reference if row.get('kind')=='scan_render'}
        self.assertEqual({row['image_id'] for row in groups['train']},
                         {row['image_id'] for row in reference}-render_ids)
        self.assertEqual({row['image_id'] for row in read('manifests/reference_only.json')},render_ids)
        training_ids={row['artifact_id'] for row in groups['train']}
        unsupported={row['artifact_id'] for row in read('unsupported_classes.json')['classes']}
        self.assertFalse(training_ids & unsupported)
        self.assertEqual(training_ids|unsupported,{row['id'] for row in read('catalog.json')})
        self.assertTrue(all(row['split']=='train' and row['source_split']=='reference' for row in groups['train']))
    def test_all_manifest_image_paths_and_hashes(self):
        active={row['id'] for row in read('catalog.json')}
        checksums={}
        for manifest in (ROOT/'manifests').glob('*.json'):
            rows=json.loads(manifest.read_text(encoding='utf-8'))
            if not isinstance(rows,list):continue
            for row in rows:
                if not isinstance(row,dict) or 'image' not in row:continue
                path=(manifest.parent/row['image']).resolve()
                with self.subTest(manifest=manifest.name,image=row['image']):
                    self.assertTrue(path.is_relative_to(ROOT))
                    self.assertTrue(path.is_file())
                    if path not in checksums:checksums[path]=hashlib.sha256(path.read_bytes()).hexdigest()
                    self.assertEqual(checksums[path],row['sha256'])
                    self.assertTrue(row.get('artifact_id') is None or row['artifact_id'] in active)

if __name__=='__main__':unittest.main()
