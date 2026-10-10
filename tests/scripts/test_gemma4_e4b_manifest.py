"""Strict CPU-only contract tests; synthetic reports are not inference evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts/mm/check_gemma4_e4b_manifest.py'
FIXTURES = ROOT / 'tests/fixtures/gemma4_e4b_google'
spec = importlib.util.spec_from_file_location('e4b_manifest', SCRIPT)
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'fixtures'
        shutil.copytree(FIXTURES, self.root)
        self.path = self.root / 'manifest.json'
        self.metadata = json.loads(self.path.read_text())

    def save(self):
        self.path.write_text(json.dumps(self.metadata))

    def report(self):
        return dict(
            model_id=validator.MODEL_ID, revision=validator.REVISION,
            vllm_revision=validator.VLLM_REVISION,
            transformers_revision=validator.TRANSFORMERS_REVISION,
            cases=[dict(id=name, status='passed', executed=1, skipped=0,
                        token_mismatches=0, integer_mismatches=0,
                        max_abs_error=0.0, max_rel_error=0.0)
                   for name in validator.MANDATORY_CASES])

    def test_original_metadata(self):
        self.assertEqual(validator.validate_metadata(self.path)['model_id'], validator.MODEL_ID)

    def test_wrong_identity(self):
        for name in ['unsloth/gemma-4-E4B-it', 'google/gemma-4-E2B-it', '']:
            with self.subTest(name=name):
                self.metadata['model_id'] = name
                self.save()
                with self.assertRaisesRegex(ValueError, 'model_id'):
                    validator.validate_metadata(self.path)

    def test_wrong_revision(self):
        for key in ['revision', 'vllm_revision', 'transformers_revision']:
            with self.subTest(key=key):
                original = self.metadata[key]
                self.metadata[key] = '0' * 40
                self.save()
                with self.assertRaisesRegex(ValueError, key):
                    validator.validate_metadata(self.path)
                self.metadata[key] = original

    def test_wrong_hash_and_self_attested_replacement(self):
        self.metadata['files'][0]['sha256'] = '0' * 64
        self.save()
        with self.assertRaisesRegex(ValueError, 'pin'):
            validator.validate_metadata(self.path)

    def test_corrupt_config(self):
        (self.root / 'config.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'hash|size'):
            validator.validate_metadata(self.path)

    def test_absent_fixture(self):
        (self.root / 'processor_config.json').unlink()
        with self.assertRaisesRegex(ValueError, 'fixture'):
            validator.validate_metadata(self.path)

    def test_missing_file_entry(self):
        self.metadata['files'].pop()
        self.save()
        with self.assertRaises(ValueError):
            validator.validate_metadata(self.path)

    def test_path_escape_and_symlink(self):
        self.metadata['files'][0]['file'] = '../config.json'
        self.save()
        with self.assertRaises(ValueError):
            validator.validate_metadata(self.path)
        self.metadata = json.loads((FIXTURES / 'manifest.json').read_text())
        self.save()
        (self.root / 'config.json').unlink()
        (self.root / 'config.json').symlink_to(FIXTURES / 'config.json')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            validator.validate_metadata(self.path)

    def test_report_structure_positive_only(self):
        self.assertIsNone(validator.validate_report_structure(self.report()))

    def test_empty_or_missing_case(self):
        for cases in [[], self.report()['cases'][:-1]]:
            report = self.report()
            report['cases'] = cases
            with self.assertRaisesRegex(ValueError, 'case'):
                validator.validate_report_structure(report)

    def test_skipped_mandatory_case(self):
        for key, value in [('status', 'skipped'), ('skipped', 1), ('executed', 0)]:
            report = self.report()
            report['cases'][0][key] = value
            with self.assertRaises(ValueError):
                validator.validate_report_structure(report)

    def test_nonfinite_and_invalid_metrics(self):
        for key in ['max_abs_error', 'max_rel_error', 'token_mismatches', 'integer_mismatches']:
            for value in [float('nan'), float('inf'), -1, True, '0', None]:
                with self.subTest(key=key, value=value):
                    report = self.report()
                    report['cases'][0][key] = value
                    with self.assertRaises(ValueError):
                        validator.validate_report_structure(report)

    def test_permissive_mismatch(self):
        for key in ['token_mismatches', 'integer_mismatches']:
            report = self.report()
            report['cases'][0][key] = 1
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                validator.validate_report_structure(report)
        report = self.report()
        report['allow_mismatch'] = True
        with self.assertRaises(ValueError):
            validator.validate_report_structure(report)

    def test_report_identity_and_duplicates(self):
        report = self.report()
        report['model_id'] = 'unsloth/gemma-4-E4B-it'
        with self.assertRaises(ValueError):
            validator.validate_report_structure(report)
        report = self.report()
        report['cases'].append(copy.deepcopy(report['cases'][0]))
        with self.assertRaises(ValueError):
            validator.validate_report_structure(report)

    def test_unresolved_thresholds_cannot_be_self_certified(self):
        with self.assertRaisesRegex(ValueError, 'PENDING'):
            validator.validate_runtime(self.report())
        report = self.report()
        report['thresholds'] = {'max_abs_error': 1000}
        with self.assertRaises(ValueError):
            validator.validate_runtime(report)

    def test_cli_offline_is_not_gateability(self):
        result = subprocess.run([sys.executable, str(SCRIPT), '--metadata', str(self.path), '--offline'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        self.assertEqual(output['level'], 'metadata-only')
        self.assertFalse(output['runtime_accepted'])
        self.assertEqual(output['original_model_gateability'], 'PENDING')

    def test_cli_runtime_fails_closed_and_offline_cannot_ignore_report(self):
        report = self.root / 'report.json'
        report.write_text(json.dumps(self.report()))
        for args in [[], ['--report', str(report)], ['--offline', '--report', str(report)]]:
            result = subprocess.run([sys.executable, str(SCRIPT), '--metadata', str(self.path), *args], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)

    def test_duplicate_json_keys_rejected(self):
        self.path.write_text('{"model_id":"wrong","model_id":"google/gemma-4-E4B-it"}')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            validator.validate_metadata(self.path)


if __name__ == '__main__':
    unittest.main()
