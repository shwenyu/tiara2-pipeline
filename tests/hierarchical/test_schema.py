import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tiara.hierarchical.schema import schema,scaled_priors
from tiara.hierarchical.labels import labels_for,euk_group
class T(unittest.TestCase):
 def test_profiles(self):
  self.assertEqual(schema('2.3.0').profile.root,('euk_nuclear','prok','organelle'))
  self.assertIn('virus',schema('2.3.1').profile.root);self.assertTrue(schema('2.3.2').profile.calibration_enabled)
 def test_priors(self):self.assertAlmostEqual(sum(scaled_priors(.1).values()),1.,places=7)
 def test_labels(self):
  self.assertEqual(labels_for('bacteria')['prok'],'bacteria');self.assertEqual(labels_for('plastids')['organelle'],'plastid');self.assertEqual(labels_for('eukarya','id sg=fungi')['euk'],'fungi')
 def test_ambiguous_archaeplastida_rejected(self):
  with self.assertRaises(ValueError):euk_group('id sg=Archaeplastida')
if __name__=='__main__':unittest.main(verbosity=2)
