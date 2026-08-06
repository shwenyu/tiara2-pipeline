import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tiara.hierarchical.schema import schema,scaled_priors,EUK
from tiara.hierarchical.labels import labels_for,euk_group
class T(unittest.TestCase):
 def test_profiles_follow_roadmap(self):
  self.assertEqual(schema('2.3.1').profile.root,schema('2.3.0').profile.root)
  self.assertTrue(schema('2.3.1').profile.euk_completeness_enabled)
  self.assertFalse(schema('2.3.1').profile.virus_enabled)
  self.assertTrue(schema('2.3.2').profile.branch_balancing_enabled)
  self.assertIn('virus',schema('2.4.0').profile.root)
 def test_eight_leaves(self):self.assertEqual(len(EUK),8)
 def test_priors(self):self.assertAlmostEqual(sum(scaled_priors(.1).values()),1.,places=7)
 def test_labels(self):self.assertEqual(labels_for('bacteria')['prok'],'bacteria');self.assertEqual(labels_for('plastids')['organelle'],'plastid');self.assertEqual(labels_for('eukarya','id sg=fungi')['euk'],'fungi')
 def test_ambiguous(self):
  with self.assertRaises(ValueError):euk_group('id sg=Archaeplastida')
if __name__=='__main__':unittest.main(verbosity=2)
