#!/usr/bin/env python3
from __future__ import annotations
import csv, json, sys, tempfile, unittest, contextlib, io
from pathlib import Path
HERE=Path(__file__).resolve().parent; REPO=HERE.parent
for p in (str(REPO/'scripts'),str(REPO)):
    if p not in sys.path: sys.path.insert(0,p)
import plan_genus_balance as G
import merge_training_indexes as M

def wt(path,cols,rows):
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols,delimiter='\t'); w.writeheader(); w.writerows(rows)
def rt(path):
    with open(path,newline='') as f:return list(csv.DictReader(f,delimiter='\t'))

class TestGenusBalance(unittest.TestCase):
 def setUp(self):
    self.d=Path(tempfile.mkdtemp()); self.tax=self.d/'tax'; self.tax.mkdir()
    nodes={1:(1,'no rank'),2:(1,'superkingdom'),10:(2,'genus'),11:(10,'species'),111:(11,'strain'),112:(11,'strain'),12:(10,'species'),121:(12,'strain'),13:(10,'species'),131:(13,'strain'),2759:(1,'superkingdom'),20:(2759,'genus'),21:(20,'species'),211:(21,'strain')}
    names={1:'root',2:'Bacteria',10:'Popularus',11:'Popularus one',111:'strain A',112:'strain B',12:'Popularus two',121:'strain C',13:'Popularus three',131:'strain D',2759:'Eukaryota',20:'Alga',21:'Alga alpha',211:'alga strain'}
    with open(self.tax/'nodes.dmp','w') as f:
      for n,(p,r) in nodes.items(): f.write(f'{n}\t|\t{p}\t|\t{r}\t|\n')
    with open(self.tax/'names.dmp','w') as f:
      for n,name in names.items(): f.write(f'{n}\t|\t{name}\t|\t\t|\tscientific name\t|\n')
    self.cand=self.d/'cand.tsv'; cols=['assembly_accession','training_class','taxid','organism_name','refseq_category','assembly_level','contig_count']
    rows=[
      {'assembly_accession':'A1','training_class':'bacteria','taxid':'111','organism_name':'p1a','refseq_category':'reference genome','assembly_level':'Complete Genome','contig_count':'1'},
      {'assembly_accession':'A2','training_class':'bacteria','taxid':'112','organism_name':'p1b','assembly_level':'Scaffold','contig_count':'100'},
      {'assembly_accession':'B1','training_class':'bacteria','taxid':'121','organism_name':'p2','assembly_level':'Complete Genome','contig_count':'2'},
      {'assembly_accession':'C1','training_class':'bacteria','taxid':'131','organism_name':'p3','assembly_level':'Complete Genome','contig_count':'3'},
      {'assembly_accession':'E1','training_class':'eukarya','taxid':'211','organism_name':'alga','assembly_level':'Complete Genome'},
      {'assembly_accession':'M1','training_class':'mitochondria','taxid':'211','organism_name':'alga mito'},]
    wt(self.cand,cols,rows); self.out=self.d/'plan.tsv'
 def test_species_then_genus_cap_and_class_separation(self):
    with contextlib.redirect_stdout(io.StringIO()):
      code=G.main(['--candidates',str(self.cand),'--taxdump',str(self.tax),'--out',str(self.out),'--max-species-per-genus','2','--quiet'])
    self.assertEqual(code,0); rows=rt(self.out); acc={r['assembly_accession'] for r in rows}
    self.assertIn('A1',acc); self.assertNotIn('A2',acc)
    self.assertEqual(len([r for r in rows if r['training_class']=='bacteria']),2)
    self.assertIn('E1',acc); self.assertIn('M1',acc)
    rep=json.loads((self.d/'genus_balance_report.json').read_text())
    self.assertEqual(rep['per_class']['bacteria']['duplicate_assemblies_removed'],1)
    self.assertEqual(rep['per_class']['bacteria']['genomes_removed_by_genus_cap'],1)

 def test_per_split_cap_is_independent(self):
    split_path=self.d/'splits.tsv'
    wt(split_path,['entity_id','split'],[
      {'entity_id':'A1','split':'train'},
      {'entity_id':'A2','split':'train'},
      {'entity_id':'B1','split':'validation'},
      {'entity_id':'C1','split':'test'},
      {'entity_id':'E1','split':'train'},
      {'entity_id':'M1','split':'train'},])
    with contextlib.redirect_stdout(io.StringIO()):
      code=G.main(['--candidates',str(self.cand),'--taxdump',str(self.tax),
                   '--out',str(self.out),'--split-assignments',str(split_path),
                   '--scope','per_split','--max-species-per-genus','1','--quiet'])
    self.assertEqual(code,0)
    rows=rt(self.out)
    bacteria=[r for r in rows if r['training_class']=='bacteria']
    self.assertEqual({r['split'] for r in bacteria},{'train','validation','test'})
    self.assertEqual(len(bacteria),3)
    rep=json.loads((self.d/'genus_balance_report.json').read_text())
    self.assertEqual(rep['scope'],'per_split')
    self.assertEqual(rep['max_species_per_genus'],1)

 def test_global_scope_matches_v21(self):
    with contextlib.redirect_stdout(io.StringIO()):
      G.main(['--candidates',str(self.cand),'--taxdump',str(self.tax),
              '--out',str(self.out),'--scope','global',
              '--max-species-per-genus','1','--quiet'])
    bacteria=[r for r in rt(self.out) if r['training_class']=='bacteria']
    self.assertEqual(len(bacteria),1)

class TestIndexMerge(unittest.TestCase):
 def test_tiara_only_is_deferred(self):
    d=Path(tempfile.mkdtemp()); old=d/'old.tsv'; st=d/'status.tsv'; ti=d/'ti.tsv'; out=d/'out'
    wt(old,['entity_id','entity_type','assembly_accession','klass','taxid'],[
      {'entity_id':'GCA_1.1','entity_type':'assembly','assembly_accession':'GCA_1.1','klass':'bacteria','taxid':'11'},
      {'entity_id':'NC_1','entity_type':'organelle','assembly_accession':'NC_1','klass':'mitochondrion','taxid':'21'}])
    wt(st,['entity_id','result'],[{'entity_id':'GCA_1.1','result':'ok'},{'entity_id':'NC_1','result':'cached'}])
    wt(ti,['accession','status','is_anchor','fine_class'],[{'accession':'GCA_1.2','status':'already_held_other_version','is_anchor':'0','fine_class':'bacteria'},{'accession':'GCA_9.1','status':'new','is_anchor':'1','fine_class':'eukarya'}])
    with contextlib.redirect_stdout(io.StringIO()): M.main(['--existing',str(old),'--download-status',str(st),'--tiara1',str(ti),'--out-dir',str(out),'--quiet'])
    cur=rt(out/'current_available_candidates.tsv'); self.assertEqual({r['training_class'] for r in cur},{'bacteria','mitochondria'})
    deferred=rt(out/'deferred_downloads.tsv'); self.assertIn('GCA_9.1',{r['tiara1_accession'] for r in deferred})

if __name__=='__main__': unittest.main(verbosity=2)
