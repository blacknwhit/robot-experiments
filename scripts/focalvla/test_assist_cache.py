import json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from assist_cache import complete,coverage,merge,label_path,save,sha,digest,NAMES

class AssistTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).parent);self.base=Path(self.tmp.name)
        self.primary=self.base/'primary';self.helper=self.base/'helper';self.identity={'test':'same_teacher'}
        self.rows=[{'id':i,'name':f'case_{i}','length':4} for i in range(4)]
        self.records=np.array([[i,t,j] for i in range(4) for j,t in enumerate([0,3])],np.int32)
    def tearDown(self):self.tmp.cleanup()
    def put(self,root,i,value=1):
        row=self.rows[i];frames=np.array([0,3],np.int32);dest=root/'cache'/f'{i:06d}';dest.mkdir(parents=True,exist_ok=True)
        lab=label_path(root,row);save(lab,{'goals':[{'text':'same label'}]})
        arrays={'state':np.zeros((4,16),np.float32),'action':np.zeros((4,12),np.float32),'frames':frames,
            'geo':np.full((2,2048),value,np.float32),'motion':np.ones((2,256),np.float32),
            'geo_valid':np.ones(2,bool),'motion_valid':np.ones(2,bool),'goal_ids':np.zeros(2,np.int32)}
        for key,x in arrays.items():np.save(dest/(key+'.npy'),x)
        save(dest/'COMPLETE.json',{'identity':self.identity,'image_storage':'source_video','frames':2,'label_sha256':sha(lab),
          'contract':digest({'inputs':self.identity,'label_sha256':sha(lab),'episode':row,'frames':[0,3]}),
          'file_bytes':{k+'.npy':(dest/(k+'.npy')).stat().st_size for k in NAMES}})
    def test_union_merge_primary_precedence_and_resume(self):
        self.put(self.primary,0,7);self.put(self.helper,0,9)
        for i in [1,2,3]:self.put(self.helper,i)
        partial=self.primary/'cache/000001';partial.mkdir();(partial/'partial').write_text('preserved')
        proof=coverage(self.primary,self.helper,self.rows,self.records,self.identity);self.assertEqual(proof['episodes'],4)
        with self.assertRaises(RuntimeError):merge(self.primary,self.helper,self.rows,self.records,self.identity)
        r=merge(self.primary,self.helper,self.rows,self.records,self.identity,writers_stopped=True)
        self.assertEqual(r,{'kept_primary':1,'merged_helper':3})
        self.assertTrue(np.all(np.load(self.primary/'cache/000000/geo.npy')==7))
        self.assertEqual(len(list((self.primary/'assist_partial_archive').glob('*/partial'))),1)
        self.assertEqual(merge(self.primary,self.helper,self.rows,self.records,self.identity,writers_stopped=True)['merged_helper'],0)
    def test_partial_and_wrong_identity_not_accepted(self):
        self.put(self.helper,0);(self.helper/'cache/000000/geo.npy').write_bytes(b'partial')
        self.assertIsNone(complete(self.helper,self.rows[0],[0,3],self.identity))
        self.assertIsNone(coverage(self.primary,self.helper,self.rows,self.records,self.identity))
        with self.assertRaises(RuntimeError):merge(self.primary,self.helper,self.rows,self.records,self.identity,writers_stopped=True)
        self.put(self.helper,0)
        with self.assertRaises(ValueError):complete(self.helper,self.rows[0],[0,3],{'test':'different_teacher'})
    def test_conflicting_labels_refused(self):
        self.put(self.helper,0);save(label_path(self.primary,self.rows[0]),{'goals':[{'text':'manual correction'}]})
        with self.assertRaises(ValueError):merge(self.primary,self.helper,self.rows[:1],self.records,self.identity,writers_stopped=True)
        self.assertTrue((self.helper/'cache/000000/COMPLETE.json').exists())
    def test_live_primary_growth_is_observed(self):
        self.assertIsNone(complete(self.primary,self.rows[0],[0,3],self.identity))
        self.put(self.primary,0)
        self.assertIsNotNone(complete(self.primary,self.rows[0],[0,3],self.identity))

if __name__=='__main__':unittest.main()
