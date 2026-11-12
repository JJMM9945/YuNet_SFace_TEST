import unittest
import numpy as np
from face_lab.albums import build_collections


class PhotoCountTests(unittest.TestCase):
    def test_photo_identity_pair_order_and_crowd_collection(self):
        photos = [dict(photo_id=str(i),relative_path=f'{i}.jpg') for i in range(10)]
        counts = {str(i):1 if i<5 else 2 if i<8 else 3 for i in range(10)}
        def face(i,g,quality=''):
            return dict(photo_id=str(i), group_id=g, quality_reason=quality)
        results = dict(solo=[face(i,'A' if i<2 else 'B') for i in range(5)],
                       pair=[face(5,'A'),face(5,'B'),face(6,'B'),face(6,'A'),face(7,'A'),face(7,'C')],
                       crowd=[face(i,g) for i in (8,9) for g in ('A','B','C')])
        cases=build_collections(results,photos,counts)
        self.assertEqual(sorted(len(g['photos']) for g in cases['solo']),[2,3])
        self.assertEqual(sorted(len(g['photos']) for g in cases['pair']),[1,2])
        self.assertEqual(len(cases['crowd']),1)
        self.assertEqual(len(cases['crowd'][0]['photos']),2)
        self.assertEqual(len({p['photo_id'] for g in cases['all'] for p in g['photos']}),10)

    def test_uncertain_pair_does_not_receive_combination(self):
        photos=[dict(photo_id='p',relative_path='p.jpg')]
        for identities,quality in [(('A',''),'') , (('A','A'),''), (('A','B'),'blurry_face')]:
            faces=[dict(photo_id='p',group_id=g,quality_reason=quality if i==0 else '') for i,g in enumerate(identities)]
            groups=build_collections({'pair':faces},photos,{'p':2})['pair']
            self.assertEqual(groups[0]['id'],'pair-pending')
            self.assertEqual(groups[0]['identities'],[])



if __name__ == '__main__':
    unittest.main()
