import unittest
from controller import receipt_valid,RELEASE,RUN
class ReceiptTests(unittest.TestCase):
    def test_foreign_or_incomplete_receipt_never_triggers_evaluation(self):
        good={'passed':True,'step':1500,'release_sha256':RELEASE,'run':str(RUN)}
        self.assertTrue(receipt_valid(good))
        for change in ({'passed':False},{'step':1000},{'release_sha256':'foreign'},{'run':'foreign'}):
            self.assertFalse(receipt_valid({**good,**change}))
if __name__=='__main__':unittest.main()
