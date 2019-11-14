import numpy as np
from math import *
import fractals
import unittest
from collections import OrderedDict
from scipy import misc
import Image

class TestFractalFunctions(unittest.TestCase):

    def setUp(self):
        self.muWhite = np.matrix(np.asarray(Image.open('whiteTest.png').convert("L")))
        self.muBlack = np.matrix(np.asarray(Image.open('blackTest.png').convert("L")))
        self.muMixed1 = np.matrix(np.asarray(Image.open('mixedTest.png').convert("L")))
        self.muMixed2 = np.matrix(np.asarray(Image.open('test1Small.png').convert("L")))

    def testDomainGet(self):
        whiteDomain = np.matrix([[255,255,255,255], [255,255,255,255], [255,255,255,255], [255,255,255,255]])
        np.testing.assert_array_equal(fractals.domainGet(self.muWhite, 0), whiteDomain)
        np.testing.assert_array_equal(fractals.domainGet(self.muWhite, 8), whiteDomain)
        blackDomain = np.matrix([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
        np.testing.assert_array_equal(fractals.domainGet(self.muBlack, 0), blackDomain)
        np.testing.assert_array_equal(fractals.domainGet(self.muBlack, 8), blackDomain)
        mixedDomain1 = np.matrix([[255,255,0,0],[255,255,0,0],[255,255,0,0], [255,255,0,0]])
        np.testing.assert_array_equal(fractals.domainGet(self.muMixed1, 1), mixedDomain1)
        np.testing.assert_array_equal(fractals.domainGet(self.muMixed1, 0), whiteDomain)
        np.testing.assert_array_equal(fractals.domainGet(self.muMixed1, 8), blackDomain)
        mixedDomain0 = np.matrix([[0,255,255,255],[255,255,255,255],[255,255,255,255],[255,255,255,255]])
        np.testing.assert_array_equal(fractals.domainGet(self.muMixed2, 0), mixedDomain0)

    def testRangeGet(self):
        whiteRange = np.matrix([[255,255],[255,255]])
        np.testing.assert_array_equal(fractals.rangeGet(self.muWhite, 0), whiteRange)
        np.testing.assert_array_equal(fractals.rangeGet(self.muWhite, 15), whiteRange)
        blackRange = np.matrix([[0,0],[0,0]])
        np.testing.assert_array_equal(fractals.rangeGet(self.muBlack, 0), blackRange)
        np.testing.assert_array_equal(fractals.rangeGet(self.muBlack, 15), blackRange)
        np.testing.assert_array_equal(fractals.rangeGet(self.muMixed1, 0), whiteRange)
        np.testing.assert_array_equal(fractals.rangeGet(self.muMixed1, 15), blackRange)
        mixedRange0 = np.matrix([[0,255],[255,255]])
        np.testing.assert_array_equal(fractals.rangeGet(self.muMixed2, 0), mixedRange0)

    def testTransform(self):
        newWhite = np.matrix([[255,255],[255,255]])
        np.testing.assert_array_equal(fractals.transform(self.muWhite, 0), newWhite)
        newBlack = np.matrix([[0,0],[0,0]])
        np.testing.assert_array_equal(fractals.transform(self.muBlack, 0), newBlack)
        np.testing.assert_array_equal(fractals.transform(self.muMixed1, 0), newWhite)
        np.testing.assert_array_equal(fractals.transform(self.muMixed1, 8), newBlack)
        newMixed0 = np.matrix([[191,255],[255,255]])
        np.testing.assert_array_equal(fractals.transform(self.muMixed2, 0), newMixed0)
        np.testing.assert_array_equal(fractals.transform(self.muMixed2, 8), newBlack)

    def testVariance(self):
        np.testing.assert_array_equal(fractals.variance(self.muWhite, 0, 4), 0)
        np.testing.assert_array_equal(fractals.variance(self.muWhite, 0, 2), 0)
        np.testing.assert_array_equal(fractals.variance(self.muBlack, 0, 4), 0)
        np.testing.assert_array_equal(fractals.variance(self.muBlack, 0, 2), 0)
        np.testing.assert_array_equal(fractals.variance(self.muMixed1, 0, 4), 0)
        np.testing.assert_array_equal(fractals.variance(self.muMixed1, 0, 2), 0)
        np.testing.assert_array_equal(fractals.variance(self.muMixed2, 0, 2), 12192.1875) #variance at the split

    def testCreateDomainDictionary(self):
        noVariance = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0}
        mixedVariance = {0:0, 1:16256.25, 2:0, 3:0, 4:16256.25, 5:0, 6:0, 7:16256.25, 8:0}
        mixedVariance2 = {0:3810.05859375, 1:16002.24609375, 2:3810.05859375, 3:0, 4:16002.24609375, 5:3810.05859375, 6:0, 7:16256.25, 8:0}
        self.assertEqual(fractals.createDomainDictionary(self.muWhite), noVariance)
        self.assertEqual(fractals.createDomainDictionary(self.muBlack), noVariance)
        self.assertEqual(fractals.createDomainDictionary(self.muMixed1), mixedVariance)
        self.assertEqual(fractals.createDomainDictionary(self.muMixed2), mixedVariance2)

    def testCreateRangeDictionary(self):
        noVariance = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0}
        mixedVariance = {0:12192.1875, 1:0, 2:0, 3:0, 4:0, 5:0, 6:12192.1875, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0}
        self.assertEqual(fractals.createRangeDictionary(self.muWhite), noVariance)
        self.assertEqual(fractals.createRangeDictionary(self.muBlack), noVariance)
        self.assertEqual(fractals.createRangeDictionary(self.muMixed1), noVariance)
        self.assertEqual(fractals.createRangeDictionary(self.muMixed2), mixedVariance)

    def testDomainSorting(self):
        noVariance = OrderedDict({0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0})
        mixedVariance = OrderedDict([(0, 0.0), (2, 0.0), (3, 0.0), (5, 0.0), (6, 0.0), (8, 0.0), (1, 16256.25), (4, 16256.25), (7, 16256.25)])
        mixedVariance2 = OrderedDict([(3, 0.0), (6, 0.0), (8, 0.0), (0, 3810.05859375), (2, 3810.05859375), (5, 3810.05859375), (1, 16002.24609375), (4, 16002.24609375), (7, 16256.25)])
        self.assertEqual(fractals.domainSorting(self.muWhite), noVariance)
        self.assertEqual(fractals.domainSorting(self.muBlack), noVariance)
        self.assertEqual(fractals.domainSorting(self.muMixed1), mixedVariance)
        self.assertEqual(fractals.domainSorting(self.muMixed2), mixedVariance2)

    def testRangeSorting(self):
        noVariance = OrderedDict({0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0})
        mixedVariance = OrderedDict([(1,0), (2,0), (3,0), (4,0), (5,0), (7,0), (8,0), (9,0), (10,0), (11,0), (12,0), (13,0), (14,0), (15,0), (0,12192.1875), (6,12192.1875)])
        self.assertEqual(fractals.rangeSorting(self.muWhite), noVariance)
        self.assertEqual(fractals.rangeSorting(self.muBlack), noVariance)
        self.assertEqual(fractals.rangeSorting(self.muMixed1), noVariance)
        self.assertEqual(fractals.rangeSorting(self.muMixed2), mixedVariance)

    def testGrouping(self):
        noVariance = {0:0, 1:0, 2:1, 3:1, 4:2, 5:2, 6:3, 7:3, 8:4, 9:5, 10:5, 11:6, 12:6, 13:7, 14:7, 15:8}
        self.assertEqual(fractals.grouping(self.muWhite), noVariance)
        self.assertEqual(fractals.grouping(self.muBlack), noVariance)


        
if __name__ == '__main__':
    unittest.main()

