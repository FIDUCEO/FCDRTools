import tempfile
import unittest

import os
import xarray as xr

from fiduceo.fcdr.writer.fcdr_writer import FCDRWriter


class ReadWriteTests(unittest.TestCase):
    def setUp(self):
        tempDir = tempfile.gettempdir()
        self.testDir = os.path.join(tempDir, 'fcdr')
        os.mkdir(self.testDir)

    def tearDown(self):
        if os.path.isdir(self.testDir):
            for i in os.listdir(self.testDir):
                os.remove(os.path.join(self.testDir, i))
            os.rmdir(self.testDir)

    def test_write_empty(self):
        testFile = os.path.join(self.testDir, 'delete_me.nc')
        emptyDataset = xr.Dataset()

        FCDRWriter.write(emptyDataset, testFile)

        self.assertTrue(os.path.isfile(testFile))

    def test_write_overwrite_true(self):
        testFile = os.path.join(self.testDir, 'delete_me.nc')
        emptyDataset = xr.Dataset()

        FCDRWriter.write(emptyDataset, testFile, overwrite=True)
        self.assertTrue(os.path.isfile(testFile))

        FCDRWriter.write(emptyDataset, testFile, overwrite=True)
        self.assertTrue(os.path.isfile(testFile))

    def test_write_overwrite_false(self):
        testFile = os.path.join(self.testDir, 'delete_me.nc')
        emptyDataset = xr.Dataset()

        FCDRWriter.write(emptyDataset, testFile)
        self.assertTrue(os.path.isfile(testFile))

        try:
            FCDRWriter.write(emptyDataset, testFile, overwrite=False)
            self.fail("IOError expected")
        except IOError:
            pass