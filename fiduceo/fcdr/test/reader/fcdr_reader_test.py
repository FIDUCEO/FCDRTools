import unittest as ut
import xarray as xr
import numpy as np
import fiduceo.fcdr.test.test_utils as tu
from fiduceo.fcdr.reader.fcdr_reader import FCDRReader


class FCDR_Reader_Test(ut.TestCase):

    def test_adding_2_three_dimensional_variables(self):
        ds = xr.Dataset()
        ds['a'] = self._get_three_dim_variable()
        ds['b'] = self._get_three_dim_variable()
        v_var = self._create_virtual_variable("a + b")
        ds["v_var"] = v_var

        FCDRReader._prepare_virtual_variables(ds)
        ds['v_var'].load()

        self.assertTrue('v_var' in ds)
        virtual_loaded = ds['v_var']
        self.assertEqual(('z', 'y', 'x'), virtual_loaded.dims)
        self.assertEqual((4, 2, 3), virtual_loaded.shape)
        self.assertEqual(v_var.attrs, virtual_loaded.attrs)

        expected = np.asarray([[[2, 4, 6], [2.2, 4.2, 6.2]],
                               [[22, 24, 26], [22.2, 24.2, 26.2]],
                               [[42, 44, 46], [42.2, 44.2, 46.2]],
                               [[62, 64, 66], [62.2, 64.2, 66.2]]])
        actual = ds['v_var'].values

        self.assertEqual("<type 'numpy.ndarray'>", str(type(actual)))
        self.assertEqual(type(expected), type(actual))
        tu.assert_array_equals_with_index_error_message(self, expected, actual)


    def test_adding_1_three_dimensional_variable_and_1_two_dimensional_variable(self):
        ds = xr.Dataset()
        ds['a'] = self._get_three_dim_variable()
        ds['b'] = self._get_two_dim_variable()
        v_var = self._create_virtual_variable("a + b")
        ds["v_var"] = v_var

        FCDRReader._prepare_virtual_variables(ds)
        ds['v_var'].load()

        self.assertTrue('v_var' in ds)
        virtual_loaded = ds['v_var']
        self.assertEqual(('z', 'y', 'x'), virtual_loaded.dims)
        self.assertEqual((4, 2, 3), virtual_loaded.shape)
        self.assertEqual(v_var.attrs, virtual_loaded.attrs)

        expected = np.asarray([[[2, 4, 6], [5.1, 7.1, 9.1]],
                               [[12, 14, 16], [15.1, 17.1, 19.1]],
                               [[22, 24, 26], [25.1, 27.1, 29.1]],
                               [[32, 34, 36], [35.1, 37.1, 39.1]]])
        actual = virtual_loaded.values

        self.assertEqual("<type 'numpy.ndarray'>", str(type(actual)))
        self.assertEqual(type(expected), type(actual))
        tu.assert_array_equals_with_index_error_message(self, expected, actual)


    def test_adding_1_three_dimensional_variable_and_1_one_dimensional_variable(self):
        ds = xr.Dataset()
        ds['a'] = self._get_three_dim_variable()
        ds['b'] = self._get_one_dim_variable()
        v_var = self._create_virtual_variable("a + b")
        ds["v_var"] = v_var

        FCDRReader._prepare_virtual_variables(ds)
        ds['v_var'].load()

        self.assertTrue('v_var' in ds)
        virtual_loaded = ds['v_var']
        self.assertEqual(('z', 'y', 'x'), virtual_loaded.dims)
        self.assertEqual((4, 2, 3), virtual_loaded.shape)
        self.assertEqual(v_var.attrs, virtual_loaded.attrs)

        expected = np.asarray([[[2.3, 4.3, 6.3], [2.4, 4.4, 6.4]],
                               [[12.3, 14.3, 16.3], [12.4, 14.4, 16.4]],
                               [[22.3, 24.3, 26.3], [22.4, 24.4, 26.4]],
                               [[32.3, 34.3, 36.3], [32.4, 34.4, 36.4]]])
        actual = virtual_loaded.values

        self.assertEqual("<type 'numpy.ndarray'>", str(type(actual)))
        self.assertEqual(type(expected), type(actual))
        tu.assert_array_equals_with_index_error_message(self, expected, actual)


    def test_prepare_virtual_variables(self):
        ds = xr.Dataset()
        ds['a'] = self._get_three_dim_variable()
        ds['b'] = self._get_two_dim_variable()
        ds["v_var"] = self._create_virtual_variable("a + b")

        self.assertEqual(3, len(ds.data_vars))
        FCDRReader._prepare_virtual_variables(ds)
        self.assertEqual(3, len(ds.data_vars))
        expected = np.full([1], 1)
        actual = ds['v_var'].values
        self.assertEqual(type(expected), type(actual))
        self.assertEqual(len(expected), len(actual))


    def _get_three_dim_variable(self):
        return xr.Variable(['z', 'y', 'x'], np.asarray(
            [[[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]],
             [[11.0, 12.0, 13.0], [11.1, 12.1, 13.1]],
             [[21.0, 22.0, 23.0], [21.1, 22.1, 23.1]],
             [[31.0, 32.0, 33.0], [31.1, 32.1, 33.1]]]))


    def _get_two_dim_variable(self):
        return xr.Variable(['y', 'x'], np.asarray(
            [[1, 2, 3], [4, 5, 6]]))


    def _get_one_dim_variable(self):
        return xr.Variable(['x'], np.asarray(
            [1.3, 2.3, 3.3]))


    def _create_virtual_variable(self, expression):
        v_var = xr.Variable(["virtual"], np.full([1], 1))
        v_var.attrs["virtual"] = "true"
        v_var.attrs["expression"] = expression
        return v_var


