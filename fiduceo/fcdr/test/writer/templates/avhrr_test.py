import unittest

import numpy as np
import xarray as xr

from fiduceo.fcdr.test.writer.templates.assertions import Assertions
from fiduceo.fcdr.writer.default_data import DefaultData
from fiduceo.fcdr.writer.templates.avhrr import AVHRR


class AVHRRTest(unittest.TestCase):
    def test_add_original_variables(self):
        ds = xr.Dataset()
        AVHRR.add_original_variables(ds, 5)

        Assertions.assert_geolocation_variables(self, ds, 409, 5)
        Assertions.assert_quality_flags(self, ds, 409, 5)

        time = ds.variables["time"]
        self.assertEqual((5,), time.shape)
        self.assertTrue(np.isnan(time.data[2]))
        self.assertTrue(np.isnan(time.attrs["_FillValue"]))
        self.assertEqual("time", time.attrs["standard_name"])
        self.assertEqual("Acquisition time in seconds since 1970-01-01 00:00:00", time.attrs["long_name"])
        self.assertEqual("s", time.attrs["units"])

        sat_zenith = ds.variables["satellite_zenith_angle"]
        self.assertEqual((5, 409), sat_zenith.shape)
        self.assertTrue(np.isnan(sat_zenith.data[0, 5]))
        self.assertEqual("sensor_zenith_angle", sat_zenith.attrs["standard_name"])
        self.assertEqual("degree", sat_zenith.attrs["units"])
        self.assertEqual(9000, sat_zenith.attrs["valid_max"])
        self.assertEqual(0, sat_zenith.attrs["valid_min"])
        self.assertEqual(np.int16, sat_zenith.encoding['dtype'])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), sat_zenith.encoding['_FillValue'])
        self.assertEqual(0.01, sat_zenith.encoding['scale_factor'])
        self.assertEqual(0.0, sat_zenith.encoding['add_offset'])

        sol_zenith = ds.variables["solar_zenith_angle"]
        self.assertEqual((5, 409), sol_zenith.shape)
        self.assertTrue(np.isnan(sol_zenith.data[0, 7]))
        self.assertEqual("solar_zenith_angle", sol_zenith.attrs["standard_name"])
        self.assertEqual("degree", sol_zenith.attrs["units"])
        self.assertEqual(18000, sol_zenith.attrs["valid_max"])
        self.assertEqual(0, sol_zenith.attrs["valid_min"])
        self.assertEqual(np.int16, sol_zenith.encoding['dtype'])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), sol_zenith.encoding['_FillValue'])
        self.assertEqual(0.01, sol_zenith.encoding['scale_factor'])
        self.assertEqual(0.0, sol_zenith.encoding['add_offset'])

        ch1_ref = ds.variables["Ch1_Ref"]
        self._assert_correct_refl_variable(ch1_ref, "Channel 1 Reflectance")

        ch2_ref = ds.variables["Ch2_Ref"]
        self._assert_correct_refl_variable(ch2_ref, "Channel 2 Reflectance")

        ch3a_ref = ds.variables["Ch3a_Ref"]
        self._assert_correct_refl_variable(ch3a_ref, "Channel 3a Reflectance")

        ch3b_bt = ds.variables["Ch3b_Bt"]
        self._assert_correct_bt_variable(ch3b_bt, "Channel 3b Brightness Temperature")

        ch4_bt = ds.variables["Ch4_Bt"]
        self._assert_correct_bt_variable(ch4_bt, "Channel 4 Brightness Temperature")

        ch5_bt = ds.variables["Ch5_Bt"]
        self._assert_correct_bt_variable(ch5_bt, "Channel 5 Brightness Temperature")

    def test_get_swath_width(self):
        self.assertEqual(409, AVHRR.get_swath_width())

    def test_add_easy_fcdr_variables(self):
        ds = xr.Dataset()
        AVHRR.add_easy_fcdr_variables(ds, 5)

        self._assert_correct_refl_uncertainty_variable(ds, "u_independent_Ch1", long_name="independent uncertainty per pixel for channel 1", units="1")
        self._assert_correct_refl_uncertainty_variable(ds, "u_structured_Ch1", long_name="structured uncertainty per pixel for channel 1", units="1")
        self._assert_correct_refl_uncertainty_variable(ds, "u_independent_Ch2", long_name="independent uncertainty per pixel for channel 2", units="1")
        self._assert_correct_refl_uncertainty_variable(ds, "u_structured_Ch2", long_name="structured uncertainty per pixel for channel 2", units="1")
        self._assert_correct_refl_uncertainty_variable(ds, "u_independent_Ch3a", long_name="independent uncertainty per pixel for channel 3a", units="1")
        self._assert_correct_refl_uncertainty_variable(ds, "u_structured_Ch3a", long_name="structured uncertainty per pixel for channel 3a", units="1")

        self._assert_correct_bt_uncertainty_variable(ds, "u_independent_Ch3b", long_name="independent uncertainty per pixel for channel 3b")
        self._assert_correct_bt_uncertainty_variable(ds, "u_structured_Ch3b", long_name="structured uncertainty per pixel for channel 3b")
        self._assert_correct_bt_uncertainty_variable(ds, "u_independent_Ch4", long_name="independent uncertainty per pixel for channel 4")
        self._assert_correct_bt_uncertainty_variable(ds, "u_structured_Ch4", long_name="structured uncertainty per pixel for channel 4")
        self._assert_correct_bt_uncertainty_variable(ds, "u_independent_Ch5", long_name="independent uncertainty per pixel for channel 5")
        self._assert_correct_bt_uncertainty_variable(ds, "u_structured_Ch5", long_name="structured uncertainty per pixel for channel 5")

    def test_add_full_fcdr_variables(self):
        ds = xr.Dataset()
        AVHRR.add_full_fcdr_variables(ds, 5)

        u_latitude = ds.variables["u_latitude"]
        self.assertEqual((5, 409), u_latitude.shape)
        self.assertTrue(np.isnan(u_latitude.data[0, 34]))
        self.assertTrue(np.isnan(u_latitude.attrs["_FillValue"]))
        self.assertEqual("uncertainty of latitude", u_latitude.attrs["long_name"])
        self.assertEqual("degree", u_latitude.attrs["units"])

        u_longitude = ds.variables["u_longitude"]
        self.assertEqual((5, 409), u_longitude.shape)
        self.assertTrue(np.isnan(u_longitude.data[1, 35]))
        self.assertTrue(np.isnan(u_longitude.attrs["_FillValue"]))
        self.assertEqual("uncertainty of longitude", u_longitude.attrs["long_name"])
        self.assertEqual("degree", u_longitude.attrs["units"])

        u_time = ds.variables["u_time"]
        self.assertEqual((5,), u_time.shape)
        self.assertTrue(np.isnan(u_time.data[1]))
        self.assertTrue(np.isnan(u_time.attrs["_FillValue"]))
        self.assertEqual("uncertainty of acquisition time", u_time.attrs["long_name"])
        self.assertEqual("s", u_time.attrs["units"])

        u_sat_azimuth = ds.variables["u_satellite_azimuth_angle"]
        self.assertEqual((5, 409), u_sat_azimuth.shape)
        self.assertTrue(np.isnan(u_sat_azimuth.data[2, 36]))
        self.assertTrue(np.isnan(u_sat_azimuth.attrs["_FillValue"]))
        self.assertEqual("uncertainty of satellite azimuth angle", u_sat_azimuth.attrs["long_name"])
        self.assertEqual("degree", u_sat_azimuth.attrs["units"])

        u_sat_zenith = ds.variables["u_satellite_zenith_angle"]
        self.assertEqual((5, 409), u_sat_zenith.shape)
        self.assertTrue(np.isnan(u_sat_zenith.data[2, 36]))
        self.assertTrue(np.isnan(u_sat_zenith.attrs["_FillValue"]))
        self.assertEqual("uncertainty of satellite zenith angle", u_sat_zenith.attrs["long_name"])
        self.assertEqual("degree", u_sat_zenith.attrs["units"])

        u_sol_azimuth = ds.variables["u_solar_azimuth_angle"]
        self.assertEqual((5, 409), u_sol_azimuth.shape)
        self.assertTrue(np.isnan(u_sol_azimuth.data[2, 36]))
        self.assertTrue(np.isnan(u_sol_azimuth.attrs["_FillValue"]))
        self.assertEqual("uncertainty of solar azimuth angle", u_sol_azimuth.attrs["long_name"])
        self.assertEqual("degree", u_sol_azimuth.attrs["units"])

        u_sol_zenith = ds.variables["u_solar_zenith_angle"]
        self.assertEqual((5, 409), u_sol_zenith.shape)
        self.assertTrue(np.isnan(u_sol_zenith.data[2, 36]))
        self.assertTrue(np.isnan(u_sol_zenith.attrs["_FillValue"]))
        self.assertEqual("uncertainty of solar zenith angle", u_sol_zenith.attrs["long_name"])
        self.assertEqual("degree", u_sol_zenith.attrs["units"])

        prt_c = ds.variables["PRT_C"]
        self.assertEqual((5, 3), prt_c.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), prt_c.data[3, 2])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), prt_c.attrs["_FillValue"])
        self.assertEqual("Prt counts", prt_c.attrs["long_name"])
        self.assertEqual("count", prt_c.attrs["units"])

        u_prt = ds.variables["u_prt"]
        self.assertEqual((5, 3), u_prt.shape)
        self.assertTrue(np.isnan(u_prt.data[4, 0]))
        self.assertTrue(np.isnan(u_prt.attrs["_FillValue"]))
        self.assertEqual("Uncertainty on the PRT counts", u_prt.attrs["long_name"])
        self.assertEqual("count", u_prt.attrs["units"])
        self.assertEqual("rectangle_absolute", u_prt.attrs["pixel_correlation_form"])
        self.assertEqual("pixel", u_prt.attrs["pixel_correlation_units"])
        self.assertEqual([-np.inf, np.inf], u_prt.attrs["pixel_correlation_scales"])
        self.assertEqual("rectangle_absolute", u_prt.attrs["scan_correlation_form"])
        self.assertEqual("line", u_prt.attrs["scan_correlation_units"])
        self.assertEqual([-np.inf, np.inf], u_prt.attrs["scan_correlation_scales"])
        self.assertEqual("rectangle", u_prt.attrs["pdf_shape"])
        self.assertEqual(0.1, u_prt.attrs["pdf_parameter"])

        r_ict = ds.variables["R_ICT"]
        self.assertEqual((5, 3), r_ict.shape)
        self.assertTrue(np.isnan(r_ict.data[0, 1]))
        self.assertTrue(np.isnan(r_ict.attrs["_FillValue"]))
        self.assertEqual("Radiance of the PRT", r_ict.attrs["long_name"])
        self.assertEqual("mW m^-2 sr^-1 cm", r_ict.attrs["units"])

        t_instr = ds.variables["T_instr"]
        self.assertEqual((), t_instr.shape)
        self.assertTrue(np.isnan(t_instr.data))
        self.assertTrue(np.isnan(t_instr.attrs["_FillValue"]))
        self.assertEqual("Instrument temperature", t_instr.attrs["long_name"])
        self.assertEqual("K", t_instr.attrs["units"])

        self._assert_correct_counts_variable(ds, "Ch1_Csp", "Ch1 Space counts")
        self._assert_correct_counts_variable(ds, "Ch2_Csp", "Ch2 Space counts")
        self._assert_correct_counts_variable(ds, "Ch3a_Csp", "Ch3a Space counts")
        self._assert_correct_counts_variable(ds, "Ch3b_Csp", "Ch3b Space counts")
        self._assert_correct_counts_variable(ds, "Ch4_Csp", "Ch4 Space counts")
        self._assert_correct_counts_variable(ds, "Ch5_Csp", "Ch5 Space counts")

        self._assert_correct_counts_variable(ds, "Ch3b_Cict", "Ch3b ICT counts")
        self._assert_correct_counts_variable(ds, "Ch4_Cict", "Ch4 ICT counts")
        self._assert_correct_counts_variable(ds, "Ch5_Cict", "Ch5 ICT counts")

        self._assert_correct_counts_variable(ds, "Ch1_Ce", "Ch1 Earth counts")
        self._assert_correct_counts_variable(ds, "Ch2_Ce", "Ch2 Earth counts")
        self._assert_correct_counts_variable(ds, "Ch3a_Ce", "Ch3a Earth counts")
        self._assert_correct_counts_variable(ds, "Ch3b_Ce", "Ch3b Earth counts")
        self._assert_correct_counts_variable(ds, "Ch4_Ce", "Ch4 Earth counts")
        self._assert_correct_counts_variable(ds, "Ch5_Ce", "Ch5 Earth counts")

        self._assert_correct_counts_uncertainty_variable(ds, "Ch1_u_Csp", "Ch1 Uncertainty on space counts")
        self._assert_counts_correlation(ds, "Ch1_u_Csp")
        self._assert_correct_counts_uncertainty_variable(ds, "Ch2_u_Csp", "Ch2 Uncertainty on space counts")
        self._assert_counts_correlation(ds, "Ch2_u_Csp")
        self._assert_correct_counts_uncertainty_variable(ds, "Ch3a_u_Csp", "Ch3a Uncertainty on space counts")
        self._assert_counts_correlation(ds, "Ch3a_u_Csp")
        self._assert_correct_counts_uncertainty_variable(ds, "Ch3b_u_Csp", "Ch3b Uncertainty on space counts")
        self._assert_counts_correlation(ds, "Ch3b_u_Csp")
        self._assert_correct_counts_uncertainty_variable(ds, "Ch4_u_Csp", "Ch4 Uncertainty on space counts")
        self._assert_counts_correlation(ds, "Ch4_u_Csp")
        self._assert_correct_counts_uncertainty_variable(ds, "Ch5_u_Csp", "Ch5 Uncertainty on space counts")
        self._assert_counts_correlation(ds, "Ch4_u_Csp")

        self._assert_correct_counts_uncertainty_variable(ds, "Ch3b_u_Cict", "Ch3b Uncertainty on ICT counts")
        self._assert_counts_correlation(ds, "Ch3b_u_Cict")
        self._assert_correct_counts_uncertainty_variable(ds, "Ch4_u_Cict", "Ch4 Uncertainty on ICT counts")
        self._assert_counts_correlation(ds, "Ch4_u_Cict")
        self._assert_correct_counts_uncertainty_variable(ds, "Ch5_u_Cict", "Ch5 Uncertainty on ICT counts")
        self._assert_counts_correlation(ds, "Ch5_u_Cict")

        self._assert_correct_counts_uncertainty_variable(ds, "Ch1_u_Ce", "Ch1 Uncertainty on earth counts")
        self._assert_earth_counts_pdf(ds, "Ch1_u_Ce")
        self._assert_correct_counts_uncertainty_variable(ds, "Ch2_u_Ce", "Ch2 Uncertainty on earth counts")
        self._assert_earth_counts_pdf(ds, "Ch2_u_Ce")
        self._assert_correct_counts_uncertainty_variable(ds, "Ch3a_u_Ce", "Ch3a Uncertainty on earth counts")
        self._assert_earth_counts_pdf(ds, "Ch3a_u_Ce")
        self._assert_correct_counts_uncertainty_variable(ds, "Ch3b_u_Ce", "Ch3b Uncertainty on earth counts")
        self._assert_earth_counts_pdf(ds, "Ch3b_u_Ce")
        self._assert_correct_counts_uncertainty_variable(ds, "Ch4_u_Ce", "Ch4 Uncertainty on earth counts")
        self._assert_earth_counts_pdf(ds, "Ch4_u_Ce")
        self._assert_correct_counts_uncertainty_variable(ds, "Ch5_u_Ce", "Ch5 Uncertainty on earth counts")
        self._assert_earth_counts_pdf(ds, "Ch5_u_Ce")

        self._assert_correct_refl_uncertainty_variable(ds, "Ch1_u_Refl", long_name="Ch1 Total uncertainty on toa reflectance", units="1")
        self._assert_correct_refl_uncertainty_variable(ds, "Ch2_u_Refl", long_name="Ch2 Total uncertainty on toa reflectance", units="1")
        self._assert_correct_refl_uncertainty_variable(ds, "Ch3a_u_Refl", long_name="Ch3a Total uncertainty on toa reflectance", units="1")

        self._assert_correct_bt_uncertainty_variable(ds, "Ch3b_u_Bt", long_name="Ch3b Total uncertainty on brightness temperature")
        self._assert_correct_bt_uncertainty_variable(ds, "Ch4_u_Bt", long_name="Ch4 Total uncertainty on brightness temperature")
        self._assert_correct_bt_uncertainty_variable(ds, "Ch5_u_Bt", long_name="Ch5 Total uncertainty on brightness temperature")

        self._assert_correct_bt_uncertainty_variable(ds, "Ch3b_ur_Bt", long_name="Ch3b Random uncertainty on brightness temperature")
        self._assert_correct_bt_uncertainty_variable(ds, "Ch4_ur_Bt", long_name="Ch4 Random uncertainty on brightness temperature")
        self._assert_correct_bt_uncertainty_variable(ds, "Ch5_ur_Bt", long_name="Ch5 Random uncertainty on brightness temperature")

        self._assert_correct_bt_uncertainty_variable(ds, "Ch3b_us_Bt", long_name="Ch3b Systematic uncertainty on brightness temperature")
        self._assert_correct_bt_uncertainty_variable(ds, "Ch4_us_Bt", long_name="Ch4 Systematic uncertainty on brightness temperature")
        self._assert_correct_bt_uncertainty_variable(ds, "Ch5_us_Bt", long_name="Ch5 Systematic uncertainty on brightness temperature")

    def _assert_earth_counts_pdf(self, ds, name):
        variable = ds.variables[name]
        self.assertEqual("digitised_gaussian", variable.attrs["pdf_shape"])

    def _assert_counts_correlation(self, ds, name):
        variable = ds.variables[name]
        self.assertEqual("rectangle_absolute", variable.attrs["pixel_correlation_form"])
        self.assertEqual("pixel", variable.attrs["pixel_correlation_units"])
        self.assertEqual([-np.inf, np.inf], variable.attrs["pixel_correlation_scales"])
        self.assertEqual("triangle_relative", variable.attrs["scan_correlation_form"])
        self.assertEqual("line", variable.attrs["scan_correlation_units"])
        self.assertEqual([-25, 25], variable.attrs["scan_correlation_scales"])
        self.assertEqual("digitised_gaussian", variable.attrs["pdf_shape"])

    def _assert_correct_counts_variable(self, ds, name, long_name):
        variable = ds.variables[name]
        self.assertEqual((5, 409), variable.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int32), variable.data[3, 306])
        self.assertEqual(DefaultData.get_default_fill_value(np.int32), variable.attrs["_FillValue"])
        self.assertEqual(long_name, variable.attrs["long_name"])
        self.assertEqual("count", variable.attrs["units"])

    def _assert_correct_counts_uncertainty_variable(self, ds, name, standard_name):
        variable = ds.variables[name]
        self.assertEqual((5, 409), variable.shape)
        self.assertTrue(np.isnan(variable.data[4, 307]))
        self.assertTrue(np.isnan(variable.attrs["_FillValue"]))
        self.assertEqual(standard_name, variable.attrs["long_name"])
        self.assertEqual("count", variable.attrs["units"])

    def _assert_correct_refl_uncertainty_variable(self, ds, name, standard_name=None, units=None, long_name=None):
        variable = ds.variables[name]
        self.assertEqual((5, 409), variable.shape)
        self.assertTrue(np.isnan(variable.data[4, 307]))
        self.assertEqual(-32767, variable.encoding["_FillValue"])
        # @todo 2 tb/tb add checks for encoding, valid min and max 2017-10-19
        if standard_name is not None:
            self.assertEqual(standard_name, variable.attrs["standard_name"])

        if long_name is not None:
            self.assertEqual(long_name, variable.attrs["long_name"])

        if units is not None:
            self.assertEqual(units, variable.attrs["units"])

    def _assert_correct_bt_uncertainty_variable(self, ds, name, standard_name=None, long_name=None):
        variable = ds.variables[name]
        self.assertEqual((5, 409), variable.shape)
        self.assertTrue(np.isnan(variable.data[4, 307]))
        self.assertEqual(-32767, variable.encoding["_FillValue"])

        if standard_name is not None:
            self.assertEqual(standard_name, variable.attrs["standard_name"])

        # if long_name is not None:
        #   self.assertEqual(long_name, variable.attrs["long_name"])

        self.assertEqual("K", variable.attrs["units"])

    def _assert_correct_refl_variable(self, variable, long_name):
        self.assertEqual((5, 409), variable.shape)
        self.assertTrue(np.isnan(variable.data[0, 8]))
        self.assertEqual("toa_reflectance", variable.attrs["standard_name"])
        self.assertEqual(long_name, variable.attrs["long_name"])
        self.assertEqual("1", variable.attrs["units"])
        self.assertEqual(np.int16, variable.encoding['dtype'])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), variable.encoding['_FillValue'])
        self.assertEqual(0.0001, variable.encoding['scale_factor'])
        self.assertEqual(0.0, variable.encoding['add_offset'])
        self.assertEqual(15000, variable.attrs["valid_max"])
        self.assertEqual(0, variable.attrs["valid_min"])

    def _assert_correct_bt_variable(self, variable, long_name):
        self.assertEqual((5, 409), variable.shape)
        self.assertTrue(np.isnan(variable.data[0, 8]))
        self.assertEqual("toa_brightness_temperature", variable.attrs["standard_name"])
        self.assertEqual(long_name, variable.attrs["long_name"])
        self.assertEqual(np.int16, variable.encoding['dtype'])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), variable.encoding['_FillValue'])
        self.assertEqual(0.01, variable.encoding['scale_factor'])
        self.assertEqual(273.15, variable.encoding['add_offset'])
        self.assertEqual("K", variable.attrs["units"])
        self.assertEqual(10000, variable.attrs["valid_max"])
        self.assertEqual(-20000, variable.attrs["valid_min"])
