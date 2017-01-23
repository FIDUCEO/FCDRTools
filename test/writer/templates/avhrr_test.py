import unittest

import numpy as np
import xarray as xr

from writer.default_data import DefaultData
from writer.templates.avhrr import AVHRR


class AVHRRTest(unittest.TestCase):
    def test_add_original_variables(self):
        ds = xr.Dataset()
        AVHRR.add_original_variables(ds, 5)

        latitude = ds.variables["latitude"]
        self.assertEqual((5, 409), latitude.shape)
        self.assertEqual(-32768.0, latitude.data[0, 0])
        self.assertEqual(-32768.0, latitude.attrs["_FillValue"])
        self.assertEqual("latitude", latitude.attrs["standard_name"])
        self.assertEqual("degrees_north", latitude.attrs["units"])

        longitude = ds.variables["longitude"]
        self.assertEqual((5, 409), longitude.shape)
        self.assertEqual(-32768.0, longitude.data[0, 1])
        self.assertEqual(-32768.0, longitude.attrs["_FillValue"])
        self.assertEqual("longitude", longitude.attrs["standard_name"])
        self.assertEqual("degrees_east", longitude.attrs["units"])

        time = ds.variables["Time"]
        self.assertEqual((5, 409), time.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), time.data[0, 2])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), time.attrs["_FillValue"])
        self.assertEqual("time", time.attrs["standard_name"])
        self.assertEqual("Acquisition time in seconds since 1970-01-01 00:00:00", time.attrs["long_name"])
        self.assertEqual("s", time.attrs["units"])

        scanline = ds.variables["scanline"]
        self.assertEqual((5,), scanline.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), scanline.data[3])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), scanline.attrs["_FillValue"])
        self.assertEqual("scanline", scanline.attrs["standard_name"])
        self.assertEqual("Level 1b line number", scanline.attrs["long_name"])
        self.assertEqual(0, scanline.attrs["valid_min"])

        sat_azimuth = ds.variables["satellite_azimuth_angle"]
        self.assertEqual((5, 409), sat_azimuth.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), sat_azimuth.data[0, 4])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), sat_azimuth.attrs["_FillValue"])
        self.assertEqual("sensor_azimuth_angle", sat_azimuth.attrs["standard_name"])
        self.assertEqual("degree", sat_azimuth.attrs["units"])

        sat_zenith = ds.variables["satellite_zenith_angle"]
        self.assertEqual((5, 409), sat_zenith.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), sat_zenith.data[0, 5])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), sat_zenith.attrs["_FillValue"])
        self.assertEqual("sensor_zenith_angle", sat_zenith.attrs["standard_name"])
        self.assertEqual(0.0, sat_zenith.attrs["add_offset"])
        self.assertEqual(0.01, sat_zenith.attrs["scale_factor"])
        self.assertEqual("degree", sat_zenith.attrs["units"])
        self.assertEqual(9000, sat_zenith.attrs["valid_max"])
        self.assertEqual(0, sat_zenith.attrs["valid_min"])

        sol_azimuth = ds.variables["solar_azimuth_angle"]
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), sol_azimuth.data[0, 6])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), sol_azimuth.attrs["_FillValue"])
        self.assertEqual("solar_azimuth_angle", sol_azimuth.attrs["standard_name"])
        self.assertEqual("degree", sol_azimuth.attrs["units"])

        sol_zenith = ds.variables["solar_zenith_angle"]
        self.assertEqual((5, 409), sol_zenith.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), sol_zenith.data[0, 7])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), sol_zenith.attrs["_FillValue"])
        self.assertEqual("solar_zenith_angle", sol_zenith.attrs["standard_name"])
        self.assertEqual(0.0, sol_zenith.attrs["add_offset"])
        self.assertEqual(0.01, sol_zenith.attrs["scale_factor"])
        self.assertEqual("degree", sol_zenith.attrs["units"])
        self.assertEqual(18000, sol_zenith.attrs["valid_max"])
        self.assertEqual(0, sol_zenith.attrs["valid_min"])

        ch1_bt = ds.variables["Ch1_Bt"]
        self.assert_correct_channel_refl_variable(ch1_bt, "Channel 1 Reflectance")

        ch2_bt = ds.variables["Ch2_Bt"]
        self.assert_correct_channel_refl_variable(ch2_bt, "Channel 2 Reflectance")

        ch3a_bt = ds.variables["Ch3a_Bt"]
        self.assert_correct_channel_refl_variable(ch3a_bt, "Channel 3a Reflectance")

        ch3b_bt = ds.variables["Ch3b_Bt"]
        self.assert_correct_channel_bt_variable(ch3b_bt, "Channel 3b Brightness Temperature")

        ch4_bt = ds.variables["Ch4_Bt"]
        self.assert_correct_channel_bt_variable(ch4_bt, "Channel 4 Brightness Temperature")

        ch5_bt = ds.variables["Ch5_Bt"]
        self.assert_correct_channel_bt_variable(ch5_bt, "Channel 5 Brightness Temperature")

        ict_temp = ds.variables["T_ICT"]
        self.assertEqual((5,), ict_temp.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), ict_temp.data[1])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), ict_temp.attrs["_FillValue"])
        self.assertEqual("Temperature of the internal calibration target", ict_temp.attrs["standard_name"])
        self.assertEqual(273.15, ict_temp.attrs["add_offset"])
        self.assertEqual(0.01, ict_temp.attrs["scale_factor"])
        self.assertEqual("kelvin", ict_temp.attrs["units"])
        self.assertEqual(10000, ict_temp.attrs["valid_max"])
        self.assertEqual(-20000, ict_temp.attrs["valid_min"])

    def assert_correct_channel_refl_variable(self, variable, long_name):
        self.assertEqual((5, 409), variable.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), variable.data[0, 8])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), variable.attrs["_FillValue"])
        self.assertEqual("toa_reflectance", variable.attrs["standard_name"])
        self.assertEqual(long_name, variable.attrs["long_name"])
        self.assertEqual(0.0, variable.attrs["add_offset"])
        self.assertEqual(1e-4, variable.attrs["scale_factor"])
        self.assertEqual("percent", variable.attrs["units"])
        self.assertEqual(15000, variable.attrs["valid_max"])
        self.assertEqual(0, variable.attrs["valid_min"])

    def assert_correct_channel_bt_variable(self, variable, long_name):
        self.assertEqual((5, 409), variable.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), variable.data[0, 8])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), variable.attrs["_FillValue"])
        self.assertEqual("toa_brightness_temperature", variable.attrs["standard_name"])
        self.assertEqual(long_name, variable.attrs["long_name"])
        self.assertEqual(273.15, variable.attrs["add_offset"])
        self.assertEqual(0.01, variable.attrs["scale_factor"])
        self.assertEqual("kelvin", variable.attrs["units"])
        self.assertEqual(10000, variable.attrs["valid_max"])
        self.assertEqual(-20000, variable.attrs["valid_min"])