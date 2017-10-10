import numpy as np
from xarray import Variable

from fiduceo.fcdr.writer.correlation import Correlation as corr
from fiduceo.fcdr.writer.default_data import DefaultData
from fiduceo.fcdr.writer.templates.templateutil import TemplateUtil as tu

SRF_VIS_DIMENSION = "srf_size_vis"
SRF_IR_WV_DIMENSION = "srf_size_ir_wv"
IR_X_DIMENSION = "x_ir_wv"
IR_Y_DIMENSION = "y_ir_wv"

SRF_SIZE_VIS = 1011
SRF_SIZE_IR_WV = 1011
SOL_IRR_SIZE = 24

scale_u_lat=1.5E-05#7.62939E-06

TIME_FILL_VALUE = -32768


class MVIRI:
    FULL_DIMENSION = 5000
    IR_DIMENSION = 2500

    IR_X_DIMENSION = "x_ir_wv"
    IR_Y_DIMENSION = "y_ir_wv"

    @staticmethod
    def add_original_variables(dataset, height):
        # height is ignored - supplied just for interface compatibility tb 2017-02-05
        # binary flag
        default_array = DefaultData.create_default_array(FULL_DIMENSION, FULL_DIMENSION, np.uint8)
        variable = Variable(["y", "x"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.uint8))
        variable.attrs["standard_name"] = "quality_pixel_bitmask"
        variable.attrs["long_name"] = "bitflags for each pixel to indicate quality [all bits = 0 -> use, any bit = 1 -> do not use]in MSB0 order: [bit 0: geolocation, bit1: timing, bit2: calibration, bit3: radiometer]"
        dataset["quality_pixel_bitmask"] = variable
        
        # time
        default_array = DefaultData.create_default_array(IR_DIMENSION, IR_DIMENSION, np.uint32)
        variable = Variable([IR_Y_DIMENSION, IR_X_DIMENSION], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.uint32))
        variable.attrs["standard_name"] = "time"
        variable.attrs["long_name"] = "Acquisition time of pixel"
        tu.add_units(variable, "seconds since 1970-01-01 00:00:00")
        tu.add_offset(variable, TIME_FILL_VALUE)
        dataset["time"] = variable

        dataset["solar_azimuth_angle"] = MVIRI._create_angle_variable_int(0.005493164, standard_name="solar_azimuth_angle", unsigned=True)
        dataset["solar_zenith_angle"] = MVIRI._create_angle_variable_int(0.005493248, standard_name="solar_zenith_angle")

        # count_ir
        default_array = DefaultData.create_default_array(MVIRI.IR_DIMENSION, MVIRI.IR_DIMENSION, np.uint8)
        variable = Variable([IR_Y_DIMENSION, IR_X_DIMENSION], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.uint8))
        variable.attrs["long_name"] = "Infrared Image Counts"
        tu.add_units(variable, "count")
        dataset["count_ir"] = variable

        # count_wv
        default_array = DefaultData.create_default_array(MVIRI.IR_DIMENSION, MVIRI.IR_DIMENSION, np.uint8)
        variable = Variable([IR_Y_DIMENSION, IR_X_DIMENSION], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.uint8))
        variable.attrs["long_name"] = "WV Image Counts"
        tu.add_units(variable, "count")
        dataset["count_wv"] = variable

        # distance_sun_earth
        dataset["distance_sun_earth"] = tu.create_scalar_float_variable(long_name="Sun-Earth distance", units="au")

        # solar_irradiance_vis
        dataset["solar_irradiance_vis"] = tu.create_scalar_float_variable(standard_name="solar_irradiance_vis", long_name="Solar effective Irradiance", units="W*m-2")

        # u_solar_irradiance_vis
        default_array = np.full([], np.NaN, np.float32)
        variable = Variable([], default_array)
        tu.add_fill_value(variable, np.NaN)
        variable.attrs["long_name"] = "Uncertainty in Solar effective Irradiance"
        tu.add_units(variable, "Wm^-2")
        variable.attrs[corr.PIX_CORR_FORM] = corr.RECT_ABS
        variable.attrs[corr.PIX_CORR_UNIT] = corr.PIXEL
        variable.attrs[corr.PIX_CORR_SCALE] = [-np.inf, np.inf]
        variable.attrs[corr.SCAN_CORR_FORM] = corr.RECT_ABS
        variable.attrs[corr.SCAN_CORR_UNIT] = corr.LINE
        variable.attrs[corr.SCAN_CORR_SCALE] = [-np.inf, np.inf]
        variable.attrs[corr.IMG_CORR_FORM] = corr.RECT_ABS
        variable.attrs[corr.IMG_CORR_UNIT] = corr.DAYS
        variable.attrs[corr.IMG_CORR_SCALE] = [-np.inf, np.inf]
        variable.attrs["pdf_shape"] = "rectangle"
        dataset["u_solar_irradiance_vis"] = variable

        # srf_vis
        default_array = DefaultData.create_default_vector(SRF_SIZE_VIS, np.float32, fill_value=np.NaN)
        variable = Variable([SRF_VIS_DIMENSION], default_array)
        tu.add_fill_value(variable, np.NaN)
        variable.attrs["long_name"] = "Spectral Response Function for visible channel"
        dataset["spectral_response_function_vis"] = variable

        # srf covariance_
        default_array = DefaultData.create_default_array(SRF_SIZE_VIS, SRF_SIZE_VIS, np.float32, fill_value=np.NaN)
        variable = Variable([SRF_VIS_DIMENSION, SRF_VIS_DIMENSION], default_array)
        tu.add_fill_value(variable, np.NaN)
        variable.attrs["long_name"] = "Covariance of the Visible Band Spectral Response Function"
        dataset["covariance_spectral_response_function_vis"] = variable

        # srf_ir
        default_array = DefaultData.create_default_vector(SRF_SIZE_IR_WV, np.float32, fill_value=np.NaN)
        variable = Variable([SRF_IR_WV_DIMENSION], default_array)
        tu.add_fill_value(variable, np.NaN)
        variable.attrs["long_name"] = "Spectral Response Function for IR channel"
        dataset["spectral_response_function_ir"] = variable

        # u_srf_ir
        default_array = DefaultData.create_default_vector(SRF_SIZE_IR_WV, np.float32, fill_value=np.NaN)
        variable = Variable([SRF_IR_WV_DIMENSION], default_array)
        tu.add_fill_value(variable, np.NaN)
        variable.attrs["long_name"] = "Uncertainty in Spectral Response Function for IR channel"
        dataset["u_spectral_response_function_ir"] = variable

        # srf_wv
        default_array = DefaultData.create_default_vector(SRF_SIZE_IR_WV, np.float32, fill_value=np.NaN)
        variable = Variable([SRF_IR_WV_DIMENSION], default_array)
        tu.add_fill_value(variable, np.NaN)
        variable.attrs["long_name"] = "Spectral Response Function for WV channel"
        dataset["spectral_response_function_wv"] = variable

        # u_srf_wv
        default_array = DefaultData.create_default_vector(SRF_SIZE_IR_WV, np.float32, fill_value=np.NaN)
        variable = Variable([SRF_IR_WV_DIMENSION], default_array)
        tu.add_fill_value(variable, np.NaN)
        variable.attrs["long_name"] = "Uncertainty in Spectral Response Function for WV channel"
        dataset["u_spectral_response_function_wv"] = variable

        dataset["a_ir"] = tu.create_scalar_float_variable(long_name="Calibration parameter a for IR Band", units="mWm^-2sr^-1cm^-1")
        dataset["b_ir"] = tu.create_scalar_float_variable(long_name="Calibration parameter b for IR Band", units="mWm^-2sr^-1cm^-1/DC")
        dataset["u_a_ir"] = tu.create_scalar_float_variable(long_name="Uncertainty of calibration parameter a for IR Band", units="mWm^-2sr^-1cm^-1")
        dataset["u_b_ir"] = tu.create_scalar_float_variable(long_name="Uncertainty of calibration parameter b for IR Band", units="mWm^-2sr^-1cm^-1/DC")
        dataset["a_wv"] = tu.create_scalar_float_variable(long_name="Calibration parameter a for WV Band", units="mWm^-2sr^-1cm^-1")
        dataset["b_wv"] = tu.create_scalar_float_variable(long_name="Calibration parameter b for WV Band", units="mWm^-2sr^-1cm^-1/DC")
        dataset["u_a_wv"] = tu.create_scalar_float_variable(long_name="Uncertainty of calibration parameter a for WV Band", units="mWm^-2sr^-1cm^-1")
        dataset["u_b_wv"] = tu.create_scalar_float_variable(long_name="Uncertainty of calibration parameter b for WV Band", units="mWm^-2sr^-1cm^-1/DC")
        dataset["q_ir"] = tu.create_scalar_float_variable(long_name="IR Band Calibration quality flag", units="1")
        dataset["q_wv"] = tu.create_scalar_float_variable(long_name="WV Band Calibration quality flag", units="1")
        dataset["unit_conversion_ir"] = tu.create_scalar_float_variable(long_name="IR Unit conversion factor", units="1")
        dataset["unit_conversion_wv"] = tu.create_scalar_float_variable(long_name="WV Unit conversion factor", units="1")
        dataset["bt_a_ir"] = tu.create_scalar_float_variable(long_name="IR Band BT conversion parameter A", units="1")
        dataset["bt_b_ir"] = tu.create_scalar_float_variable(long_name="IR Band BT conversion parameter B", units="1")
        dataset["bt_a_wv"] = tu.create_scalar_float_variable(long_name="WV Band BT conversion parameter A", units="1")
        dataset["bt_b_wv"] = tu.create_scalar_float_variable(long_name="WV Band BT conversion parameter B", units="1")
        dataset["years_since_launch"] = tu.create_scalar_float_variable(long_name="Fractional year since launch of satellite", units="years")

    @staticmethod
    def get_swath_width():
        return MVIRI.FULL_DIMENSION

    @staticmethod
    def add_easy_fcdr_variables(dataset, height):
        # height is ignored - supplied just for interface compatibility tb 2017-02-05
        #FRue: added 3.7.2017:
        #SSP
        dataset["sub_satellite_latitude_start"] = tu.create_scalar_float_variable(long_name="latitude of the sub satellite point at image start", units="degree_north")
        dataset["sub_satellite_longitude_start"] = tu.create_scalar_float_variable(long_name="longitude of the sub satellite point at image start", units="degree_east")
        dataset["sub_satellite_latitude_end"] = tu.create_scalar_float_variable(long_name="latitude of the sub satellite point at image end", units="degree_north")
        dataset["sub_satellite_longitude_end"] = tu.create_scalar_float_variable(long_name="longitude of the sub satellite point at image end", units="degree_east")

        
        # reflectance
        default_array = DefaultData.create_default_array(FULL_DIMENSION, FULL_DIMENSION, np.uint16)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["standard_name"] = "toa_bidirectional_reflectance_vis"
        variable.attrs["long_name"] = "top of atmosphere bidirectional reflectance factor per pixel of the visible band with central wavelength 0.7"
        tu.add_units(variable, "percent")
        tu.add_encoding(variable, np.uint16, DefaultData.get_default_fill_value(np.uint16), 1.52588E-05)
        dataset["toa_bidirectional_reflectance_vis"] = variable

        # u_random
        default_array = DefaultData.create_default_array(FULL_DIMENSION, FULL_DIMENSION, np.uint16, fill_value=np.NaN)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["long_name"] = "fractional random uncertainty per pixel"
        tu.add_units(variable, "fraction of toa_bidirectional_reflectance")
        tu.set_unsigned(variable)
        tu.add_encoding(variable, np.uint16, DefaultData.get_default_fill_value(np.uint16), 1.52588E-04)
        dataset["u_random_toa_bidirectional_reflectance"] = variable

        # u_non_random
        default_array = DefaultData.create_default_array(FULL_DIMENSION, FULL_DIMENSION, np.uint16, fill_value=np.NaN)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["long_name"] = "fractional non-random uncertainty per pixel"
        tu.add_units(variable, "fraction of toa_bidirectional_reflectance")
        tu.set_unsigned(variable)
        tu.add_encoding(variable, np.uint16, DefaultData.get_default_fill_value(np.uint16), 1.52588E-04)
        dataset["u_non_random_toa_bidirectional_reflectance"] = variable

    @staticmethod
    def add_full_fcdr_variables(dataset, height):
        # height is ignored - supplied just for interface compatibility tb 2017-02-05

        # count_vis
        default_array = DefaultData.create_default_array(MVIRI.FULL_DIMENSION, MVIRI.FULL_DIMENSION, np.uint8)
        variable = Variable(["y", "x"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.uint8))
        variable.attrs["long_name"] = "Image counts"
        tu.add_units(variable, "count")
        dataset["count_vis"] = variable

        dataset["u_latitude"] = MVIRI._create_angle_variable_int(scale_u_lat, long_name="Uncertainty in Latitude", unsigned=True)
        MVIRI._add_geo_correlation_attributes(dataset["u_latitude"])

        dataset["u_longitude"] = MVIRI._create_angle_variable_int(scale_u_lat, long_name="Uncertainty in Longitude", unsigned=True)
        MVIRI._add_geo_correlation_attributes(dataset["u_longitude"])

        # u_time
        default_array = DefaultData.create_default_vector(MVIRI.IR_DIMENSION, np.float32, fill_value=np.NaN)
        variable = Variable([IR_Y_DIMENSION], default_array)
        variable.attrs["standard_name"] = "Uncertainty in Time"
        tu.add_units(variable, "s")
        tu.add_encoding(variable, np.uint16, DefaultData.get_default_fill_value(np.uint16), 0.009155273)
        variable.attrs["pdf_shape"] = "rectangle"
        dataset["u_time"] = variable

        dataset["u_satellite_zenith_angle"] = MVIRI._create_angle_variable_int(7.62939E-05, long_name="Uncertainty in Satellite Zenith Angle", unsigned=True)
        dataset["u_satellite_azimuth_angle"] = MVIRI._create_angle_variable_int(7.62939E-05, long_name="Uncertainty in Satellite Azimuth Angle", unsigned=True)
        dataset["u_solar_zenith_angle"] = MVIRI._create_angle_variable_int(7.62939E-05, long_name="Uncertainty in Solar Zenith Angle", unsigned=True)
        dataset["u_solar_azimuth_angle"] = MVIRI._create_angle_variable_int(7.62939E-05, long_name="Uncertainty in Solar Azimuth Angle", unsigned=True)

        dataset["a0_vis"] = tu.create_scalar_float_variable("Calibration Coefficient at Launch", units="Wm^-2sr^-1/count")
        dataset["a1_vis"] = tu.create_scalar_float_variable("Time variation of a0", units="Wm^-2sr^-1/count day^-1 10^5")
        dataset["mean_count_space_vis"] = tu.create_scalar_float_variable("Space count", units="count")

        # u_a0_vis
        variable = tu.create_scalar_float_variable("Uncertainty in a0", units="Wm^-2sr^-1/count")
        MVIRI._add_calibration_coeff_correlation_attributes(variable)
        dataset["u_a0_vis"] = variable

        # u_a1_vis
        variable = tu.create_scalar_float_variable("Uncertainty in a1", units="Wm^-2sr^-1/count day^-1 10^5")
        MVIRI._add_calibration_coeff_correlation_attributes(variable)
        dataset["u_a1_vis"] = variable

        # covariance_a0_a1_vis
        variable = tu.create_scalar_float_variable("Covariance of calibration coefficients")
        tu.add_fill_value(variable, np.NaN)
        dataset["covariance_a0_a1_vis"] = variable

        dataset["u_electronics_counts_vis"] = tu.create_scalar_float_variable("Uncertainty due to Electronics noise", units="count")
        dataset["u_digitization_counts_vis"] = tu.create_scalar_float_variable("Uncertainty due to digitization", units="count")

        # allan_deviation_counts_space_vis
        variable = tu.create_scalar_float_variable("Uncertainty of space count", units="count")
        variable.attrs[corr.SCAN_CORR_FORM] = corr.RECT_ABS
        variable.attrs[corr.SCAN_CORR_UNIT] = corr.LINE
        variable.attrs[corr.SCAN_CORR_SCALE] = [-np.inf, np.inf]
        variable.attrs["pdf_shape"] = "digitised_gaussian"
        dataset["allan_deviation_counts_space_vis"] = variable

    @staticmethod
    def _add_geo_correlation_attributes(geo_variable):
        geo_variable.attrs[corr.PIX_CORR_FORM] = corr.TRI_REL
        geo_variable.attrs[corr.PIX_CORR_UNIT] = corr.PIXEL
        geo_variable.attrs[corr.PIX_CORR_SCALE] = [-250, 250]
        geo_variable.attrs[corr.SCAN_CORR_FORM] = corr.TRI_REL
        geo_variable.attrs[corr.SCAN_CORR_UNIT] = corr.LINE
        geo_variable.attrs[corr.SCAN_CORR_SCALE] = [-250, 250]
        geo_variable.attrs[corr.IMG_CORR_FORM] = corr.TRI_REL
        geo_variable.attrs[corr.IMG_CORR_UNIT] = corr.IMG
        geo_variable.attrs[corr.IMG_CORR_SCALE] = [-12, 0]
        geo_variable.attrs["pdf_shape"] = "gaussian"

    @staticmethod
    def _add_calibration_coeff_correlation_attributes(coeff_variable):
        coeff_variable.attrs[corr.PIX_CORR_FORM] = corr.RECT_ABS
        coeff_variable.attrs[corr.PIX_CORR_UNIT] = corr.PIXEL
        coeff_variable.attrs[corr.PIX_CORR_SCALE] = [-np.inf, np.inf]
        coeff_variable.attrs[corr.SCAN_CORR_FORM] = corr.RECT_ABS
        coeff_variable.attrs[corr.SCAN_CORR_UNIT] = corr.LINE
        coeff_variable.attrs[corr.SCAN_CORR_SCALE] = [-np.inf, np.inf]
        coeff_variable.attrs[corr.IMG_CORR_FORM] = corr.TRI_REL
        coeff_variable.attrs[corr.IMG_CORR_UNIT] = corr.MONTHS
        coeff_variable.attrs[corr.IMG_CORR_SCALE] = [-1.5, 1.5]
        coeff_variable.attrs["pdf_shape"] = "gaussian"

    @staticmethod
    def _create_angle_variable_int(scale_factor, standard_name=None, long_name=None, unsigned=False, fill_value=None):
        
        if unsigned is True:
            data_type = np.uint16
        else:
            data_type = np.int16
            
        default_array = DefaultData.create_default_array(FULL_DIMENSION, FULL_DIMENSION, data_type, fill_value=np.NaN)
        variable = Variable(["y", "x"], default_array)

        if fill_value is None:
            fill_value = DefaultData.get_default_fill_value(data_type)

        if standard_name is not None:
            variable.attrs["standard_name"] = standard_name

        if long_name is not None:
            variable.attrs["long_name"] = long_name

        tu.add_units(variable, "degree")
        tu.add_encoding(variable, data_type, fill_value, scale_factor)
        return variable
