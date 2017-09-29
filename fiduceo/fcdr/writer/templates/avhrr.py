import numpy as np
from xarray import Variable

from fiduceo.fcdr.writer.correlation import Correlation as corr
from fiduceo.fcdr.writer.default_data import DefaultData
from fiduceo.fcdr.writer.templates.templateutil import TemplateUtil as tu

SWATH_WIDTH = 409
PRT_WIDTH = 3
N_CHANS = 6

COUNT_CORRELATION_ATTRIBUTES = {corr.PIX_CORR_FORM: corr.RECT_ABS, corr.PIX_CORR_UNIT: corr.PIXEL,
                                corr.PIX_CORR_SCALE: [-np.inf, np.inf], corr.SCAN_CORR_FORM: corr.TRI_REL,
                                corr.SCAN_CORR_UNIT: corr.LINE,
                                corr.SCAN_CORR_SCALE: [-25, 25], "pdf_shape": "digitised_gaussian"}


class AVHRR:
    @staticmethod
    def add_original_variables(dataset, height):
        tu.add_geolocation_variables(dataset, SWATH_WIDTH, height)

        # Time
        default_array = DefaultData.create_default_vector(height, np.float64, fill_value=np.NaN)
        variable = Variable(["y"], default_array)
        tu.add_fill_value(variable, np.NaN)
        tu.add_units(variable, "s")
        variable.attrs["standard_name"] = "time"
        variable.attrs["long_name"] = "Acquisition time in seconds since 1970-01-01 00:00:00"
        dataset["Time"] = variable

        # relative_azimuth_angle
        default_array = DefaultData.create_default_array(SWATH_WIDTH, height, np.float32, fill_value=np.NaN)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["standard_name"] = "relative_azimuth_angle"
        tu.add_units(variable, "degree")
        tu.add_encoding(variable, np.int16, DefaultData.get_default_fill_value(np.int16), 0.01)
        variable.attrs["valid_max"] = 18000
        variable.attrs["valid_min"] = -18000
        dataset["relative_azimuth_angle"] = variable

        # satellite_zenith_angle
        default_array = DefaultData.create_default_array(SWATH_WIDTH, height, np.float32, fill_value=np.NaN)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["standard_name"] = "sensor_zenith_angle"
        tu.add_units(variable, "degree")
        tu.add_encoding(variable, np.int16, DefaultData.get_default_fill_value(np.int16), 0.01)
        variable.attrs["valid_max"] = 9000
        variable.attrs["valid_min"] = 0
        dataset["satellite_zenith_angle"] = variable

        # solar_zenith_angle
        default_array = DefaultData.create_default_array(SWATH_WIDTH, height, np.float32, fill_value=np.NaN)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["standard_name"] = "solar_zenith_angle"
        tu.add_units(variable, "degree")
        tu.add_encoding(variable, np.int16, DefaultData.get_default_fill_value(np.int16), 0.01)
        variable.attrs["valid_max"] = 18000
        variable.attrs["valid_min"] = 0
        dataset["solar_zenith_angle"] = variable

        # Ch1_Ref
        variable = AVHRR._create_channel_refl_variable(height, "Channel 1 Reflectance")
        dataset["Ch1_Ref"] = variable

        # Ch2_Ref
        variable = AVHRR._create_channel_refl_variable(height, "Channel 2 Reflectance")
        dataset["Ch2_Ref"] = variable

        # Ch3a_Ref
        variable = AVHRR._create_channel_refl_variable(height, "Channel 3a Reflectance")
        dataset["Ch3a_Ref"] = variable

        # Ch3b_Bt
        variable = AVHRR._create_channel_bt_variable(height, "Channel 3b Brightness Temperature")
        dataset["Ch3b_Bt"] = variable

        # Ch4_Bt
        variable = AVHRR._create_channel_bt_variable(height, "Channel 4 Brightness Temperature")
        dataset["Ch4_Bt"] = variable

        # Ch5_Bt
        variable = AVHRR._create_channel_bt_variable(height, "Channel 5 Brightness Temperature")
        dataset["Ch5_Bt"] = variable

    @staticmethod
    def get_swath_width():
        return SWATH_WIDTH

    @staticmethod
    def add_easy_fcdr_variables(dataset, height):
        # u_random_Ch1-3a
        long_names = ["random uncertainty per pixel for channel 1", "random uncertainty per pixel for channel 2",
                      "random uncertainty per pixel for channel 3a"]
        names = ["u_random_Ch1", "u_random_Ch2", "u_random_Ch3a"]
        AVHRR._add_refl_uncertainties_variables_long_name(dataset, height, names, long_names)

        # u_non_random_Ch1-3a
        long_names = ["non-random uncertainty per pixel for channel 1",
                      "non-random uncertainty per pixel for channel 2",
                      "non-random uncertainty per pixel for channel 3a"]
        names = ["u_non_random_Ch1", "u_non_random_Ch2", "u_non_random_Ch3a"]
        AVHRR._add_refl_uncertainties_variables_long_name(dataset, height, names, long_names,systematic=True)

        # u_random_Ch3b-5
        long_names = ["random uncertainty per pixel for channel 3b", "random uncertainty per pixel for channel 4",
                      "random uncertainty per pixel for channel 5"]
        names = ["u_random_Ch3b", "u_random_Ch4", "u_random_Ch5"]
        AVHRR._add_bt_uncertainties_variables(dataset, height, names, long_names)

        # u_non_random_Ch3b-5
        long_names = ["non-random uncertainty per pixel for channel 3b",
                      "non-random uncertainty per pixel for channel 4",
                      "non-random uncertainty per pixel for channel 5"]
        names = ["u_non_random_Ch3b", "u_non_random_Ch4", "u_non_random_Ch5"]
        AVHRR._add_bt_uncertainties_variables(dataset, height, names, long_names)
        default_array = DefaultData.create_default_vector(height, np.uint8, fill_value=None)
        variable = Variable(["y"], default_array)
        variable.attrs["long_name"] = 'Bitmask for quality per scanline'
        variable.attrs["flag_masks"] = '1,2,4,8,16,32,64'
        variable.attrs['flag_meanings'] = 'DO_NOT_USE, BAD_TIME, BAD_NAVIGATION, BAD_CALIBRATION, CHANNEL3A_PRESENT,SOLAR_CONTAMINATION_FAILURE,SOLAR_CONTAMINATION'
        dataset['quality_scanline_bitmask'] = variable

        default_array = DefaultData.create_default_array(N_CHANS, height, np.uint8, fill_value=None)
        variable = Variable(["y","nchan"], default_array)
        variable.attrs["long_name"] = 'Bitmask for quality per channel/scanline'
        variable.attrs["flag_masks"] = '1,2'
        variable.attrs['flag_meanings'] = 'BAD_CHANNEL, SOME_PIXELS_NOT_DETECTED_2SIGMA'
        dataset['quality_channel_bitmask'] = variable

    @staticmethod
    def add_full_fcdr_variables(dataset, height):
        # u_latitude
        variable = AVHRR._create_angle_uncertainty_variable("latitude", height)
        dataset["u_latitude"] = variable

        # u_longitude
        variable = AVHRR._create_angle_uncertainty_variable("longitude", height)
        dataset["u_longitude"] = variable

        # u_time
        default_array = DefaultData.create_default_vector(height, np.float64, fill_value=np.NaN)
        variable = Variable(["y"], default_array)
        tu.add_fill_value(variable, np.NaN)
        tu.add_units(variable, "s")
        variable.attrs["long_name"] = "uncertainty of acquisition time"
        dataset["u_time"] = variable

        # u_satellite_azimuth_angle
        variable = AVHRR._create_angle_uncertainty_variable("satellite azimuth angle", height)
        dataset["u_satellite_azimuth_angle"] = variable

        # u_satellite_zenith_angle
        variable = AVHRR._create_angle_uncertainty_variable("satellite zenith angle", height)
        dataset["u_satellite_zenith_angle"] = variable

        # u_solar_azimuth_angle
        variable = AVHRR._create_angle_uncertainty_variable("solar azimuth angle", height)
        dataset["u_solar_azimuth_angle"] = variable

        # u_solar_zenith_angle
        variable = AVHRR._create_angle_uncertainty_variable("solar zenith angle", height)
        dataset["u_solar_zenith_angle"] = variable

        # PRT_C
        default_array = DefaultData.create_default_array(PRT_WIDTH, height, np.int16)
        variable = Variable(["y", "n_prt"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int16))
        variable.attrs["long_name"] = "Prt counts"
        tu.add_units(variable, "count")
        dataset["PRT_C"] = variable

        # u_prt
        default_array = DefaultData.create_default_array(PRT_WIDTH, height, np.float32, fill_value=np.NaN)
        variable = Variable(["y", "n_prt"], default_array)
        tu.add_fill_value(variable, np.NaN)
        variable.attrs["long_name"] = "Uncertainty on the PRT counts"
        tu.add_units(variable, "count")
        variable.attrs[corr.PIX_CORR_FORM] = corr.RECT_ABS
        variable.attrs[corr.PIX_CORR_UNIT] = corr.PIXEL
        variable.attrs[corr.PIX_CORR_SCALE] = [-np.inf, np.inf]
        variable.attrs[corr.SCAN_CORR_FORM] = corr.RECT_ABS
        variable.attrs[corr.SCAN_CORR_UNIT] = corr.LINE
        variable.attrs[corr.SCAN_CORR_SCALE] = [-np.inf, np.inf]
        variable.attrs["pdf_shape"] = "rectangle"
        variable.attrs["pdf_parameter"] = 0.1
        dataset["u_prt"] = variable

        # R_ICT
        default_array = DefaultData.create_default_array(PRT_WIDTH, height, np.float32, fill_value=np.NaN)
        variable = Variable(["y", "n_prt"], default_array)
        tu.add_fill_value(variable, np.NaN)
        variable.attrs["long_name"] = "Radiance of the PRT"
        tu.add_units(variable, "mW m^-2 sr^-1 cm")
        dataset["R_ICT"] = variable

        # T_instr
        default_array = np.NaN
        variable = Variable([], default_array)
        tu.add_fill_value(variable, np.NaN)
        variable.attrs["long_name"] = "Instrument temperature"
        tu.add_units(variable, "K")
        dataset["T_instr"] = variable

        # Chx_Csp
        standard_names = ["Ch1 Space counts", "Ch2 Space counts", "Ch3a Space counts", "Ch3b Space counts",
                          "Ch4 Space counts", "Ch5 Space counts"]
        names = ["Ch1_Csp", "Ch2_Csp", "Ch3a_Csp", "Ch3b_Csp", "Ch4_Csp", "Ch5_Csp"]
        AVHRR._add_counts_variables(dataset, height, names, standard_names)

        # Chx_Cict
        standard_names = ["Ch3b ICT counts", "Ch4 ICT counts", "Ch5 ICT counts"]
        names = ["Ch3b_Cict", "Ch4_Cict", "Ch5_Cict"]
        AVHRR._add_counts_variables(dataset, height, names, standard_names)

        # Chx_Ce
        standard_names = ["Ch1 Earth counts", "Ch2 Earth counts", "Ch3a Earth counts", "Ch3b Earth counts",
                          "Ch4 Earth counts", "Ch5 Earth counts"]
        names = ["Ch1_Ce", "Ch2_Ce", "Ch3a_Ce", "Ch3b_Ce", "Ch4_Ce", "Ch5_Ce"]
        AVHRR._add_counts_variables(dataset, height, names, standard_names)

        # Chx_u_Csp
        standard_names = ["Ch1 Uncertainty on space counts", "Ch2 Uncertainty on space counts",
                          "Ch3a Uncertainty on space counts", "Ch3b Uncertainty on space counts",
                          "Ch4 Uncertainty on space counts", "Ch5 Uncertainty on space counts"]
        names = ["Ch1_u_Csp", "Ch2_u_Csp", "Ch3a_u_Csp", "Ch3b_u_Csp", "Ch4_u_Csp", "Ch5_u_Csp"]
        AVHRR._add_counts_uncertainties_variables(dataset, height, names, standard_names, COUNT_CORRELATION_ATTRIBUTES)

        # Chx_Cict
        standard_names = ["Ch3b Uncertainty on ICT counts", "Ch4 Uncertainty on ICT counts",
                          "Ch5 Uncertainty on ICT counts"]
        names = ["Ch3b_u_Cict", "Ch4_u_Cict", "Ch5_u_Cict"]
        AVHRR._add_counts_uncertainties_variables(dataset, height, names, standard_names, COUNT_CORRELATION_ATTRIBUTES)

        # Chx_u_Ce
        standard_names = ["Ch1 Uncertainty on earth counts", "Ch2 Uncertainty on earth counts",
                          "Ch3a Uncertainty on earth counts", "Ch3b Uncertainty on earth counts",
                          "Ch4 Uncertainty on earth counts", "Ch5 Uncertainty on earth counts"]
        names = ["Ch1_u_Ce", "Ch2_u_Ce", "Ch3a_u_Ce", "Ch3b_u_Ce", "Ch4_u_Ce", "Ch5_u_Ce"]
        attributes = {"pdf_shape": "digitised_gaussian"}
        AVHRR._add_counts_uncertainties_variables(dataset, height, names, standard_names, attributes)

        # Chx_u_Refl
        long_names = ["Ch1 Total uncertainty on reflectance", "Ch2 Total uncertainty on reflectance",
                      "Ch3a Total uncertainty on reflectance"]
        names = ["Ch1_u_Refl", "Ch2_u_Refl", "Ch3a_u_Refl"]
        AVHRR._add_refl_uncertainties_variables_long_name(dataset, height, names, long_names)

        # Chx_u_Bt
        standard_names = ["Ch3b Total uncertainty on brightness temperature",
                          "Ch4 Total uncertainty on brightness temperature",
                          "Ch5 Total uncertainty on brightness temperature"]
        names = ["Ch3b_u_Bt", "Ch4_u_Bt", "Ch5_u_Bt"]
        AVHRR._add_bt_uncertainties_variables(dataset, height, names, standard_names)

        # Chx_ur_Bt
        standard_names = ["Ch3b Random uncertainty on brightness temperature",
                          "Ch4 Random uncertainty on brightness temperature",
                          "Ch5 Random uncertainty on brightness temperature"]
        names = ["Ch3b_ur_Bt", "Ch4_ur_Bt", "Ch5_ur_Bt"]
        AVHRR._add_bt_uncertainties_variables(dataset, height, names, standard_names)

        # Chx_us_Bt
        standard_names = ["Ch3b Systematic uncertainty on brightness temperature",
                          "Ch4 Systematic uncertainty on brightness temperature",
                          "Ch5 Systematic uncertainty on brightness temperature"]
        names = ["Ch3b_us_Bt", "Ch4_us_Bt", "Ch5_us_Bt"]
        AVHRR._add_bt_uncertainties_variables(dataset, height, names, standard_names)

    @staticmethod
    def _add_counts_uncertainties_variables(dataset, height, names, long_names, attributes=None):
        for i, name in enumerate(names):
            variable = AVHRR._create_counts_uncertainty_variable(height, long_names[i])
            if attributes is not None:
                for key, value in attributes.items():
                    variable.attrs[key] = value
            dataset[name] = variable

    @staticmethod
    def _add_bt_uncertainties_variables(dataset, height, names, long_names):
        for i, name in enumerate(names):
            variable = AVHRR._create_bt_uncertainty_variable(height, long_name=long_names[i])
            dataset[name] = variable

    # @staticmethod
    # def _add_refl_uncertainties_variables(dataset, height, names, standard_names,systematic=False):
    #     for i, name in enumerate(names):
    #         variable = AVHRR._create_refl_uncertainty_variable(height, standard_name=standard_names[i],systematic=systematic)
    #         dataset[name] = variable

    @staticmethod
    def _add_refl_uncertainties_variables_long_name(dataset, height, names, long_names,systematic=False):
        for i, name in enumerate(names):
            variable = AVHRR._create_refl_uncertainty_variable(height, long_name=long_names[i],systematic=systematic)
            dataset[name] = variable

    @staticmethod
    def _add_counts_variables(dataset, height, names, standard_names):
        for i, name in enumerate(names):
            variable = AVHRR._create_counts_variable(height, standard_names[i])
            dataset[name] = variable

    @staticmethod
    def _create_counts_uncertainty_variable(height, long_name):
        variable = tu.create_float_variable(SWATH_WIDTH, height, long_name=long_name, fill_value=np.NaN)
        tu.add_units(variable, "count")
        return variable

    @staticmethod
    def _create_counts_variable(height, long_name):
        default_array = DefaultData.create_default_array(SWATH_WIDTH, height, np.int32)
        variable = Variable(["y", "x"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int32))
        variable.attrs["long_name"] = long_name
        tu.add_units(variable, "count")
        return variable

    @staticmethod
    def _create_angle_uncertainty_variable(angle_name, height):
        variable = tu.create_float_variable(SWATH_WIDTH, height, long_name="uncertainty of " + angle_name, fill_value=np.NaN)
        tu.add_units(variable, "degree")
        return variable

    @staticmethod
    def _create_refl_uncertainty_variable(height, standard_name=None, long_name=None,systematic=False):
        default_array = DefaultData.create_default_array(SWATH_WIDTH, height, np.float32, fill_value=np.NaN)
        variable = Variable(["y", "x"], default_array)
        if systematic:
            tu.add_units(variable, "relative uncertainty ratio")
        else:
            tu.add_units(variable, "albedo")
        if systematic:
            tu.add_encoding(variable, np.int16, DefaultData.get_default_fill_value(np.int16), 0.01)
            variable.attrs["valid_max"] = 3
            variable.attrs["valid_min"] = 5
        else:
            tu.add_encoding(variable, np.int16, DefaultData.get_default_fill_value(np.int16), 0.00001)
            variable.attrs["valid_max"] = 10
            variable.attrs["valid_min"] = 1000
#        variable = tu.create_float_variable(SWATH_WIDTH, height, standard_name=standard_name, long_name=long_name, fill_value=np.NaN)
        return variable

    @staticmethod
    def _create_bt_uncertainty_variable(height, long_name):
#        variable = tu.create_float_variable(SWATH_WIDTH, height, long_name=long_name, fill_value=np.NaN)
#        tu.add_units(variable, "K")
        default_array = DefaultData.create_default_array(SWATH_WIDTH, height, np.float32, fill_value=np.NaN)
        variable = Variable(["y", "x"], default_array)
        tu.add_units(variable, "K")
        tu.add_encoding(variable, np.int16, DefaultData.get_default_fill_value(np.int16), 0.001)
        variable.attrs["valid_max"] = 1
        variable.attrs["valid_min"] = 15000
        return variable

    @staticmethod
    def _add_temperature_attributes(variable):
        variable.attrs["add_offset"] = 273.15
        tu.add_scale_factor(variable, 0.01)
        tu.add_units(variable, "K")
        variable.attrs["valid_max"] = 10000
        variable.attrs["valid_min"] = -20000

    @staticmethod
    def _create_channel_refl_variable(height, long_name):
        default_array = DefaultData.create_default_array(SWATH_WIDTH, height, np.float32, fill_value=np.NaN)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["standard_name"] = "toa_reflectance"
        variable.attrs["long_name"] = long_name
        tu.add_units(variable, "albedo")
        tu.add_encoding(variable, np.int16, DefaultData.get_default_fill_value(np.int16), 1e-4)
        variable.attrs["valid_max"] = 15000
        variable.attrs["valid_min"] = 0
        return variable

    @staticmethod
    def _create_channel_bt_variable(height, long_name):
        default_array = DefaultData.create_default_array(SWATH_WIDTH, height, np.float32, fill_value=np.NaN)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["standard_name"] = "toa_brightness_temperature"
        variable.attrs["long_name"] = long_name
        tu.add_units(variable, "K")
        variable.attrs["valid_max"] = 10000
        variable.attrs["valid_min"] = -20000
        tu.add_encoding(variable, np.int16, DefaultData.get_default_fill_value(np.int16), 0.01, 273.15)
        return variable
