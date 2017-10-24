import numpy as np
from xarray import Variable

from fiduceo.fcdr.writer.default_data import DefaultData


class TemplateUtil:
    @staticmethod
    def add_geolocation_variables(dataset, width, height):
        default_array = DefaultData.create_default_array(width, height, np.float32, fill_value=np.NaN)

        variable = Variable(["y", "x"], default_array)
        variable.attrs["standard_name"] = "latitude"
        TemplateUtil.add_units(variable, "degrees_north")
        TemplateUtil.add_encoding(variable, np.int16, -32768, scale_factor=0.0027466658)
        dataset["latitude"] = variable

        variable = Variable(["y", "x"], default_array)
        variable.attrs["standard_name"] = "longitude"
        TemplateUtil.add_units(variable, "degrees_east")
        TemplateUtil.add_encoding(variable, np.int16, -32768, scale_factor=0.0054933317)
        dataset["longitude"] = variable

    @staticmethod
    def create_scalar_float_variable(long_name=None, standard_name=None, units=None, fill_value=np.NaN):
        default_array = fill_value

        variable = Variable([], default_array)
        TemplateUtil.add_fill_value(variable, fill_value)

        if long_name is not None:
            variable.attrs["long_name"] = long_name

        if standard_name is not None:
            variable.attrs["standard_name"] = standard_name

        if units is not None:
            TemplateUtil.add_units(variable, units)
        return variable

    @staticmethod
    def create_float_variable(width, height, standard_name=None, long_name=None, dim_names=None, fill_value=None):
        if fill_value is None:
            default_array = DefaultData.create_default_array(width, height, np.float32)
        else:
            default_array = DefaultData.create_default_array(width, height, np.float32, fill_value=fill_value)

        if dim_names is None:
            variable = Variable(["y", "x"], default_array)
        else:
            variable = Variable(dim_names, default_array)

        if fill_value is None:
            variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.float32)
        else:
            variable.attrs["_FillValue"] = fill_value

        if standard_name is not None:
            variable.attrs["standard_name"] = standard_name

        if long_name is not None:
            variable.attrs["long_name"] = long_name

        return variable

    @staticmethod
    def set_unsigned(variable):
        variable.attrs["_Unsigned"] = "true"

    @staticmethod
    def add_fill_value(variable, fill_value):
        variable.attrs["_FillValue"] = fill_value

    @staticmethod
    def add_units(variable, units):
        variable.attrs["units"] = units

    @staticmethod
    def add_scale_factor(variable, scale_factor):
        variable.attrs["scale_factor"] = scale_factor

    @staticmethod
    def add_offset(variable, offset):
        variable.attrs["add_offset"] = offset

    @staticmethod
    def add_encoding(variable, data_type, fill_value, scale_factor=1.0, offset=0.0):
        variable.encoding = dict([('dtype', data_type), ('_FillValue', fill_value), ('scale_factor', scale_factor), ('add_offset', offset)])
        for key, value in variable.encoding.iteritems():
          if not key in 'dtype':
            variable.attrs[key] = value