from fiduceo.fcdr.writer.templates.hirs import HIRS

MAX_SRF_SIZE = 102


class HIRS2(HIRS):
    @staticmethod
    def add_original_variables(dataset, height, srf_size=None):
        if srf_size is None:
            srf_size = MAX_SRF_SIZE

        HIRS.add_original_variables(dataset, height, srf_size)

    @staticmethod
    def add_easy_fcdr_variables(dataset, height, corr_dx=None, corr_dy=None, lut_size=None):
        HIRS.add_easy_fcdr_variables(dataset, height, corr_dx, corr_dy, lut_size)

    @staticmethod
    def add_full_fcdr_variables(dataset, height):
        HIRS.add_full_fcdr_variables(dataset, height)

    @staticmethod
    def get_swath_width():
        return HIRS.get_swath_width()

    @staticmethod
    def add_template_key(dataset):
        dataset.attrs["template_key"] = "HIRS2"
