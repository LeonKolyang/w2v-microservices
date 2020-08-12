class PreprocessingException(Exception):
    pass

class MissingDataException(PreprocessingException):
    def __init__(self, data, message = "Data is missing in Preprocessor Unit:",*args, **kwargs):
        datastring = ""
        for d in data:
            self.datastring += data+" "
        self.message = message + datastring
        super().__init__(self.message)