from .DataHandler import DataHandler

def prepHRT(handler:DataHandler):
    handler.removeFeature("PatientId")
    handler.removeFeature("Species")
    handler.removeInvalidData()
    handler.translateData()
    handler.normalizeData()
    return handler

def prepIRS(handler:DataHandler):
    handler.removeFeature("Id")
    handler.removeInvalidData()
    handler.translateData()
    handler.normalizeData()
    return handler

def prepHPR(handler:DataHandler):
    handler.removeFeature("id")
    handler.removeFeature("date")
    handler.removeInvalidData()
    handler.translateData()
    handler.normalizeData()
    return handler