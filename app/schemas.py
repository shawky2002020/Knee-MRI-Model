from pydantic import BaseModel

class PredictionResponse(BaseModel):
    status: str
    acl_probability: float
    meniscus_probability: float
    view_type: str

class Result(BaseModel):
    status: str
    acl_prob: float
    meniscus_prob: float

class FinalResponse(BaseModel):
    result: Result
    mri_scan: str
    heat_map: str
    report: str