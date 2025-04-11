from pydantic import BaseModel

class PredictionResponse(BaseModel):
    status: str
    acl_probability: float
    meniscus_probability: float
    view_type: str