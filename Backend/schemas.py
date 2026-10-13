from pydantic import BaseModel, Field


class RecruitmentRequest(BaseModel):
    combined_text: str = Field(
        ...,
        min_length=1,
        description="Combined candidate and job description text"
    )

    candidate_experience_years: float = Field(..., ge=0)
    job_required_experience: float = Field(..., ge=0)
    experience_gap: float
