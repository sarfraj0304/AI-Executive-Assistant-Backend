from pydantic import BaseModel, Field


class WorkExperience(BaseModel):
    company: str = Field(description="Company / organization name")
    title: str = Field(description="Job title / role")
    start_date: str | None = Field(default=None, description="e.g. 'Jan 2022'")
    end_date: str | None = Field(default=None, description="e.g. 'Present'")
    description: str | None = Field(
        default=None, description="Summary of responsibilities/achievements"
    )


class Education(BaseModel):
    institution: str
    degree: str | None = None
    field_of_study: str | None = None
    start_date: str | None = None
    end_date: str | None = None


class ResumeSchema(BaseModel):
    """Structured representation of a candidate's resume."""

    full_name: str | None = None
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    summary: str | None = Field(
        default=None, description="A short professional summary / objective"
    )
    skills: list[str] = Field(default_factory=list)
    work_experience: list[WorkExperience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    total_years_experience: float | None = Field(
        default=None, description="Estimated total years of professional experience"
    )
