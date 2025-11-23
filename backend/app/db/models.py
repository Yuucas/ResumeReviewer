"""
SQLAlchemy models for search history.
"""

from sqlalchemy import Column, Integer, String, Text, Float, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base


class JobSearch(Base):
    """Model for storing job search queries."""

    __tablename__ = "job_searches"

    id = Column(Integer, primary_key=True, index=True)
    job_title = Column(String(255), nullable=True)  # Extracted job title/role
    job_description = Column(Text, nullable=False)
    top_k = Column(Integer, default=5)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    total_candidates = Column(Integer, default=0)
    processing_time = Column(Float, nullable=True)

    # Relationship to results
    results = relationship("SearchResult", back_populates="job_search", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<JobSearch(id={self.id}, created_at={self.created_at})>"


class SearchResult(Base):
    """Model for storing individual candidate results from a search."""

    __tablename__ = "search_results"

    id = Column(Integer, primary_key=True, index=True)
    job_search_id = Column(Integer, ForeignKey("job_searches.id"), nullable=False)

    # Candidate information
    filename = Column(String(255), nullable=False)
    match_score = Column(Float, nullable=True)
    recommendation = Column(String(50), nullable=True)

    # Analysis results
    strengths = Column(JSON, nullable=True)  # List of strengths
    weaknesses = Column(JSON, nullable=True)  # List of weaknesses
    overall_assessment = Column(Text, nullable=True)

    # Additional candidate info
    experience_years = Column(Integer, nullable=True)
    role_category = Column(String(100), nullable=True)
    email = Column(String(255), nullable=True)

    # GitHub information
    github_username = Column(String(100), nullable=True)
    github_profile_url = Column(String(255), nullable=True)
    github_relevance_score = Column(Float, nullable=True)
    github_top_languages = Column(JSON, nullable=True)  # List of languages
    github_relevant_projects = Column(JSON, nullable=True)  # List of projects
    github_summary = Column(Text, nullable=True)

    # Relationship to job search
    job_search = relationship("JobSearch", back_populates="results")

    def __repr__(self):
        return f"<SearchResult(id={self.id}, filename={self.filename}, match_score={self.match_score})>"
