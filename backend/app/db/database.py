"""
PostgreSQL database connection and session management.
"""

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import logging
from typing import Generator

logger = logging.getLogger(__name__)

# Database configuration
DATABASE_HOST = os.getenv("DATABASE_HOST", "localhost")
DATABASE_PORT = os.getenv("DATABASE_PORT", "5432")
DATABASE_NAME = os.getenv("DATABASE_NAME", "ResumeReviewer")
DATABASE_USER = os.getenv("DATABASE_USER", "postgres")
DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD", "1234")

# Build connection URL
DATABASE_URL = f"postgresql://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Verify connections before using
    pool_size=5,
    max_overflow=10
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db() -> Generator:
    """
    Dependency to get database session.

    Usage in FastAPI endpoints:
        @app.get("/items")
        def read_items(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database - create all tables and run migrations."""
    Base.metadata.create_all(bind=engine)

    # Run migrations
    _migrate_add_job_title()


def _migrate_add_job_title():
    """
    Migration: Add job_title column if it doesn't exist.
    This runs automatically when the app starts.
    """
    try:
        inspector = inspect(engine)
        columns = [col['name'] for col in inspector.get_columns('job_searches')]

        if 'job_title' not in columns:
            logger.info("Running migration: Adding job_title column...")

            with engine.connect() as conn:
                conn.execute(text("""
                    ALTER TABLE job_searches
                    ADD COLUMN job_title VARCHAR(255)
                """))
                conn.commit()

            logger.info("✓ Migration completed: job_title column added")

            # Extract titles for existing records
            _extract_titles_for_existing_searches()
        else:
            logger.debug("job_title column already exists, skipping migration")

    except Exception as e:
        logger.error(f"Migration error: {e}")


def _extract_titles_for_existing_searches():
    """Extract job titles for existing searches that don't have one."""
    try:
        from ..utils.text_utils import extract_job_title
        from .models import JobSearch

        db = SessionLocal()
        try:
            searches = db.query(JobSearch).filter(JobSearch.job_title.is_(None)).all()

            if searches:
                logger.info(f"Extracting job titles for {len(searches)} existing searches...")

                for search in searches:
                    search.job_title = extract_job_title(search.job_description)

                db.commit()
                logger.info(f"✓ Updated {len(searches)} searches with job titles")
        except Exception as e:
            logger.error(f"Failed to extract titles for existing searches: {e}")
            db.rollback()
        finally:
            db.close()

    except ImportError:
        logger.warning("Could not import text_utils, skipping job title extraction")
