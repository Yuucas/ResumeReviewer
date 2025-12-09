#!/usr/bin/env python3
"""
Database migration script to add job_title column to job_searches table.

This script adds the job_title column and populates it by extracting titles
from existing job descriptions.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from app.db.database import engine, SessionLocal
from app.db.models import JobSearch
from app.utils.text_utils import extract_job_title
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def migrate():
    """Run the migration to add job_title column."""

    logger.info("Starting migration: Adding job_title column to job_searches table")

    db = SessionLocal()

    try:
        # Step 1: Check if column already exists
        logger.info("Checking if job_title column exists...")

        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'job_searches'
                AND column_name = 'job_title'
            """))

            if result.fetchone():
                logger.info("✓ job_title column already exists, skipping migration")
                return

        # Step 2: Add the column
        logger.info("Adding job_title column...")

        with engine.connect() as conn:
            conn.execute(text("""
                ALTER TABLE job_searches
                ADD COLUMN job_title VARCHAR(255)
            """))
            conn.commit()

        logger.info("✓ job_title column added successfully")

        # Step 3: Populate job_title for existing records
        logger.info("Extracting job titles from existing job descriptions...")

        searches = db.query(JobSearch).all()
        count = 0

        for search in searches:
            if not search.job_title:  # Only update if not already set
                job_title = extract_job_title(search.job_description)
                search.job_title = job_title
                count += 1

                if count % 10 == 0:
                    logger.info(f"Processed {count}/{len(searches)} records...")

        db.commit()
        logger.info(f"✓ Updated {count} existing records with extracted job titles")

        logger.info("Migration completed successfully!")

    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        db.rollback()
        raise

    finally:
        db.close()


if __name__ == "__main__":
    try:
        migrate()
    except KeyboardInterrupt:
        logger.info("\nMigration interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Migration error: {e}")
        sys.exit(1)
