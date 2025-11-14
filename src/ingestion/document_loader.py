import os
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class DocumentLoader():
    '''
    Loading configuration for document ingestion.
    '''
    
    def __init__(self, dataset_root: str):
        super().__init__()

        self.dataset_root = Path(dataset_root)
        
        if not self.dataset_root.exists():
            raise ValueError("DATASET_ROOT_PATH environment variable is not set.")
        
        logger.info(f"Dataset root path set to: {self.dataset_root}")
        
    def get_all_resumes(self, file_extensions: Optional[List[str]] = None) -> List[Path]:
        '''
        Retrieve all resume files from the dataset root path with specified extensions.
        
        Args:
            file_extensions (Optional[List[str]]): List of file extensions to filter resumes. 
                                                   Defaults to common resume formats.
        
        Returns:
            List of dictionaries with resume metadata:
            [
                {
                    'file_path': 'absolute/path/to/resume.pdf',
                    'role_category': 'data_scientist',
                    'filename': 'John_Doe_Resume.pdf',
                    'extension': '.pdf'
                },
                ...
            ]
        '''
        if file_extensions is None:
            file_extensions = ['.pdf', '.docx', '.txt']
        
        logger.info(f"Dataset Root Path: {self.dataset_root}")

        # Folders to skip (not role categories)
        skip_folders = {'processed_resumes', 'cleaned_markdown', 'output', '.git', '__pycache__'}

        resumes = []
        # Iterate through role folders
        for role_folder in self.dataset_root.iterdir():
            logger.info(f"Role folder: {role_folder}")
            if not role_folder.is_dir():
                continue

            # Skip non-role folders
            if role_folder.name in skip_folders:
                logger.debug(f"Skipping non-role folder: {role_folder.name}")
                continue

            logger.info(f"Scanning role folder: {role_folder.name}")
            
            # Find all resume files in this role folder
            for file_path in role_folder.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in file_extensions:
                    resume_info = {
                        'file_path': str(file_path.absolute()),
                        'role_category': role_folder.name,
                        'filename': file_path.name,
                        'extension': file_path.suffix.lower()
                    }
                    resumes.append(resume_info)
                    logger.debug(f"Found resume: {file_path.name} in {role_folder.name}")
    
        logger.info(f"Total resumes found: {len(resumes)}")
        return resumes

        
    def get_single_resume(self, role_category: str, filename: str) -> Optional[Dict]:
        """
        Get a specific resume file.
        
        Args:
            role_category: Name of the role folder
            filename: Name of the resume file
        
        Returns:
            Resume metadata dictionary or None if not found
        """
        file_path = self.dataset_root / role_category / filename
        
        if file_path.is_file():
            resume ={
                'file_path': str(file_path.absolute()),
                'role_category': role_category,
                'filename': filename,
                'extension': file_path.suffix.lower()
            }
            logger.debug(f"Found resume: {file_path.name} in {role_category}")
            
            return resume
        else:
            logger.error(f"Resume not found: {file_path}")
            return None
        
    
    def validate_dataset_structure(self) -> Dict[str, int]:
        """
        Validate the dataset structure and return statistics.
        
        Returns:
            Dictionary with role categories and resume counts:
            {
                'data_scientist': 5,
                'fullstack_engineer': 3,
                'it': 7,
                'total': 15
            }
        """
        stats = {}
        total = 0
        
        for role_folder in self.dataset_root.iterdir():
            if not role_folder.is_dir():
                continue
            
            # Count resume files
            resume_count = len([
                f for f in role_folder.rglob('*')
                if f.is_file() and f.suffix.lower() in ['.pdf', '.docx', '.doc']
            ])
            
            stats[role_folder.name] = resume_count
            total += resume_count
        
        stats['total'] = total
        logger.info(f"Dataset validation: {stats}")
        return stats
    
    def get_available_roles(self) -> List[str]:
        """
        Get list of available role categories in the dataset.
        
        Returns:
            List of role category names
        """
        roles = [
            folder.name for folder in self.dataset_root.iterdir()
            if folder.is_dir()
        ]
        logger.info(f"Available roles: {roles}")
        return roles
            
    
    
if __name__ == "__main__":
    dataset_root_path = os.getenv("DATASET_ROOT_PATH")
    doc_loader = DocumentLoader(dataset_root_path)
    resumes = doc_loader.get_all_resumes()
    stats = doc_loader.validate_dataset_structure()
    print(stats)