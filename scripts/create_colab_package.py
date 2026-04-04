"""
Creates a zip package of your project for upload to Google Colab.
Excludes large data files and only includes what's needed for training.

Usage:
    python scripts/create_colab_package.py
"""

import zipfile
import os
from pathlib import Path

def create_package():
    output_zip = 'APS360_project.zip'
    
    # Directories and files to include
    include_dirs = ['src', 'scripts', 'configs', 'data/processed', 'data/new_test']
    include_files = ['requirements.txt']
    
    # Patterns to exclude (large raw data)
    exclude_patterns = [
        '__pycache__', '.pyc', '.git',
        'Kaggle_dataset', 'Stanford_dataset',
        'checkpoints', 'logs', 'results',
        '.ipynb_checkpoints'
    ]
    
    def should_exclude(path_str):
        return any(pat in path_str for pat in exclude_patterns)
    
    print(f'Creating {output_zip}...')
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add directories
        for dir_path in include_dirs:
            if not Path(dir_path).exists():
                print(f'  Skipping (not found): {dir_path}')
                continue
            for file_path in Path(dir_path).rglob('*'):
                if file_path.is_file() and not should_exclude(str(file_path)):
                    zf.write(file_path)
                    
        # Add root files
        for f in include_files:
            if Path(f).exists():
                zf.write(f)
        
        # Add the notebook itself
        if Path('APS360_Training_Colab.ipynb').exists():
            zf.write('APS360_Training_Colab.ipynb')
    
    size_mb = Path(output_zip).stat().st_size / 1e6
    print(f'\nCreated: {output_zip} ({size_mb:.1f} MB)')
    print('\nUpload this file to Google Drive, then open APS360_Training_Colab.ipynb in Colab.')

if __name__ == '__main__':
    create_package()
