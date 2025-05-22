import os
import shutil
import datetime
import sys
import fnmatch

def should_ignore(path):
    """Check if the path should be ignored in backup"""
    ignore_patterns = [
        # Virtual environments
        '.venv', 'venv', 'env', 'ENV',
        # Cache directories
        '__pycache__', '.pytest_cache', '.coverage',
        # IDE specific
        '.idea', '.vscode', '.vs',
        # Log files
        '*.log', 'logs',
        # Temporary files
        '*.tmp', '*.temp', '*.swp',
        # Backup files
        'backup_*',
        # Git
        '.git', '.gitignore',
        # System files
        '.DS_Store', 'Thumbs.db'
    ]
    
    # Check if path matches any ignore pattern
    for pattern in ignore_patterns:
        if fnmatch.fnmatch(path, pattern) or pattern in path.split(os.sep):
            return True
    return False

def create_backup():
    """Create a backup of the project files"""
    try:
        # Create backup directory with timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = f'backup_{timestamp}'
        os.makedirs(backup_dir, exist_ok=True)
        
        # Get all files in current directory
        for root, dirs, files in os.walk('.'):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if not should_ignore(d)]
            
            # Create relative path for backup
            rel_path = os.path.relpath(root, '.')
            if rel_path == '.':
                backup_path = backup_dir
            else:
                backup_path = os.path.join(backup_dir, rel_path)
                os.makedirs(backup_path, exist_ok=True)
            
            # Copy files
            for file in files:
                if not should_ignore(file):
                    src_file = os.path.join(root, file)
                    dst_file = os.path.join(backup_path, file)
                    try:
                        shutil.copy2(src_file, dst_file)
                        print(f"Backed up: {src_file}")
                    except Exception as e:
                        print(f"Error backing up {src_file}: {str(e)}")
        
        # Create requirements.txt if it doesn't exist
        if not os.path.exists(os.path.join(backup_dir, 'requirements.txt')):
            requirements = [
                'pandas',
                'numpy',
                'requests',
                'pyotp',
                'python-telegram-bot',
                'matplotlib',
                'scipy',
                'SmartApi'
            ]
            with open(os.path.join(backup_dir, 'requirements.txt'), 'w') as f:
                f.write('\n'.join(requirements))
            print("Created requirements.txt")
        
        # Create a README with backup information
        readme_content = f"""Project Backup
Created: {timestamp}

This is an automated backup of the trading application.
Contains all source code, configuration files, and necessary data.

Excluded from backup:
- Virtual environments (.venv, venv, env)
- Cache files (__pycache__, .pytest_cache)
- Log files and directories
- IDE specific files (.idea, .vscode)
- Temporary files
- Git related files
- System files

To restore:
1. Copy all files to your project directory
2. Install requirements: pip install -r requirements.txt
3. Update API credentials in config.py
"""
        
        with open(os.path.join(backup_dir, 'README.txt'), 'w') as f:
            f.write(readme_content)
        
        # Calculate and display backup size
        total_size = get_dir_size(backup_dir)
        print(f"\nBackup completed successfully in directory: {backup_dir}")
        print(f"Total size: {total_size / (1024*1024):.2f} MB")
        
        return backup_dir
        
    except Exception as e:
        print(f"Error during backup: {str(e)}")
        raise

def get_dir_size(path):
    """Calculate total size of directory in bytes"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):  # Skip symlinks
                total_size += os.path.getsize(fp)
    return total_size

if __name__ == "__main__":
    try:
        backup_dir = create_backup()
        print(f"\nBackup location: {os.path.abspath(backup_dir)}")
    except Exception as e:
        print(f"Backup failed: {str(e)}")
        sys.exit(1) 