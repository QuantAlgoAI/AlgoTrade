import os
import shutil
import time
import schedule
import zipfile
from tqdm import tqdm  # Import tqdm for progress bar

# Set the source directory (your Python project folder)
SOURCE_DIR = r"C:\Users\rajiv\OneDrive\PERSONAL\Desktop\LAST_PROJECT"

# Set the main backup directory
BACKUP_BASE_DIR = r"D:\Backup\LAST_PROJECT"

# Validate the source directory
if not os.path.exists(SOURCE_DIR):
    print(f"Warning: Source directory '{SOURCE_DIR}' does not exist. Waiting for it to be created...")
    while not os.path.exists(SOURCE_DIR):
        time.sleep(10)  # Check every 10 seconds

def backup_project():
    """Backup the project folder by creating a new ZIP archive with timestamp."""
    start_time = time.time()
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    backup_zip_path = os.path.join(BACKUP_BASE_DIR, f"backup_{timestamp}.zip")
    
    os.makedirs(BACKUP_BASE_DIR, exist_ok=True)
    print(f"Creating backup: {backup_zip_path}")
    
    try:
        with zipfile.ZipFile(backup_zip_path, 'w', zipfile.ZIP_DEFLATED) as backup_zip:
            for foldername, subfolders, filenames in os.walk(SOURCE_DIR):
                for filename in tqdm(filenames, desc="Backing up", unit="file"):
                    file_path = os.path.join(foldername, filename)
                    arcname = os.path.relpath(file_path, SOURCE_DIR)
                    backup_zip.write(file_path, arcname)
    except Exception as e:
        print(f"Error during backup: {e}")
        return
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Backup completed at {time.strftime('%Y-%m-%d %H:%M:%S')} in {elapsed_time:.2f} seconds")

def countdown_timer(minutes):
    """Display a countdown timer in minutes for the next backup."""
    for remaining in range(minutes, 0, -1):
        print(f"Next backup in {remaining} minute(s)", end="\r")
        time.sleep(60)
    print("Starting backup now...")

def start_backup_now():
    """Allow the user to manually start a backup."""
    user_input = input("Do you want to start a backup now? (yes/no): ").strip().lower()
    if user_input in ['yes', 'y']:
        backup_project()

# Schedule the backup every 30 minutes
schedule.every(30).minutes.do(backup_project)

print("Automatic backup started. Press Ctrl+C to stop.")

# Run an immediate backup on startup
backup_project()

while True:
    schedule.run_pending()
    countdown_timer(30)
    start_backup_now()
