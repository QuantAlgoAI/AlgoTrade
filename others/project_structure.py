import os
from collections import defaultdict

def get_project_structure(directory, output_file="Project Structure.txt"):
    file_count = defaultdict(int)
    folder_count = defaultdict(int)
    
    with open(output_file, "w", encoding="utf-8") as f:
        # Details section at the top
        f.write("Project Structure Report\n")
        f.write("===================================\n")
        f.write("- This report provides the directory structure of the project.\n")
        f.write("- The .venv directory has been skipped.\n")
        f.write("- The node_modules directory has been skipped.\n")
        f.write("- Duplicate files and folders are listed at the end.\n")
        f.write("===================================\n\n")
        
        for root, dirs, files in os.walk(directory):
            # Skip the .venv directory
            if ".venv" in dirs:
                dirs.remove(".venv")
            if "project_structure.py" in files:
                files.remove("project_structure.py")
            if "Project Structure.txt" in files:
                files.remove("Project Structure.txt")
            if "PS.txt" in files:
                files.remove("__pycache__")
            if ".pytest_cache" in dirs:
                dirs.remove(".pytest_cache")
            if "node_modules" in dirs:
                dirs.remove("node_modules")
            if "others" in dirs:
                dirs.remove("others")
            if ".git" in dirs:
                dirs.remove(".git")
          
            # Track duplicate folders
            folder_name = os.path.basename(root)
            folder_count[folder_name] += 1
            
            # Format directory structure
            level = root.replace(directory, "").count(os.sep)
            indent = "|   " * level + "|-- "
            f.write(f"{indent}{folder_name}/\n")
            
            sub_indent = "|   " * (level + 1) + "|-- "
            for file in files:
                file_count[file] += 1
                f.write(f"{sub_indent}{file}\n")
                
        # Report duplicate files and folders
        f.write("\nDuplicate Files:\n")
        for file, count in file_count.items():
            if count > 1:
                f.write(f"- {file} (found {count} times)\n")
        
        f.write("\nDuplicate Folders:\n")
        for folder, count in folder_count.items():
            if count > 1:
                f.write(f"- {folder} (found {count} times)\n")
    
    print(f"Project structure has been saved to '{output_file}'")

if __name__ == "__main__":
    project_directory = os.getcwd()  # Change this if needed
    get_project_structure(project_directory)
