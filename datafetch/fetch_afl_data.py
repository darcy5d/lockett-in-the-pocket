#!/usr/bin/env python3
"""
Script to fetch the latest AFL data from the akareen/AFL-Data-Analysis repository
"""

import os
import requests
import zipfile
import io
import shutil
from pathlib import Path

def download_github_directory(user, repo, directory, destination):
    """
    Download a specific directory from a GitHub repository
    """
    # Create the destination directory if it doesn't exist
    os.makedirs(destination, exist_ok=True)
    
    print(f"Downloading {directory} from {user}/{repo}...")
    
    # First, get the contents of the directory using the GitHub API
    api_url = f"https://api.github.com/repos/{user}/{repo}/contents/{directory}"
    response = requests.get(api_url)
    
    if response.status_code != 200:
        print(f"Failed to get directory contents: {response.status_code}")
        print(response.json().get('message', ''))
        return False
    
    # Process each item in the directory
    contents = response.json()
    for item in contents:
        item_path = item['path']
        item_name = os.path.basename(item_path)
        item_dest = os.path.join(destination, item_name)
        
        if item['type'] == 'dir':
            # Recursively download subdirectories
            download_github_directory(user, repo, item_path, item_dest)
        else:
            # Download the file
            download_url = item['download_url']
            print(f"Downloading {item_name}...")
            file_response = requests.get(download_url)
            
            if file_response.status_code == 200:
                with open(item_dest, 'wb') as f:
                    f.write(file_response.content)
            else:
                print(f"Failed to download {item_name}: {file_response.status_code}")
    
    return True

def download_github_repo_zip(user, repo, destination):
    """
    Download the entire GitHub repository as a zip file
    """
    print(f"Downloading the entire {user}/{repo} repository...")
    
    # URL for downloading the repository as a zip
    zip_url = f"https://github.com/{user}/{repo}/archive/refs/heads/main.zip"
    response = requests.get(zip_url)
    
    if response.status_code != 200:
        print(f"Failed to download repository: {response.status_code}")
        return False
    
    # Create a BytesIO object from the content
    zip_file = io.BytesIO(response.content)
    
    # Extract all the contents to the destination directory
    with zipfile.ZipFile(zip_file) as z:
        z.extractall(destination)
    
    # The extracted contents will be in a subdirectory named "{repo}-main"
    # We'll rename this to just the repository name
    extracted_dir = os.path.join(destination, f"{repo}-main")
    final_dir = os.path.join(destination, repo)
    
    # Remove the final directory if it already exists
    if os.path.exists(final_dir):
        shutil.rmtree(final_dir)
    
    # Rename the extracted directory to the final directory name
    os.rename(extracted_dir, final_dir)
    
    print(f"Repository downloaded and extracted to {final_dir}")
    return True

def extract_data_directories(source_repo_dir, base_data_dir):
    """
    Extract only the data directories from the downloaded repository
    """
    # Source directories in the downloaded repository
    source_data_dir = os.path.join(source_repo_dir, "data")
    source_odds_dir = os.path.join(source_repo_dir, "odds_data")
    
    # Destination directories
    dest_data_dir = os.path.join(base_data_dir, "data")
    dest_odds_dir = os.path.join(base_data_dir, "odds_data")
    
    # Copy data directory
    if os.path.exists(source_data_dir):
        # Create destination directories if they don't exist
        os.makedirs(dest_data_dir, exist_ok=True)
        
        # Copy lineups, matches, and players directories
        for subdir in ["lineups", "matches", "players"]:
            source_subdir = os.path.join(source_data_dir, subdir)
            dest_subdir = os.path.join(dest_data_dir, subdir)
            
            if os.path.exists(source_subdir):
                if os.path.exists(dest_subdir):
                    shutil.rmtree(dest_subdir)
                print(f"Copying {subdir} directory...")
                shutil.copytree(source_subdir, dest_subdir)
    
    # Copy odds_data directory
    if os.path.exists(source_odds_dir):
        if os.path.exists(dest_odds_dir):
            shutil.rmtree(dest_odds_dir)
        print(f"Copying odds_data directory...")
        shutil.copytree(source_odds_dir, dest_odds_dir)
    
    print("Data extraction completed!")

def main():
    # GitHub repository details
    user = "akareen"
    repo = "AFL-Data-Analysis"
    
    # Get the project root directory (two levels up from this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Create data directories
    base_data_dir = os.path.join(project_root, "afl_data")
    temp_dir = os.path.join(project_root, "temp_download")
    os.makedirs(base_data_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Download the entire repository as a zip file
    # This is necessary especially for the players directory which has too many files
    # for the GitHub API to handle with the directory download method
    success = download_github_repo_zip(user, repo, temp_dir)
    
    if success:
        # Extract only the data directories that we need
        extract_data_directories(os.path.join(temp_dir, repo), base_data_dir)
        
        # Clean up temporary directory
        print("Cleaning up temporary files...")
        shutil.rmtree(temp_dir)
        
        print("Data fetching completed successfully!")
    else:
        print("Failed to download the repository.")

if __name__ == "__main__":
    main() 