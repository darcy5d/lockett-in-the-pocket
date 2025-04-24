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

def main():
    # GitHub repository details
    user = "akareen"
    repo = "AFL-Data-Analysis"
    
    # Create a data directory to store the downloaded data
    base_data_dir = "afl_data"
    os.makedirs(base_data_dir, exist_ok=True)
    
    # Option 1: Download specific directories
    # This is more efficient if you only need certain parts of the data
    directories_to_download = ["data", "odds_data"]
    
    for directory in directories_to_download:
        destination_dir = os.path.join(base_data_dir, directory)
        success = download_github_directory(user, repo, directory, destination_dir)
        if success:
            print(f"Successfully downloaded {directory} to {destination_dir}")
        else:
            print(f"Failed to download {directory}")
    
    # Option 2: Download the entire repository
    # This is simpler but downloads more data
    # Uncomment the following lines to use this option instead
    # download_github_repo_zip(user, repo, base_data_dir)
    
    print("Data fetching completed!")

if __name__ == "__main__":
    main() 