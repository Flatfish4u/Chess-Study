import subprocess
import os

# Path to your local Chess-Study repository
repo_path = '/Users/benjaminrosales/Desktop/Chess-Study Coding'

# Set Git user name and email for the specific repository
def set_git_local_config(repo_path):
    try:
        # Change the working directory to the repository
        os.chdir(repo_path)

        # Set the user name for the repository
        subprocess.run(
            ["git", "config", "user.name", "Benjamin Rosales"],
            check=True,
        )
        # Set the user email for the repository
        subprocess.run(
            ["git", "config", "user.email", "benjammin2014@gmail.com"],
            check=True,
        )
        print("Git local configuration updated successfully for the repository.")
    except subprocess.CalledProcessError as e:
        print("Error configuring Git locally:", e)
    except FileNotFoundError:
        print("Repository path not found. Please check the path:", repo_path)

# Run the function with the specified path
set_git_local_config('/Users/benjaminrosales/Desktop/Chess-Study Coding')
