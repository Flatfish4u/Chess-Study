import requests
import os
import sys

def download_model():
    url = "https://raw.githubusercontent.com/Flatfish4u/elocator/main/src/elocator/api/model/model.pth"
    output_path = os.path.join('complexity', 'models', 'model.pth')
    
    print(f"Downloading model from {url}")
    print(f"Saving to {output_path}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get total size if available
        total_size = int(response.headers.get('content-length', 0))
        current_size = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                current_size += len(chunk)
                if total_size:
                    progress = (current_size / total_size) * 100
                    print(f"\rDownloading: {progress:.1f}%", end='')
                    
        print("\nDownload complete!")
        
        # Verify file size
        file_size = os.path.getsize(output_path)
        print(f"File size: {file_size/1024/1024:.2f} MB")
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_model()