import os
import time
import requests
from bs4 import BeautifulSoup

# Base URL template
BASE_URL = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/{year}/"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

def download_station_data():
    # Get station ID from user
    station_id = input("Enter station ID (e.g. 01001099999): ").strip()
    if not station_id:
        print("No station ID provided. Exiting.")
        return
    
    # Add .csv extension if missing
    station_file = station_id if station_id.endswith('.csv') else f"{station_id}.csv"
    
    # Create main directory
    os.makedirs('NOAA_GSOD', exist_ok=True)
    
    print(f"\nDownloading data for station: {station_id}")
    total_downloaded = 0
    total_skipped = 0
    total_errors = 0
    
    for year in range(2000, 2025):
        year_dir = os.path.join('NOAA_GSOD', str(year))
        os.makedirs(year_dir, exist_ok=True)
        
        file_path = os.path.join(year_dir, station_file)
        
        # Skip existing files
        if os.path.exists(file_path):
            print(f"Already exists: {year}/{station_file} - skipping")
            total_skipped += 1
            continue
            
        # Construct download URL
        csv_url = BASE_URL.format(year=year) + station_file
        
        try:
            response = requests.get(csv_url, headers=HEADERS)
            response.raise_for_status()
            
            # Save file
            with open(file_path, 'wb') as f:
                f.write(response.content)
                
            total_downloaded += 1
            print(f"Downloaded {year}/{station_file}")
            
            # Short delay between downloads
            time.sleep(0.1)
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                print(f"Data not available: {year}/{station_file}")
                total_skipped += 1
            else:
                print(f"HTTP Error {response.status_code} for {year}/{station_file}")
                total_errors += 1
        except Exception as e:
            print(f"Error downloading {year}/{station_file}: {str(e)}")
            total_errors += 1
    
    print("\nDownload summary:")
    print(f"Station: {station_id}")
    print(f"Years processed: 2000-2024")
    print(f"Files downloaded: {total_downloaded}")
    print(f"Files skipped: {total_skipped}")
    print(f"Errors encountered: {total_errors}")
    print(f"\nData location: NOAA_GSOD/{{YEAR}}/{station_file}")

if __name__ == "__main__":
    while True:
        download_station_data()