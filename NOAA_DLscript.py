import os
import time
import requests
from bs4 import BeautifulSoup

# Base URL template
BASE_URL = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/{year}/"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

# Create main directory
os.makedirs('NOAA_GSOD', exist_ok=True)

# Step 1: Get first 50 stations from year 2000
print("Identifying first 50 stations from year 2000...")
station_list = []

try:
    response = requests.get(BASE_URL.format(year=2000), headers=HEADERS)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract all CSV links and take first 50
    all_links = [a['href'] for a in soup.find_all('a') if a['href'].endswith('.csv')]
    station_list = all_links[:50]
    
    print(f"Identified {len(station_list)} stations to track")
    # Save station list for reference
    with open('NOAA_GSOD/station_list.txt', 'w') as f:
        f.write('\n'.join(station_list))
    
except Exception as e:
    print(f"Error getting station list: {str(e)}")
    exit()

# Step 2: Download data for these stations across all years
for year in range(2000, 2025):
    year_dir = os.path.join('NOAA_GSOD', str(year))
    os.makedirs(year_dir, exist_ok=True)
    
    print(f"\nProcessing year {year} for {len(station_list)} stations...")
    downloaded = 0
    skipped = 0
    errors = 0
    
    for station in station_list:
        file_path = os.path.join(year_dir, station)
        
        # Skip existing files
        if os.path.exists(file_path):
            skipped += 1
            continue
            
        # Construct download URL
        csv_url = BASE_URL.format(year=year) + station
        
        try:
            response = requests.get(csv_url, headers=HEADERS)
            response.raise_for_status()
            
            # Save file if found
            with open(file_path, 'wb') as f:
                f.write(response.content)
            downloaded += 1
            print(f"Downloaded {station} for {year}")
            
            # Short delay between downloads
            time.sleep(0.3)
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                skipped += 1
                #print(f"Not found: {station} for {year}")
            else:
                errors += 1
                print(f"Error downloading {station} ({year}): HTTP {response.status_code}")
        except Exception as e:
            errors += 1
            print(f"Error downloading {station} ({year}): {str(e)}")
    
    print(f"Year {year} summary: Downloaded {downloaded}, Skipped {skipped}, Errors {errors}")

print("\nDownload completed! Stations saved in NOAA_GSOD directory")