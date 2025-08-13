import os
import requests
from bs4 import BeautifulSoup

# Base URL template
BASE_URL = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/{year}/"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

def get_stations_for_year(year):
    """Retrieve all station CSV links for a given year"""
    print(f"Fetching stations for {year}...")
    try:
        response = requests.get(BASE_URL.format(year=year), headers=HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract all CSV links
        return [a['href'] for a in soup.find_all('a') if a['href'].endswith('.csv')]
    except Exception as e:
        print(f"Error retrieving stations for {year}: {str(e)}")
        return []

def main():
    # Create output directory
    os.makedirs('NOAA_GSOD', exist_ok=True)
    
    # Years to compare
    target_years = [2000,2002,2005, 2007, 2009, 2011, 2013, 2016, 2018, 2020, 2023, 2024]
    station_sets = {}
    
    # Get stations for each target year
    for year in target_years:
        stations = get_stations_for_year(year)
        station_sets[year] = set(stations)
        print(f"Found {len(stations)} stations in {year}")
        
        # Save individual year list
        with open(f'NOAA_GSOD/stations_{year}.txt', 'w') as f:
            f.write('\n'.join(stations))
    
    # Find stations present in all three years
    common_stations = station_sets[2000] & station_sets[2009] & station_sets[2018]
    
    # Save common stations
    with open('NOAA_GSOD/common_stations.txt', 'w') as f:
        f.write('\n'.join(sorted(common_stations)))
    
    print(f"\nFound {len(common_stations)} stations common to all three years")
    print("Results saved in:")
    print("- NOAA_GSOD/stations_2000.txt")
    print("- NOAA_GSOD/stations_2009.txt")
    print("- NOAA_GSOD/stations_2018.txt")
    print("- NOAA_GSOD/common_stations.txt")

if __name__ == "__main__":
    main()