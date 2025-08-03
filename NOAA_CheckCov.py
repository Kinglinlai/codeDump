import os
import csv

# Configuration
BASE_DIR = 'NOAA_GSOD'
YEARS = list(range(2000, 2025))  # 2000 to 2024 inclusive
MIN_YEARS = len(YEARS)  # 25 years

def check_station_coverage():
    # Get all stations from the station list
    try:
        with open(os.path.join(BASE_DIR, 'station_list.txt'), 'r') as f:
            stations = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print("Error: station_list.txt not found. Trying to identify stations from year directories...")
        stations = set()
        for year in YEARS:
            year_dir = os.path.join(BASE_DIR, str(year))
            if os.path.exists(year_dir):
                stations.update([f for f in os.listdir(year_dir) if f.endswith('.csv')])
        stations = sorted(stations)
    
    print(f"Checking coverage for {len(stations)} stations across {MIN_YEARS} years...")
    
    # Check coverage for each station
    complete_stations = []
    missing_data = {}
    
    for station in stations:
        missing_years = []
        for year in YEARS:
            file_path = os.path.join(BASE_DIR, str(year), station)
            if not os.path.exists(file_path):
                missing_years.append(year)
        
        if not missing_years:
            complete_stations.append(station)
        else:
            missing_data[station] = missing_years
    
    # Generate report
    report_path = os.path.join(BASE_DIR, 'station_coverage_report.csv')
    with open(report_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Station ID', 'Years Missing', 'Years Present', 'Coverage Status'])
        
        for station in stations:
            missing_years = missing_data.get(station, [])
            present_count = MIN_YEARS - len(missing_years)
            status = "COMPLETE" if present_count == MIN_YEARS else "INCOMPLETE"
            writer.writerow([
                station,
                len(missing_years),
                present_count,
                status
            ])
    
    print("\n" + "="*50)
    print(f"Station coverage report saved to: {report_path}")
    print(f"Stations with complete data ({MIN_YEARS} years): {len(complete_stations)}")
    print(f"Stations with incomplete data: {len(missing_data)}")
    
    # Print stations with complete coverage
    if complete_stations:
        print("\nStations with complete coverage (2000-2024):")
        for station in complete_stations:
            print(f"- {station}")
    else:
        print("\nNo stations have complete coverage for all years")
    
    # Print stations with most missing years
    if missing_data:
        print("\nStations with the most missing years:")
        sorted_missing = sorted(missing_data.items(), key=lambda x: len(x[1]), reverse=True)[:5]
        for station, years in sorted_missing:
            print(f"- {station}: Missing {len(years)} years (e.g. {years[:3]}{'...' if len(years)>3 else ''})")

if __name__ == "__main__":
    check_station_coverage()