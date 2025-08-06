import os
import csv
import re
import itertools
from tqdm import tqdm

# Configuration
BASE_DIR = 'NOAA_GSOD'
YEARS = list(range(2000, 2025))  # 2000 to 2024 inclusive
MIN_YEARS = len(YEARS)  # 25 years

# Pattern to detect placeholder missing values like '9999.9', '999.9', '99.9', etc.
PLACEHOLDER_RE = re.compile(r'^9+(\.9+)?$')

def get_measured_features(file_path):
    """
    Read the header + first five data rows of the given CSV file and
    return a set of feature names (columns, skipping the first two)
    for which there are at most 2 placeholder values in those rows.
    """
    with open(file_path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        sample_rows = list(itertools.islice(reader, 5))
    measured = set()
    for col_idx in range(2, len(header)):
        ph_count = sum(
            1 for row in sample_rows
            if PLACEHOLDER_RE.match(row[col_idx].strip())
        )
        if ph_count <= 2:
            measured.add(header[col_idx])
    return measured

def check_station_coverage():
    # Load station list or scan directories
    try:
        with open(os.path.join(BASE_DIR, 'station_list.txt'), 'r') as f:
            stations = [line.strip() for line in f]
    except FileNotFoundError:
        print("Warning: station_list.txt not found. Scanning directories for station files...")
        stations = set()
        for year in YEARS:
            year_dir = os.path.join(BASE_DIR, str(year))
            if os.path.isdir(year_dir):
                stations.update(
                    fname for fname in os.listdir(year_dir)
                    if fname.endswith('.csv')
                )
        stations = sorted(stations)

    print(f"Checking coverage for {len(stations)} stations across {MIN_YEARS} years...\n")

    complete_stations = []
    missing_data = {}
    station_features = {}

    # Wrap the station loop in tqdm
    for station in tqdm(stations, desc="Stations", unit="stn"):
        present_years = []
        features_intersection = None

        # Optionally wrap the inner year loop, too
        for year in tqdm(YEARS, desc=f"{station}", unit="yr", leave=False):
            file_path = os.path.join(BASE_DIR, str(year), station)
            if os.path.exists(file_path):
                present_years.append(year)
                feats = get_measured_features(file_path)
                features_intersection = feats if features_intersection is None else (features_intersection & feats)

        missing_years = [y for y in YEARS if y not in present_years]
        station_features[station] = features_intersection or set()

        if not missing_years:
            complete_stations.append(station)
        else:
            missing_data[station] = missing_years

    # Generate report CSV
    report_path = os.path.join(BASE_DIR, 'station_coverage_report.csv')
    with open(report_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'Station ID',
            'Years Missing',
            'Years Present',
            'Coverage Status',
            'Measured Features'
        ])

        for station in stations:
            missing_years = missing_data.get(station, [])
            present_count = MIN_YEARS - len(missing_years)
            status = "COMPLETE" if present_count == MIN_YEARS else "INCOMPLETE"
            feats = sorted(station_features[station])
            writer.writerow([
                station,
                len(missing_years),
                present_count,
                status,
                ";".join(feats)
            ])

    # Final summary
    print("\n" + "="*60)
    print(f"Station coverage report saved to: {report_path}")
    print(f"Stations with complete data ({MIN_YEARS} years): {len(complete_stations)}")
    print(f"Stations with incomplete data: {len(missing_data)}")

    if complete_stations:
        print("\nStations with complete coverage (2000–2024):")
        for st in complete_stations:
            print(f"- {st}")
    else:
        print("\nNo stations have complete coverage for all years.")

    if missing_data:
        print("\nTop 5 stations with most missing years:")
        top5 = sorted(missing_data.items(), key=lambda x: len(x[1]), reverse=True)[:5]
        for st, years in top5:
            preview = years[:3] + (['...'] if len(years) > 3 else [])
            print(f"- {st}: Missing {len(years)} years (e.g. {preview})")

if __name__ == "__main__":
    check_station_coverage()
