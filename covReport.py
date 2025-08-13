#!/usr/bin/env python3
import os
import csv
import sys

# Features we care about
FEATURE_COLUMNS = [
    "TEMP",
    "DEWP",
    "SLP",
    "STP",
    "VISIB",
    "WDSP",
    "PRCP",
]

# Path to your coverage report (adjust if needed)
REPORT_CSV = os.path.join('NOAA_GSOD', 'station_coverage_report.csv')

def load_complete_stations_with_features(report_path, required_feats):
    """
    Reads the coverage report CSV and returns a list of station IDs
    that have COMPLETE coverage and report all of required_feats.
    """
    matched = []
    with open(report_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = row['Coverage Status'].strip().upper()
            if status != 'COMPLETE':
                continue

            # Parse the semicolon-separated measured features
            measured = {feat.strip() for feat in row['Measured Features'].split(';') if feat.strip()}

            # Check if all required features are present
            if all(feat in measured for feat in required_feats):
                matched.append(row['Station ID'])

    return matched

def main():
    if not os.path.exists(REPORT_CSV):
        print(f"Error: coverage report not found at {REPORT_CSV}", file=sys.stderr)
        sys.exit(1)

    stations = load_complete_stations_with_features(REPORT_CSV, FEATURE_COLUMNS)
    if not stations:
        print("No stations found meeting all criteria.")
        sys.exit(0)

    print("First 20 COMPLETE stations reporting all requested features:")
    for station in stations[:20]:
        print(station)
    
    # Save all marked stations to a file
    output_file = 'ac_stations.txt'
    try:
        with open(output_file, 'w') as f:
            for station_id in stations:
                f.write(f"{station_id}\n")
        print(f"\nSaved all {len(stations)} stations to {output_file}")
    except IOError as e:
        print(f"Error: Could not write to file {output_file}. Reason: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()