import os
import time
import requests
from datetime import datetime

# S3 HTTP endpoint (public, no auth needed)
S3_HTTP_BASE = "https://noaa-gsod-pds.s3.amazonaws.com/{year}/{station}.csv"

# Be a good citizen with a clear UA
HEADERS = {
    "User-Agent": "gsod-station-downloader/1.0 (+contact@example.com)"
}

# Years to fetch (match your previous script; change as you like)
START_YEAR = 2000
END_YEAR = 2024  # or datetime.utcnow().year

# Simple exponential backoff fetcher
def fetch_with_retries(url, headers=None, attempts=6, timeout=60):
    for i in range(attempts):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            # Retry on common transient statuses
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(60, 2 ** i))
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if i == attempts - 1:
                raise
            time.sleep(min(60, 2 ** i))

def download_station_data():
    station_id = input("Enter station ID (e.g. 01001099999 or USW00094728): ").strip()
    if not station_id:
        print("No station ID provided. Exiting.")
        return

    # normalize to bare ID (we append .csv when formatting URL/file)
    station_id = station_id.replace(".csv", "")

    os.makedirs("NOAA_GSOD", exist_ok=True)
    print(f"\nDownloading GSOD (S3) for station: {station_id}")

    total_downloaded = 0
    total_skipped = 0
    total_errors = 0

    for year in range(START_YEAR, END_YEAR + 1):
        year_dir = os.path.join("NOAA_GSOD", str(year))
        os.makedirs(year_dir, exist_ok=True)
        out_path = os.path.join(year_dir, f"{station_id}.csv")

        if os.path.exists(out_path):
            print(f"Already exists: {year}/{station_id}.csv - skipping")
            total_skipped += 1
            continue

        url = S3_HTTP_BASE.format(year=year, station=station_id)

        try:
            resp = fetch_with_retries(url, headers=HEADERS)
            # If the object doesn't exist, S3 returns 404 — treat as "no data that year"
            # (requests.raise_for_status already handled non-200s)

            # Some years legitimately have no data: handle tiny "AccessDenied" HTML, etc.
            text_head = resp.text[:64].lower()
            if resp.status_code == 200 and not text_head.startswith("<!doctype") and "accessdenied" not in text_head:
                with open(out_path, "wb") as f:
                    f.write(resp.content)
                print(f"Downloaded {year}/{station_id}.csv")
                total_downloaded += 1
            else:
                # Guard: if the content looks wrong, treat as missing
                print(f"No data (or not a CSV): {year}/{station_id}.csv")
                total_skipped += 1

            time.sleep(0.1)  # gentle pacing

        except requests.HTTPError as e:
            if getattr(e.response, "status_code", None) == 404:
                print(f"Data not available: {year}/{station_id}.csv")
                total_skipped += 1
            else:
                code = getattr(e.response, "status_code", "HTTPError")
                print(f"HTTP Error {code} for {year}/{station_id}.csv")
                total_errors += 1
        except Exception as e:
            print(f"Error downloading {year}/{station_id}.csv: {e}")
            total_errors += 1

    print("\nDownload summary:")
    print(f"Station: {station_id}")
    print(f"Years processed: {START_YEAR}-{END_YEAR}")
    print(f"Files downloaded: {total_downloaded}")
    print(f"Files skipped: {total_skipped}")
    print(f"Errors encountered: {total_errors}")
    print(f"\nData location: NOAA_GSOD/{{YEAR}}/{station_id}.csv")

if __name__ == "__main__":
    while True:
        download_station_data()
