import re
import sys
from pathlib import Path

import requests
import pandas as pd
from bs4 import BeautifulSoup


URL = "https://datacenters.google/efficiency/"
OUTDIR = Path(".")


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


YEAR_RE = re.compile(r"^(20\d{2}) PUE Yearly Report$")
QUARTER_RE = re.compile(r"^Quarter\s+0?([1-4])$")

# Matches:
# Fleet 1.08 1.09
# Berkeley County, South Carolina 1.08 1.09
# London, England 1.26
ROW_RE = re.compile(
    r"^(?P<location>.+?)\s+"
    r"(?P<quarterly_pue>\d+(?:\.\d+)?)"
    r"(?:\s+(?P<ttm_pue>\d+(?:\.\d+)?))?$"
)


def clean_location_name(name: str) -> str:
    name = re.sub(r"\s+", " ", name).strip()

    replacements = {
        # Google's page currently spells this as "Changua"
        "Changua County, Taiwan": "Changhua County, Taiwan",

        # Older rows use New Albany, later rows use Central Ohio
        "New Albany, Ohio": "Central Ohio (New Albany), Ohio",

        # Older Google language sometimes used Lowcountry
        "Lowcountry, South Carolina": "Berkeley County, South Carolina",
    }

    return replacements.get(name, name)


def fetch_page_lines(url: str) -> list[str]:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove junk that can pollute text extraction
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in soup.get_text("\n").splitlines()
    ]

    lines = [line for line in lines if line]

    return lines


def parse_google_pue(lines: list[str]) -> pd.DataFrame:
    records = []

    current_year = None
    current_quarter = None
    section = None  # None, "fleet", or "campus"

    for line in lines:
        year_match = YEAR_RE.match(line)
        if year_match:
            current_year = int(year_match.group(1))
            current_quarter = None
            section = None
            continue

        quarter_match = QUARTER_RE.match(line)
        if quarter_match:
            current_quarter = int(quarter_match.group(1))
            section = None
            continue

        if "Fleet wide PUE" in line and "Quarterly PUE" in line:
            section = "fleet"
            continue

        if line.startswith("Campuses Quarterly PUE"):
            section = "campus"
            continue

        # End of a campus/fleet table block
        if (
            line.startswith("*We report")
            or line.startswith("For Q")
            or line == "##"
        ):
            section = None
            continue

        if current_year is None or current_quarter is None or section is None:
            continue

        if "Quarterly PUE" in line or "Trailing twelve-month" in line:
            continue

        row_match = ROW_RE.match(line)
        if not row_match:
            continue

        location = clean_location_name(row_match.group("location"))

        quarterly_pue = float(row_match.group("quarterly_pue"))
        ttm_text = row_match.group("ttm_pue")
        ttm_pue = float(ttm_text) if ttm_text is not None else pd.NA

        if section == "fleet":
            location = "Fleet"

        records.append(
            {
                "year": current_year,
                "quarter": current_quarter,
                "period": f"Q{current_quarter} {current_year}",
                "location": location,
                "quarterly_pue": quarterly_pue,
                "ttm_pue": ttm_pue,
            }
        )

        # Fleet section has only one data row
        if section == "fleet":
            section = None

    return pd.DataFrame(records)


def make_wide(df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    wide = df.pivot_table(
        index=["year", "quarter", "period"],
        columns="location",
        values=value_column,
        aggfunc="first",
    )

    wide = wide.reset_index()
    wide = wide.sort_values(["year", "quarter"])
    wide = wide.drop(columns=["year", "quarter"])
    wide = wide.set_index("period")

    # Put Fleet first if it exists
    columns = list(wide.columns)
    if "Fleet" in columns:
        columns = ["Fleet"] + [col for col in columns if col != "Fleet"]
        wide = wide[columns]

    return wide


def main():
    lines = fetch_page_lines(URL)

    df_long = parse_google_pue(lines)

    if df_long.empty:
        debug_path = OUTDIR / "google_pue_debug_lines.txt"
        debug_path.write_text("\n".join(lines[:300]), encoding="utf-8")

        print("ERROR: No PUE rows were parsed.")
        print(f"I saved the first 300 extracted lines here: {debug_path}")
        print("Open that file and check whether the Google page text loaded correctly.")
        sys.exit(1)

    df_long = df_long.sort_values(
        ["year", "quarter", "location"],
        ascending=[True, True, True],
    )

    long_path = OUTDIR / "google_pue_long_clean.csv"
    quarterly_path = OUTDIR / "google_pue_quarterly_wide.csv"
    ttm_path = OUTDIR / "google_pue_ttm_wide.csv"

    df_long.to_csv(long_path, index=False)

    quarterly_wide = make_wide(df_long, "quarterly_pue")
    quarterly_wide.to_csv(quarterly_path)

    ttm_wide = make_wide(df_long, "ttm_pue")
    ttm_wide.to_csv(ttm_path)

    print("Saved:")
    print(f"  {long_path}")
    print(f"  {quarterly_path}")
    print(f"  {ttm_path}")

    print()
    print("Rows parsed:", len(df_long))
    print("Periods parsed:", df_long["period"].nunique())
    print("Locations parsed:", df_long["location"].nunique())

    print()
    print("Preview of quarterly wide CSV:")
    print(quarterly_wide.tail())


if __name__ == "__main__":
    main()