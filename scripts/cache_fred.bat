@echo off
REM Download FRED macroeconomic series to data\fred\ as CSV files.
REM
REM Uses the FRED REST API (api.stlouisfed.org), which is more reliable
REM than the public fredgraph.csv endpoint (Cloudflare rate-limits us
REM from some Windows setups). Requires a free API key:
REM
REM   1. Register at https://fredaccount.stlouisfed.org/apikeys
REM   2. Copy your 32-character key
REM   3. Set env var in this terminal:  set FRED_API_KEY=yourkeyhere
REM   4. Run this script:               scripts\cache_fred.bat
REM
REM Re-run periodically to refresh.

setlocal
if "%FRED_API_KEY%"=="" (
    echo ERROR: FRED_API_KEY is not set.
    echo.
    echo Get a free key at https://fredaccount.stlouisfed.org/apikeys
    echo Then run:  set FRED_API_KEY=yourkey
    echo And re-run this script.
    exit /b 1
)

set OUT=data\fred
if not exist "%OUT%" mkdir "%OUT%"

set SERIES=CPIAUCSL UNRATE FEDFUNDS DGS10 DGS3MO BAA INDPRO VIXCLS

for %%S in (%SERIES%) do (
    echo Downloading %%S...
    curl -sSL --max-time 30 ^
        "https://api.stlouisfed.org/fred/series/observations?series_id=%%S&api_key=%FRED_API_KEY%&file_type=json&observation_start=1970-01-01" ^
        -o "%OUT%\%%S.json"
    if errorlevel 1 (
        echo   FAILED - %%S
    ) else (
        for %%F in ("%OUT%\%%S.json") do echo   OK - %%~zF bytes
    )
)

echo.
echo Done. Files in %OUT%:
dir /b "%OUT%"
endlocal
