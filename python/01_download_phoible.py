#!/usr/bin/env python3

"""
01_download_phoible.py
-----------------------

Download the PHOIBLE 2.0 phoneme inventory data and the Glottolog
languoid table into the `data/raw` directory.  The script first
checks whether the files already exist locally.  If they do, it
skips the download.  Otherwise it attempts to fetch the files via
HTTP.  Because the build environment may restrict outbound network
requests, the script traps download errors and reports them while
continuing if local copies are available.

The PHOIBLE CSV is large (~25 MB); ensure adequate disk space.

"""
import os
import sys
from urllib.error import URLError, HTTPError
from urllib.request import urlretrieve

# Target directories
RAW_DIR = os.path.join('data', 'raw')
os.makedirs(RAW_DIR, exist_ok=True)

# URLs for the resources
PHOIBLE_URL = 'https://github.com/phoible/dev/raw/master/data/phoible.csv'
GLOTTOLOG_ZIP_URL = 'https://cdstar.eva.mpg.de/bitstreams/EAEA0-2198-D710-AA36-0/glottolog_languoid.csv.zip'

# Local file names
PHOIBLE_FILE = os.path.join(RAW_DIR, 'phoible.csv')
GLOTTOLOG_ZIP = os.path.join(RAW_DIR, 'glottolog_languoid.csv.zip')
LANGUOID_CSV = os.path.join(RAW_DIR, 'languoid.csv')


def safe_download(url: str, dest: str) -> bool:
    """Attempt to download a file to `dest` and return True on success.

    If the download fails due to network problems, print a message
    and return False.  Partial downloads are removed.
    """
    try:
        print(f"[download] downloading {os.path.basename(dest)} …")
        urlretrieve(url, dest)
        print(f"[download] saved to {dest}")
        return True
    except (HTTPError, URLError) as e:
        print(f"[download] failed to download {url}: {e}")
        # remove partial file if exists
        try:
            os.remove(dest)
        except FileNotFoundError:
            pass
        return False


def main() -> None:
    # Download PHOIBLE CSV if needed
    if not os.path.exists(PHOIBLE_FILE):
        ok = safe_download(PHOIBLE_URL, PHOIBLE_FILE)
        if not ok:
            print(f"Error: Could not download PHOIBLE data and no local copy present at {PHOIBLE_FILE}")
            sys.exit(1)
    else:
        print(f"[download] using existing PHOIBLE CSV at {PHOIBLE_FILE}")

    # Download Glottolog languoid ZIP if needed
    if not os.path.exists(LANGUOID_CSV):
        if not os.path.exists(GLOTTOLOG_ZIP):
            ok = safe_download(GLOTTOLOG_ZIP_URL, GLOTTOLOG_ZIP)
            if not ok:
                print(f"Error: Could not download Glottolog languoid zip and no local copy present at {GLOTTOLOG_ZIP}")
                sys.exit(1)
        else:
            print(f"[download] using existing Glottolog zip at {GLOTTOLOG_ZIP}")
        # extract languoid.csv from zip
        import zipfile
        with zipfile.ZipFile(GLOTTOLOG_ZIP, 'r') as zf:
            print("[download] extracting languoid.csv …")
            zf.extract('languoid.csv', path=RAW_DIR)
            print(f"[download] extracted languoid.csv to {LANGUOID_CSV}")
    else:
        print(f"[download] using existing languoid.csv at {LANGUOID_CSV}")

    print("[download] all required files present")


if __name__ == '__main__':
    main()