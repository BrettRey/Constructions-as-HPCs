PYTHON=python3

.PHONY: all clean

# Default target: run full analysis pipeline
all: data_download features profile prcurve session

# Download UD corpora
data_download:
	$(PYTHON) src/10_download_ud.py

# Extract let‑alone features and stats
features:
	$(PYTHON) src/11_extract_let_alone.py

# Generate the profile figure
profile:
	$(PYTHON) src/12_profile_plot.py

# Run cross‑corpus prediction and plot PR curves
prcurve:
	$(PYTHON) src/13_predict_prcurve.py

# Record session information (installed package versions)
session:
	$(PYTHON) -c "import sys, json, pkg_resources, platform, datetime, pathlib; print('Python', sys.version); print('Platform', platform.platform()); print('Date', datetime.datetime.now().isoformat()); print('Packages'); [print(f'{p.project_name}=={p.version}') for p in pkg_resources.working_set]" > SESSION.txt

# Remove downloaded data and outputs
clean:
	rm -rf data/ud/gum data/ud/ewt out images SESSION.txt
