# spatial-sector-dashboard

A shiny dashboard demo for pypsa-eur results.

# Deploy locally

To deploy locally, and avoid any cartopy conflicts, download the environment.yaml file from [pypsa-eur](https://github.com/PyPSA/pypsa-eur/blob/master/envs/environment.yaml) and install the environment with the name `dashboard`. This can take few minutes.  
```bash
wget https://github.com/PyPSA/pypsa-eur/blob/83a01ad4f5afe3e02890374272b7c9b9f55b139a/envs/environment.yaml
conda env create -n dashboard --file environment.yaml
```

Afterwards, activate the conda environment and install the requirement.txt.
```bash
conda activate dashboard
pip install -r requirements.txt
```

Deploy now the streamlit application locally:
```bash
streamlit run streamlit_app.py
```