mamba env remove -n tf
mamba env create -f req.yml -n tf
python -m ipykernel install --user --name tf --display-name "Python 3.9 (tf ENV)"