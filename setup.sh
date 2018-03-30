
# install reqiured python libraries
pip install -r requirements.txt

# create required empty directories
mkdir -p "figures/"
mkdir -p "data/_cache/"

# unzip data
unzip "data/data.zip" -d "data/"