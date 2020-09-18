rm -rf .env
virtualenv -p $(which python3) .env
. .env/bin/activate
pip install -r requirements.txt
deactivate