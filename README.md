# MasterThesis

source code/py39_env/bin/activate
pip freeze --local > requirements.txt

deactivate 

virtualenv -p /usr/bin/python2.6 py26_env

pip install -r requirements.txt
