import subprocess

venv = \
{"Miniconda3-py37_4.9.2-Linux-x86_64.sh":"/data/pyvenv/py37",
"Miniconda3-py39_4.9.2-Linux-x86_64.sh":"/data/pyvenv/py39",
"Miniconda3-py38_4.9.2-Linux-x86_64.sh":"/data/pyvenv/py38",
"Miniconda3-4.5.1-Linux-x86_64.sh":"/data/pyvenv/py36"}

for i,v in venv.items():
    install_venv = "%s -b -p %s" % (i,v)
    subprocess.run(install_venv.split())
    