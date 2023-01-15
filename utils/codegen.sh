wget https://github.91chi.fun//https://github.com/qhduan/code-clippy-vscode/archive/refs/heads/master.zip &
wget https://github.91chi.fun//https://github.com/qhduan/CodeGen/archive/refs/heads/main.zip && unzip CodeGen-main.zip && cd CodeGen-main

if [! -d "venv"]; then
    /data/python_env/python3.8 -m venv venv
fi
echo "venv success"
source venv/bin/activate && pip install -r requirements.txt 
echo "install success"
python main.py
python run.py &
cd .. && unzip code-clippy-vscode.zip && cd code-clippy-vscode 
vsce package
code --enable-proposed-api qhduan.codegen