import json
from utils.a import A
# from .a import A

with open("../utils/a.json", "rb") as f:
    t = json.loads(f.read())

print(A(**t).children)
