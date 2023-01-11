import importlib

if importlib.util.find_spec("scipy.sparse") == None:
    print("gg")
else:
    print("tes")
    

print(importlib.util.find_spec("scipy.sparse"))