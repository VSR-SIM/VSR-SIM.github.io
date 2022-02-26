# VSR-SIM
Spatio-temporal Vision Transforer for Super-resolution Microscopy

## Run inference server locally
On Linux
```
 FLASK_APP=index.py FLASK_ENV=development FLASK_RUN_PORT=80 flask run --host=0.0.0.0
```

On Windows
```
 $env:FLASK_APP = "index.py"; $env:FLASK_ENV = 'development'; $env:FLASK_RUN_PORT=80; flask run --host=0.0.0.0
```
