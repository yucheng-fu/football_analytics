# Mlflow commands
Start MLFlow tracking server
``` 
mlflow server --port 8080
```

Delete experiments and runs permanently using garbage collection
```
mlflow gc --tracking-uri "http://127.0.0.1:8080/"
```