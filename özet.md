# Final Folder Structure

```
app
|_ train 
|_ inference
|_ cli.py

setup.py
```

# Command to Run

```
app train {config_path}

app inference {config_path}
```

# Notes

- setup.py ismi değiştirilmemeli.

- Ayrı bir requirements dosyası olmayacak, her şey setup.py üzerinden çalışacak. Sadece ```pip install -e .``` çalıştırarak bütün kod kullanıma hazır package haline gelmeli.

- pyproject.toml'a da göz atabilirsiniz. setup.py'nin güncel versiyonu.