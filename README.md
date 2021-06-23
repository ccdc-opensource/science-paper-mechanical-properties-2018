# Mechanical Properties Script


Supporting code for https://doi.org/10.1039/C8CE00454D

---

# Using Script

The `mechanical_properties.py` script can be used through the command line.

First we activate the miniconda that contains the `csd-python-api`.

```commandline
"C:\Program Files\CCDC\Python_API_2021\miniconda\condabin\activate.bat"
```

The script can then be run for a given ref code from the CSD or a `.cif` file.

```commandline
python mechanical_properties.py REF_CODE|file_name.cif
```

Output will then be printed to the console.

By default the tool will scan for "flat" planes between miller planes [-2 -2 -2] and [2 2 2]. If none are found, it will
extend the search from [-4 -4 -4] to [4 4 4]. This range can be explicitly set, or the additional scan on failure
suppressed.

```commandline
python mechanical_properties.py REF_CODE --p low 
python mechanical_properties.py REF_CODE --p high 
python mechanical_properties.py REF_CODE --s true 
```

# Dependencies:


- [CSD-Python-API](https://ccdc-cambridge.slack.com/archives/DQ88Q1VNJ/p1624458665065900)
