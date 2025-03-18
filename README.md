## Installing the modules

If you have multiple python versions then use this command:

```bash
py -<python version> -m pip install -r requirements.txt
```

else you can use the following command:

```bash
pip install -r requirements.txt
```

## trec_eval commands

For getting most of the evaluation done

```cygwin
trec_eval -m all_trec <qrel_file> <run_file> 
```

You can add ``` > <file name> ``` to specify the name of the output file.

### Download [trec_eval](https://github.com/usnistgov/trec_eval) and install it using "make".

The '*main.ipynb*' contains all the code for the <b>Respective Assignments</b> in their <b>Respective folders</b>.

P.S. You can install wsl for smoother use