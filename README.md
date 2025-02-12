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
trec_eval <qrel_file> <run_file>
```

For getting specificly ndcg

```cygwin
trec_eval -m ndcg <qrel_file> <run_file>
```

### Download [trec_eval](https://github.com/usnistgov/trec_eval) and install it using "make".