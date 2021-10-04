# Spec
## rules 
- ClassName : upper camel case
- functionName : lower camel case
- CONST_VAR_NAME : UPPER CASE WITH _
## Paths

- folders : lower camel case 資料夾小駝峰
- files : lower camel case 檔案小駝峰
```
./
    datasets/
        dataset1/..
        dataset2/..
        testset/..
    configs/
        xxxtraingConfig.yaml   (training task Config 1st priority)
        override.yaml  (override default for this PC 2nd)
        default.yaml   (default configs for all PC 3rd)
    models/
        models.py
    scripts/
        utils.py
        efficientTransformerSR.py
    checkpoints/
        ...
    main.py    
```