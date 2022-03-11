import datasets
import sys
import json

rouge = datasets.load_metric('../../utils/rouge.py')

if __name__ == "__main__":
    f = sys.argv[1]
    results = json.load(open(f))
    predictions = [result['hyp'] for result in results]
    references = [result['ref'] for result in results]
    results = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    print("Rouge1: ", results["rouge1"].mid.fmeasure)
    print("Rouge2: ", results["rouge2"].mid.fmeasure)
    print("RougeL: ", results["rougeL"].mid.fmeasure)
