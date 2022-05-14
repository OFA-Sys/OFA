import json
import os

# Open results file and sort by question_id
results_file = 'test_predict_vizwiz.json'
with open(results_file, 'r') as f:
    results = sorted(json.load(f), key=lambda x: x['question_id'])

# Get image names
img_names = sorted(os.listdir('../dataset/vizwiz_data/test'))

# Merge image names and answers
results = [{'image': img_name, 'answer': answer} for img_name, answer in zip(img_names, [r['answer'] for r in results])]

# Save updated results
with open('test_predict_vizwiz_fixed.json', 'w') as f:
    json.dump(results, f)
