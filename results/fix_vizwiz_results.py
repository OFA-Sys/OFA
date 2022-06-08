import json
import os

print('Fixing vizwiz results...')

# Open results file and sort by question_id
results_file = 'test_predict.json'
assert os.path.exists(results_file), f'Results file not found: {results_file}'
with open(results_file, 'r') as f:
    results = sorted(json.load(f), key=lambda x: x['question_id'])

# Get image names
img_names = [f'VizWiz_test_{i:08d}.jpg' for i in range(8000)]

# Merge image names and answers
results = [{'image': img_name, 'answer': answer} for img_name, answer in zip(img_names, [r['answer'] for r in results])]

# Save updated results
with open(f'{results_file}_fixed.json', 'w') as f:
    json.dump(results, f)

print('Finished fixing vizwiz results!')
