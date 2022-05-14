import json
import os

print('Fixing vizwiz results...')

# Open results file and sort by question_id
results_file = 'test_predict'
with open(f'{results_file}.json', 'r') as f:
    results = sorted(json.load(f), key=lambda x: x['question_id'])

# Get image names
img_names = sorted(os.listdir('../dataset/vizwiz_data/test'))

# Merge image names and answers
results = [{'image': img_name, 'answer': answer} for img_name, answer in zip(img_names, [r['answer'] for r in results])]

# Save updated results
with open(f'{results_file}_fixed.json', 'w') as f:
    json.dump(results, f)

print('Finished fixing vizwiz results!')
