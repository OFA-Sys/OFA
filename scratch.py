import os
from glob import glob

# JW: Set root folder for Telegram Bot
# Expected structure: root/question_id/image.jpg and question.txt file
data = '../../tgbot/data/'
assert os.path.exists(data), 'Data folder does not exist!'

# Add loop to queue multiple questions
print("Waiting for question, image pair...")
while True:
    queue = []
    for folder in glob(f'{data}/*/*'):
        if not os.path.exists(f'{folder}/answer.txt'):
            queue.append(folder)
    print(f'Found {len(queue)} questions')
