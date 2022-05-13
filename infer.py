import glob
import re
import torch
from matplotlib import pyplot as plt
import numpy as np
from fairseq import checkpoint_utils
from fairseq import distributed_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from PIL import Image
from tasks.mm_tasks.vqa_gen import VqaGenTask
from utils.zero_shot_utils import zero_shot_step

# JW: imports for TTS and Speech Recognition
import pyttsx3
import speech_recognition as sr

# JW: imports for random selection of VizWiz images
import random


def main():
    def tts(text):
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()

    # JW: Let user know that the program is starting up
    tts('I am starting up. Please give me a second.')

    # Register VQA task
    tasks.register_task('vqa_gen', VqaGenTask)

    # turn on cuda if GPU is available
    use_cuda = torch.cuda.is_available()
    # use fp16 only when GPU is available
    use_fp16 = False

    # specify some options for evaluation
    parser = options.get_generation_parser()
    input_args = ["", "--task=vqa_gen", "--beam=100", "--unnormalized", "--path=checkpoints/ofa_large.pt",
                  "--bpe-dir=utils/BPE"]
    args = options.parse_args_and_arch(parser, input_args)
    cfg = convert_namespace_to_omegaconf(args)

    # Load pretrained ckpt & config
    task = tasks.setup_task(cfg.task)
    models, cfg = checkpoint_utils.load_model_ensemble(utils.split_paths(cfg.common_eval.path), task=task)

    # Move models to GPU
    for model in models:
        model.eval()
        if use_fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Initialize generator
    generator = task.build_generator(models, cfg.generation)

    # Image transform
    from torchvision import transforms
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize((cfg.task.patch_image_size, cfg.task.patch_image_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Text preprocess
    bos_item = torch.LongTensor([task.src_dict.bos()])
    eos_item = torch.LongTensor([task.src_dict.eos()])
    pad_idx = task.src_dict.pad()

    # Normalize the question
    def pre_question(question, max_ques_words):
        question = question.lower().lstrip(",.!?*#:;~").replace('-', ' ').replace('/', ' ')
        question = re.sub(r"\s{2,}", ' ', question, )
        question = question.rstrip('\n')
        question = question.strip(' ')
        # truncate question
        question_words = question.split(' ')
        if len(question_words) > max_ques_words:
            question = ' '.join(question_words[:max_ques_words])
        return question

    def encode_text(text, length=None, append_bos=False, append_eos=False):
        s = task.tgt_dict.encode_line(line=task.bpe.encode(text), add_if_not_exist=False, append_eos=False).long()
        if length is not None:
            s = s[:length]
        if append_bos:
            s = torch.cat([bos_item, s])
        if append_eos:
            s = torch.cat([s, eos_item])
        return s

    # Construct input for open-domain VQA task
    def construct_sample(image: Image, question: str):
        patch_image = patch_resize_transform(image).unsqueeze(0)
        patch_mask = torch.tensor([True])

        question = pre_question(question, task.cfg.max_src_length)
        question = question + '?' if not question.endswith('?') else question
        src_text = encode_text(f' {question}', append_bos=True, append_eos=True).unsqueeze(0)

        src_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in src_text])
        ref_dict = np.array([{'yes': 1.0}])  # just placeholder
        sample = {
            "id": np.array(['42']),
            "net_input": {
                "src_tokens": src_text,
                "src_lengths": src_length,
                "patch_images": patch_image,
                "patch_masks": patch_mask,
            },
            "ref_dict": ref_dict,
        }
        return sample

    # Function to turn FP32 to FP16
    def apply_half(t):
        if t.dtype is torch.float32:
            return t.to(dtype=torch.half)
        return t

    # Text preprocess
    bos_item = torch.LongTensor([task.src_dict.bos()])
    eos_item = torch.LongTensor([task.src_dict.eos()])
    pad_idx = task.src_dict.pad()

    # JW: Open any image file
    random.seed(420)
    path = random.choice(glob.glob('dataset/vizwiz_data/val/*.jpg'))
    print(path.split('/')[-1][:-4])
    image = Image.open(path)

    # JW: Show image via PIL
    plt.imshow(image)
    plt.show()

    # JW Initialize speech recognizer
    r = sr.Recognizer()
    m = sr.Microphone(0)

    # JW: Add loop to ask multiple questions
    while True:
        try:
            with m as source:
                r.adjust_for_ambient_noise(source)
                tts('Please ask a question.')
                audio = r.listen(source)
                question = r.recognize_google(audio).lower()
        except sr.RequestError as e:
            tts('Sorry, I could not understand your question.')
            continue
        except sr.UnknownValueError:
            return tts('No question detected, exiting...')

        # Construct input sample & preprocess for GPU if cuda available
        sample = construct_sample(image, question)
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample

        # Run eval step for open-domain VQA
        with torch.no_grad():
            result, scores = zero_shot_step(task, generator, models, sample)

        # Answer question
        tts(f'Your question was: {question}')
        tts(f'OFA\'s Answer is: {result[0]["answer"]}\n')

        # Ask if user wants to ask another question
        with m as source:
            r.adjust_for_ambient_noise(source)
            tts('Would you like to ask another question?')
            while True:
                audio = r.listen(source)
                try:
                    response = r.recognize_google(audio).lower()
                    if 'yes' in response:
                        break
                    elif 'no' in response:
                        return tts('Exiting...')
                    else:
                        tts('Sorry, I could not understand your response.')
                except sr.UnknownValueError:
                    tts('Sorry, I could not understand your response.')


if __name__ == "__main__":
    main()
