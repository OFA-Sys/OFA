import re
import torch
import numpy as np
from fairseq import checkpoint_utils
from fairseq import distributed_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from PIL import Image
from tasks.mm_tasks.vqa_gen import VqaGenTask
from utils.zero_shot_utils import zero_shot_step

# JW: Imports for Bot interaction
import glob
import os
import warnings
from time import sleep
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def main():
    # Register VQA task
    tasks.register_task('vqa_gen', VqaGenTask)

    # turn on cuda if GPU is available
    use_cuda = torch.cuda.is_available()
    # use fp16 only when GPU is available
    use_fp16 = False

    # specify some options for evaluation
    parser = options.get_generation_parser()
    input_args = ["", "--task=vqa_gen", "--beam=100", "--unnormalized", "--path=checkpoints/vizwiz_base_best.pt",
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

    # JW: Set root folder for Telegram Bot
    # Expected structure: root/question_id/image.jpg and question.txt file
    data = '../../tgbot/data/'
    assert os.path.exists(data), 'Data folder does not exist!'

    # Add loop to queue multiple questions
    print("Waiting for question, image pair...")
    while True:
        if os.path.exists(f'{data}/answer.txt'):
            continue

        image_path, question_path = f'{data}/image.png', f'{data}/question.txt'
        if os.path.exists(image_path) and os.path.exists(question_path):
            print(f'Processing question, image pair')

            image = Image.open(image_path)
            question = open(question_path).read()

            # Construct input sample & preprocess for GPU if cuda available
            sample = construct_sample(image, question)
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample

            # Run eval step for open-domain VQA
            with torch.no_grad():
                result, scores = zero_shot_step(task, generator, models, sample)

            # Save answer as TXT file in folder
            with open(f'{data}/answer.txt', 'w') as f:
                f.write(result[0]['answer'])

            # Delete image and question
            try:
                os.remove(image_path)
                os.remove(question_path)
            except OSError:
                return f'Either the image or question file is open in another process, close it and try again!'

            print(f'Answer saved to {data}answer.txt')
            print("Waiting for question, image pair...")
        else:
            sleep(1)


if __name__ == "__main__":
    main()
