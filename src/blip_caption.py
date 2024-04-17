import json
import os
import clip
import PIL.Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip import blip_decoder
import torchvision.transforms.functional as FT
from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize, Resize
from torchvision.transforms import InterpolationMode
import spacy
import numpy as np
from utils import targetpad_transform
from argparse import ArgumentParser

device = "cuda" if torch.cuda.is_available() else "cpu"


def split_caption(img_caption, nlp):
    # Split the input caption into individual words
    words = img_caption.split()
    # Get the last word in the caption
    last_word = words[-1]
    # Calculate the length of the repeat portion at the end of the caption
    repeat_length = len(words)

    # Remove the repeated words at the end of the caption
    for u in range(len(words) - 2, -1, -1):
        if words[u] == last_word:
            repeat_length -= 1
        else:
            break

    # Join the words without the repeat portion
    result = ' '.join(words[:repeat_length])

    # If the resulting caption has less than 4 words, return it as both parts
    if len(result.split()) < 4:
        return result, result

    # Analyze the resulting caption using the specified NLP model
    doc = nlp(result)

    # Find the position of the first noun token in the caption
    first_noun_position = None
    for j, token in enumerate(doc):
        if token.pos_ == "NOUN":
            first_noun_position = j + 1
            break

    # Split the caption into two parts based on the position of the first noun
    first_part = result[0:2]
    second_part = result[2:]

    # If a noun is found, update the first and second parts accordingly
    if first_noun_position is not None:
        first_part = " ".join(token.text for token in doc[:first_noun_position])
        second_part = " ".join(token.text for token in doc[first_noun_position:])

        # If the resulting caption has a length smaller or equal to first_noun_position + 2,
        # set the second part to be the same as the first part
        if len(result) <= first_noun_position + 2:
            second_part = first_part

    return first_part, second_part


def get_blip_model(model_path, image_size):
    model = blip_decoder(pretrained=model_path, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)

    preprocess = targetpad_transform(1.25, image_size)
    # preprocess = transforms.Compose([
    #     transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    # ])

    return model, preprocess


def get_imageId_list():
    imageId = []
    t = 100000001
    for i in range(100000):
        imgId = t + i
        imgId = str(imgId)[1:]
        imageId.append(imgId)
    return imageId


def caption_generation(model_path, image_path, image_size):
    nlp = spacy.load("en_core_web_sm")
    model, preprocess = get_blip_model(model_path, image_size)
    imageId = get_imageId_list()
    img_caption_blip = []

    for img_id in imageId:
        img_path = os.path.join(image_path, f"test/ILSVRC2012_test_{img_id}.JPEG")
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f).convert('RGB')
            image = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            captions = model.generate(image, sample=False, num_beams=3, max_length=25, min_length=10)
            img_exp = dict()
            subject, attribute = split_caption(captions[0], nlp)
            if subject != attribute:
                img_exp["img_id"] = img_id
                img_exp["subject"] = subject
                img_exp["attribute"] = attribute
                img_caption_blip.append(img_exp)

    return img_caption_blip


def main():
    parser = ArgumentParser()
    # test dataset path
    parser.add_argument('--blip_model_url', type=str, default="./model_base_capfilt_large.pth")
    parser.add_argument('--img_path', type=str, default="/data/ImageNet/")
    parser.add_argument('--preprocess_type', type=str, default="targetpad")
    parser.add_argument('--image_size', type=int, default=384)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--clip_model_name', type=str, default='ViT-L/14')

    # fixed the seed
    args = parser.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False

    img_caption_blip = caption_generation(args.model_url, args.img_path, args.image_size)

    with open("blip_pairs.json", "w") as json_file:
        json.dump(img_caption_blip, json_file)


if __name__ == '__main__':
    main()
