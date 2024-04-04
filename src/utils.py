import multiprocessing
import PIL
import torch
import json
import torchvision.transforms.functional as FT
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize, Resize
from torchvision.transforms import InterpolationMode
from dataset import FashionIQDataset
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def extract_index_features(dataset, clip_model):
    """
    Extract FashionIQ or CIRR or CIRCOindex features
    :param dataset: FashionIQ or CIRR or CIRCO dataset in 'classic' mode
    :param clip_model: CLIP model
    :return: a tensor of features and a list of images
    """
    feature_dim = clip_model.visual.output_dim
    classic_val_loader = DataLoader(dataset=dataset, batch_size=32, num_workers=multiprocessing.cpu_count(),
                                    pin_memory=True, collate_fn=collate_fn)
    index_features = torch.empty((0, feature_dim)).to(device, non_blocking=True)
    index_names = []
    if isinstance(dataset, FashionIQDataset):
        print(f"extracting fashionIQ {dataset.dress_types} - {dataset.split} index features")
    for batch in tqdm(classic_val_loader):
        images = batch.get('image')
        names = batch.get('image_name')
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            batch_features = clip_model.encode_image(images)
            index_features = torch.vstack((index_features, batch_features))
            index_names.extend(names)
    return index_features, index_names


def get_img_patch_feats(img, clip_model):
    """
        Get the output of second-to-last layer of CLIP visual encoder
    """

    with torch.no_grad():
        x = clip_model.visual.conv1(img)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                                   dtype=x.dtype, device=x.device), x],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + clip_model.visual.positional_embedding.to(x.dtype)
        x = clip_model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND

        layer = clip_model.visual.transformer.layers - 1

        for i in range(layer):
            x = clip_model.visual.transformer.resblocks[i](x)
        clip_last_layer = clip_model.visual.transformer.resblocks[layer]

        x = x + clip_last_layer.attn(clip_last_layer.ln_1(x), clip_last_layer.ln_1(x), clip_last_layer.ln_1(x),
                                     need_weights=False, attn_mask=None)[0]
        x = x + clip_last_layer.mlp(clip_last_layer.ln_2(x))

        x = x.permute(1, 0, 2)  # LND -> NLD  [batchsize, 257, 1024]

    return x


def contrastive_loss(v1, v2, temperature: float):
    v1 = F.normalize(v1, dim=1)
    v2 = F.normalize(v2, dim=1)

    numerator = torch.exp(torch.diag(torch.inner(v1, v2)) / temperature)
    numerator = torch.cat((numerator, numerator), 0)
    joint_vector = torch.cat((v1, v2), 0)
    pairs_product = torch.exp(torch.mm(joint_vector, joint_vector.t()) / temperature)
    denominator = torch.sum(pairs_product - pairs_product * torch.eye(joint_vector.shape[0]).to(v2.device), 0)

    loss = -torch.mean(torch.log(numerator / denominator))

    return loss


def token_replace(local_attr_num, text, img_subj_token, img_attr_tokens, clip_model, map_type=0):
    """
    Replace symbols * with subject-oriented pseudo-word token or attribute-oriented pseudo-word token:
    map_type=0,replace both
    map_type=1,replace * with subject-oriented pseudo-word token
    map_type=2,replace [*,...,*] with several attribute-oriented pseudo-word tokens
    """

    img_subj_token = img_subj_token.view(img_subj_token.shape[0], 1, -1)

    split_ind = clip.tokenize(["*"])[0][1]
    end_id = clip_model.vocab_size - 1
    x = clip_model.token_embedding(text).type(clip_model.dtype)  # [batch_size, n_ctx, d_model]
    collect_ind = text == end_id
    collect_ind = collect_ind.nonzero()[:, 1]
    bs = x.shape[0]

    if map_type == 0 or map_type == 1:
        ind_insert = text[0] == split_ind
        ind_insert = ind_insert.nonzero()[0]
        if map_type == 0:
            for i in range(bs):
                # get selected attribute-oriented pseudo-word token
                selected_local_tokens = img_attr_tokens[i, :local_attr_num[i], :]
                x[i] = torch.cat(
                    [x[i][:ind_insert], img_subj_token[i], x[i][ind_insert + 1],
                     selected_local_tokens,
                     x[i][ind_insert + local_attr_num[i] + 2:]], dim=0)
        elif map_type == 1:
            # replace * with subject-oriented pseudo-word token
            x = torch.cat(
                [x[:, :ind_insert], img_subj_token, x[:, ind_insert + 1:]], dim=1)
    else:
        for i in range(bs):
            # replace [*,...,*] with several attribute-oriented pseudo-word tokens
            ind_insert = text[i] == split_ind
            ind_insert = ind_insert.nonzero()[0]
            selected_local_tokens = img_attr_tokens[i, :local_attr_num[i], :]
            x[i] = torch.cat([x[i][:ind_insert], selected_local_tokens, x[i][ind_insert + local_attr_num[i]:]],
                             dim=0)

    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)
    x = x[torch.arange(x.size(0)), collect_ind] @ clip_model.text_projection
    return x


def collate_fn(batch: list):
    """
    Discard None images in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class TargetPad:
    """
    If an image aspect ratio is above a target ratio, pad the image to match such target ratio.
    For more details see Baldrati et al. 'Effective conditioned and composed image retrieval combining clip-based features.' Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2022).
    """

    def __init__(self, target_ratio: float, size: int):
        """
        :param target_ratio: target ratio
        :param size: preprocessing output dimension
        """
        self.size = size
        self.target_ratio = target_ratio

    def __call__(self, image: PIL.Image.Image) -> PIL.Image.Image:
        w, h = image.size
        actual_ratio = max(w, h) / min(w, h)
        if actual_ratio < self.target_ratio:  # check if the ratio is above or below the target ratio
            return image
        scaled_max_wh = max(w, h) / self.target_ratio  # rescale the pad to match the target ratio
        hp = max(int((scaled_max_wh - w) / 2), 0)
        vp = max(int((scaled_max_wh - h) / 2), 0)
        padding = [hp, vp, hp, vp]
        return FT.pad(image, padding, 0, 'constant')


def targetpad_transform(target_ratio: float, dim: int) -> torch.Tensor:
    """
    CLIP-like preprocessing transform computed after using TargetPad pad
    :param target_ratio: target ratio for TargetPad
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        TargetPad(target_ratio, dim),
        Resize(dim, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class RunningAverage:
    """A simple class that maintains the running average of a quantity"""

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)
