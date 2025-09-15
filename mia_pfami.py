from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import sys
import numpy as np
import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, PNDMScheduler, LMSDiscreteScheduler, DDIMScheduler
from tqdm.auto import tqdm
from datasets import load_dataset
import time
import os


class Flag(object):
    pass


timestart = time.strftime('%m%d_%H%M%S', time.localtime()).split()[0]
device = "cuda" if torch.cuda.is_available() else "cpu"
print('device ---- ', device)
flags = Flag

unet = None
tokenizer = None
text_encoder = None
scheduler = None
vae = None


def init_model(diff_path):
    global unet, tokenizer, text_encoder, scheduler, vae
    ### LOAD MODEL ########
    print('init model...')
    vae = AutoencoderKL.from_pretrained(
        diff_path, subfolder='vae', use_auth_token=True)
    print('vae loaded.')
    # vae = vae.float()

    tokenizer = CLIPTokenizer.from_pretrained(diff_path, subfolder="tokenizer", )
    text_encoder = CLIPTextModel.from_pretrained(diff_path, subfolder="text_encoder", )
    # text_encoder = text_encoder.float()
    print('tokenizer, textencoder loaded.')

    unet = UNet2DConditionModel.from_pretrained(
        diff_path,
        subfolder='unet', )  #
    print('unet loaded.')

    scheduler = DDIMScheduler.from_pretrained(diff_path, subfolder="scheduler")
    print('sch loaded.', scheduler)
    # noise_temp = scheduler.add_noise()

    vae = vae.to(device)
    vae.eval()
    text_encoder = text_encoder.to(device)
    unet.eval()
    unet = unet.to(device)
    unet.eval()
    print('all model loaded.')


def get_data(flags=flags, dataset_name=None, if_cut=False):
    '''
    Loading data
    '''
    assert dataset_name != None

    # DataLoaders creation:
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        # pixel_values_cut = torch.stack([example["pixel_values_cut"] for example in examples])
        # pixel_values_cut = pixel_values_cut.to(memory_format=torch.contiguous_format).float()

        input_ids = torch.stack([example["input_ids"] for example in examples])

        return {"pixel_values": pixel_values,
                "input_ids": input_ids, }  # "pixel_values_cut": pixel_values_cut, "input_ids": input_ids, }

    # Preprocessing the datasets.
    train_transforms_ori = transforms.Compose(
        [
            transforms.Resize(flags.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(flags.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    train_transforms_cut = transforms.Compose(
        [

            transforms.Resize(flags.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(flags.resolution),

            transforms.CenterCrop(size=int(flags.resolution * flags.strength)),
            transforms.Resize(flags.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            ###########################

            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    if if_cut:
        train_transforms = train_transforms_cut
    else:
        train_transforms = train_transforms_ori

    flags.dataset_config_name = None
    flags.cache_dir = None
    flags.train_data_dir = None
    import os

    if dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            dataset_name,
            flags.dataset_config_name,
            cache_dir=flags.cache_dir,
            data_dir=flags.train_data_dir,
        )
    else:
        data_files = {}
        if flags.train_data_dir is not None:
            data_files["train"] = os.path.join(flags.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=flags.cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    DATASET_NAME_MAPPING = {
        "lambdalabs/pokemon-blip-captions": ("image", "text"),
    }

    # Get the column names for input/target.
    dataset_columns = DATASET_NAME_MAPPING.get(dataset_name, None)
    if flags.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = flags.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if flags.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = flags.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    import random

    print('---\ncaption_column:', caption_column, '\n---')

    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        return inputs.input_ids  # inputs_0.input_ids, inputs_1.input_ids, inputs_2.input_ids, inputs_null.input_ids

    def preprocess_train_cut(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    test_dataset = dataset["train"].with_transform(preprocess_train_cut)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=flags.train_batch_size,
        num_workers=flags.dataloader_num_workers, )

    return test_dataloader


@torch.no_grad()
def att_measure(diffusion, sample, metric, device='cuda'):
    diffusion = diffusion.to(device).float()
    sample = sample.to(device).float()

    if len(diffusion.shape) == 5:
        num_timestep = diffusion.size(0)
        diffusion = diffusion.permute(1, 0, 2, 3, 4).reshape(-1, num_timestep * 3, 32, 32)
        sample = sample.permute(1, 0, 2, 3, 4).reshape(-1, num_timestep * 3, 32, 32)

    if metric == 'l2':
        score = ((diffusion - sample) ** 2).flatten(1).sum(dim=-1)
    elif isinstance(metric, int):
        ### 如果是整数，就代表p范数
        score = (torch.abs(diffusion - sample) ** metric).flatten(1).sum(dim=-1)
    else:
        raise NotImplementedError

    return score


# Noise_global = None  # torch.randn_like(x)
import os
import torch.nn.functional as F


@torch.no_grad()
def fluc_mi(model, batch, vae, text_encoder, device, ):
    batch["pixel_values"] = batch["pixel_values"].to(device)
    latents = vae.encode(batch["pixel_values"].to(torch.float32)).latent_dist.sample()
    latents = latents * vae.config.scaling_factor

    loss_list = []

    for i in range(len(latents)):
        latent = latents[i]
        # latent_cut = latents_cut[i]

        ### t_list
        ts = torch.tensor(flags.attack_ts).long()  ### flags.trials_eacht=1

        ##############
        latent_list = latent.view(-1, 4, 64, 64).expand(len(ts), 4, 64, 64)
        # latent_cut_list = latent_cut.view(-1, 4, 64, 64).expand(len(ts), 4, 64, 64)

        ############
        input_ids = batch["input_ids"][i].expand(len(ts), -1)
        emds = text_encoder(input_ids.to(device))[0]

        Noise_1 = torch.randn(4, 64, 64).unsqueeze(0).expand(latent_list.shape[0], -1, -1, -1)

        #############
        noisy_latents = scheduler.add_noise(latent_list.to(device), Noise_1.to(device), ts.to(device))
        # noisy_cut_latents = scheduler.add_noise(latent_cut_list.to(device), Noise_2.to(device), ts.to(device))

        ##############   original ##########
        noise_pred = model(noisy_latents, ts.to(device), emds).sample
        loss_avg = F.mse_loss(noise_pred.float(), Noise_1.float().to(device), reduction="mean")

        loss_list.append(float(loss_avg.detach().cpu()))

    return loss_list


def get_asr_acu_tpr1():
    print('\n-------- cal asr auc ------------\n')
    # from cal_and_draw_th import get_ori_data
    # from cal_and_draw_th import deal_data_first
    from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

    datas = []
    labels = []
    with open(flags.outdir + output_paths[0], 'r') as ftrain:
        lines = ftrain.readlines()[1:]
        float_list = [float(line.split('\t')[0].strip()) for line in lines]

        label_list = [0] * len(float_list)
        datas.extend(float_list)
        labels.extend(label_list)

    with open(flags.outdir + output_paths[1], 'r') as ftest:
        lines = ftest.readlines()[1:]
        float_list = [float(line.split('\t')[0].strip()) for line in lines]
        label_list = [1] * len(float_list)
        datas.extend(float_list)
        labels.extend(label_list)

    # print('len(datas), len(labels):', len(datas), len(labels))

    best_threshold = None
    best_accuracy = 0.0

    min_threshold = min(datas)
    max_threshold = max(datas)
    threshold_step = (max_threshold - min_threshold) / 2000

    for threshold in list(np.arange(min_threshold, max_threshold, threshold_step)):
        predicted_values = [1 if value > threshold else 0 for value in datas]

        accuracy = accuracy_score(labels, predicted_values)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    print('\n**************:', output_paths)
    print('name', flags.attack, '|   best_accuracy, best_threshold, th% :', best_accuracy, best_threshold,
          (best_threshold - min_threshold) / (max_threshold - min_threshold))

    auc = roc_auc_score(labels, [(e - min_threshold) / (max_threshold - min_threshold) for e in datas])
    print('name', flags.attack, "|    AUC Score:", auc)

    fpr, tpr, _ = roc_curve(labels, [(e - min_threshold) / (max_threshold - min_threshold) for e in datas])
    idx_1_percent_fpr = next(i for i, fpr_value in enumerate(fpr) if fpr_value >= 0.01)
    tpr_at_1_percent_fpr = tpr[idx_1_percent_fpr]

    print('name', flags.attack, "|   tpr_at_1_percent_fpr:", tpr_at_1_percent_fpr)

    return best_accuracy, auc, tpr_at_1_percent_fpr


####
def init_paths(task_name):
    # global flags
    diff_path = {

        "xxx": r"xxxxxxxxxxxxxxxxxxxxx",

    }[task_name]

    train_data_dict = {
        "xxx": r"xxxxxxxxxxxxxxxxxxxxx",

    }


    dataset_train_name = train_data_dict[Use_data_model_name]

    test_data_dict = {
        "xxx": r"xxxxxxxxxxxxxxxxxxxxx",
    }

    dataset_test_name = test_data_dict[Use_data_model_name]
  
    return diff_path, dataset_train_name, dataset_test_name


torch.no_grad()
if __name__ == '__main__':
    flags.T = 1000
    flags.train_batch_size = 16
    flags.dataloader_num_workers = 0
    flags.resolution = 512
    flags.image_column = "image"
    flags.caption_column = "text"
    flags.t_sec = 100
    flags.timestep = 10
    flags.stpsnumi = 1
    flags.outdir = 'outputs'
    ######### fluc (PFAMI) setting ###########
    flags.strength = 0.825
    flags.attack_ts = [1, 51, 101, 151, 201, 251, 301, 351, 401, 451]
    ####################


    model_names = [
        'xxx',
    ]

    output_paths = []
    for Use_data_model_name, Attak in zip(model_names, ['fluc'] * len(model_names)):
        print('\n====================================================================================')
        flags.attack = Attak
        flags.task_name = Use_data_model_name
        assert flags.attack in ['fluc']  # 'sec', 'noise', 'pia', 'loss'
        print('flags.attack, flags.task_name:', flags.attack, flags.task_name)

        flags.diff_path, flags.dataset_train_name, flags.dataset_test_name = init_paths(flags.task_name)
        init_model(flags.diff_path)

        Template_name = flags.dataset_train_name.split('/')[-1].replace('train', '').replace('test', '')
        Time = timestart

        print(str(
            flags.__dict__) + '\n' + flags.diff_path + '\n' + flags.attack + '\n' + Template_name + '-------' + '\n')

        loader_flag = 0

        for data_name in [flags.dataset_train_name, flags.dataset_test_name]:
            loss_ori = []
            loss_cut = []

            if loader_flag == 0:
                trainOrtest = "train"
                loader_flag += 1
            else:
                trainOrtest = "test"

            if_cut = False
            loader = get_data(flags, data_name, if_cut=if_cut)
            print(f"if_cut {if_cut}")
            print("*** trainOrtest ***  ", trainOrtest)
            for step, batch in enumerate(tqdm(loader)):
                if flags.attack == 'fluc':
                    loss_batch = fluc_mi(unet, batch, vae, text_encoder, device, )
                    loss_ori.extend(loss_batch)
                    # if step < 3: print(step, f'[ {if_cut} ]  loss_batch:   ', loss_batch)
                else:
                    print('Error, No implement!', flags.attack)
                    exit()

            if_cut = True
            loader = get_data(flags, data_name, if_cut=if_cut)
            print(f"if_cut {if_cut}")
            print("*** trainOrtest ***  ", trainOrtest)
            for step, batch in enumerate(tqdm(loader)):
                if flags.attack == 'fluc':
                    loss_batch = fluc_mi(unet, batch, vae, text_encoder, device, )
                    loss_cut.extend(loss_batch)
                    # if step < 3: print(step, f'[ {if_cut} ]  loss_batch:   ', loss_batch)
                else:
                    print('Error, No implement!', flags.attack)
                    exit()

            loss_dif = [cut - ori for cut, ori in zip(loss_cut, loss_ori)]

            path_temp = flags.outdir + '/Atk_{}_M_{}_DATA_{}_TRTE_{}_T_{}.txt'.format(flags.attack, flags.task_name,
                                                                                      Template_name,
                                                                                      trainOrtest, Time)
            with open(path_temp, 'w', encoding='utf8') as f:
                f.write(str(flags.__dict__) + '\t' + flags.diff_path + '\t' + '\n')

                f.write('\n'.join(['{:.5g}'.format(e) for e in loss_dif]))

            output_paths.append(path_temp)
