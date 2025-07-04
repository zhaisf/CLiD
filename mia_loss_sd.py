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


def get_data(flags=flags, dataset_name=None):
    '''
    Loading data
    '''
    assert dataset_name != None

    # DataLoaders creation:
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        input_ids_0 = torch.stack([example["input_ids_0"] for example in examples])
        input_ids_1 = torch.stack([example["input_ids_1"] for example in examples])
        input_ids_2 = torch.stack([example["input_ids_2"] for example in examples])
        input_ids_null = torch.stack([example["input_ids_null"] for example in examples])

        return {"pixel_values": pixel_values, "input_ids": input_ids, "input_ids_0": input_ids_0,
                "input_ids_1": input_ids_1, "input_ids_2": input_ids_2, "input_ids_null": input_ids_null, }

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(flags.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(flags.resolution),  # if args.center_crop else transforms.RandomCrop(args.resolution),
            # transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

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

    def tokenize_captions_multi(examples, is_train=True):
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

        ###  the following parts are not used in this code 


        inputs_0 = tokenizer(
            [e[:int(len(e) / 3)] for e in captions], max_length=tokenizer.model_max_length, padding="max_length",
            truncation=True, return_tensors="pt"
        )

        
        inputs_1 = tokenizer(
            [e[int(len(e) / 3):int(2 * len(e) / 3)] for e in captions], max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True, return_tensors="pt"
        )

        inputs_2 = tokenizer(
            [e[int(2 * len(e) / 3):] for e in captions], max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True, return_tensors="pt"
        )

        inputs_null = tokenizer(
            ["" for e in captions], max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True, return_tensors="pt"
        )

        return inputs.input_ids, inputs_0.input_ids, inputs_1.input_ids, inputs_2.input_ids, inputs_null.input_ids

    def preprocess_train_multi(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"], examples["input_ids_0"], examples["input_ids_1"], examples["input_ids_2"], examples[
            "input_ids_null"] = tokenize_captions_multi(examples)
        return examples

    test_dataset = dataset["train"].with_transform(preprocess_train_multi)

 
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=flags.batch_size,
        num_workers=flags.dataloader_num_workers,
    )

    return test_dataloader


@torch.no_grad()
def att_measure(diffusion, sample, norm, device='cuda'):
    diffusion = diffusion.to(device).float()
    sample = sample.to(device).float()

    # if len(diffusion.shape) == 5:
    #     num_timestep = diffusion.size(0)
    #     diffusion = diffusion.permute(1, 0, 2, 3, 4).reshape(-1, num_timestep * 3, 32, 32)
    #     sample = sample.permute(1, 0, 2, 3, 4).reshape(-1, num_timestep * 3, 32, 32)
    # if metric == 'l2':

    score = ((diffusion - sample) ** norm).flatten(1).sum(dim=-1)
    
    # elif isinstance(metric, int):
    #     score = (torch.abs(diffusion - sample) ** metric).flatten(1).sum(dim=-1)
    # else:
    #     raise NotImplementedError

    return score


# Noise_global = None  # torch.randn_like(x)
import os


@torch.no_grad()
def loss_mi(model, batch, vae, text_encoder, device, x_sec_list, x_sec_recon_list):
    
    Noise_global = torch.randn(4, 64, 64)


    Noise_4mi = Noise_global.unsqueeze(0).expand(batch["pixel_values"].shape[0], -1, -1, -1)


    # global Noise_4mi
    batch["pixel_values"] = batch["pixel_values"].to(device)
    latents = vae.encode(batch["pixel_values"].to(torch.float32)).latent_dist.sample()
    latents = latents * vae.config.scaling_factor
    x = latents
    embd_cond = text_encoder(batch["input_ids"].to(device))[0]

    # exit()
    #     print('\n*** Noise init ***\n')
    # else:
    #     pass

    timesteps_sch = torch.tensor([450] * x.shape[0], ).long()
    # noisy_latents, noise_t = add_noise(x, Noise_4mi.to(device), timesteps_sch)
    noisy_latents = scheduler.add_noise(x.to(device), Noise_4mi.to(device), timesteps_sch.to(device))

    noise_pre = model(noisy_latents, flags.t_sec, embd_cond).sample

    x_sec_list.append(Noise_4mi)
    x_sec_recon_list.append(noise_pre)




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

    print('len(datas), len(labels):', len(datas), len(labels))


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
    flags.batch_size = 16
    flags.dataloader_num_workers = 0
    flags.resolution = 512
    flags.image_column = "image"
    flags.caption_column = "text"
    flags.t_sec = 450  ### timestep for MI attack
    flags.stpsnumi = 1
    flags.outdir = 'outputs'


    model_names = [
        'xxx',  # 'xxx' is a placeholder for the actual model names
    ]

    output_paths = []
    for Use_data_model_name, Attak in zip(model_names, ['loss'] * len(model_names)):
        print('\n====================================================================================')
        flags.attack = Attak
        flags.task_name = Use_data_model_name
        # Model_name = Use_data_model_name
        assert flags.attack in ['sec', 'noise', 'pia', 'loss']
        print('flags.attack, flags.task_name:', flags.attack, flags.task_name)

        flags.diff_path, flags.dataset_train_name, flags.dataset_test_name = init_paths(flags.task_name)
        init_model(flags.diff_path)

        Template_name = flags.dataset_train_name.split('/')[-1].replace('train', '').replace('test', '')
        Time = timestart

        print(str(
            flags.__dict__) + '\n' + flags.diff_path + '\n' + flags.attack + '\n' + Template_name + '-------' + '\n')

        loader_flag = 0

        for data_name in [flags.dataset_train_name, flags.dataset_test_name]:
            x_sec_list = []
            x_sec_recon_list = []

            if loader_flag == 0:
                trainOrtest = "train"
                loader_flag += 1
            else:
                trainOrtest = "test"
            loader = get_data(flags, data_name)

            print("*** trainOrtest ***  ", trainOrtest)

            for step, batch in enumerate(tqdm(loader)):
                # if step>3:break
                if flags.attack == 'loss':
                    loss_mi(unet, batch, vae, text_encoder, device, x_sec_list, x_sec_recon_list)
                # elif flags.attack == 'noise':
                #     noise_mi(model, batch, vae, text_encoder, device,  x_sec_list, x_sec_recon_list)
                # elif flags.attack == 'pia':
                #     prox_mi(unet, batch, vae, text_encoder, device, x_sec_list, x_sec_recon_list)
                else:
                    print('Error, No implement!', flags.attack)
                    exit()

            x_sec_s = torch.concat(x_sec_list)  ## 1
            x_sec_recon_s = torch.concat(x_sec_recon_list)  ## 2

            norm = 5 if flags.attack == 'prox' else 'l2'
            print('****   norm {} *****'.format(norm))
            scores = att_measure(x_sec_s, x_sec_recon_s, norm, device=device).cpu()
            scores = scores.numpy().tolist()

            print(scores[:3])

            path_temp = flags.outdir + '/Atk_{}_M_{}_DATA_{}_TRTE_{}_T_{}.txt'.format(flags.attack, flags.task_name,
                                                                                      Template_name,
                                                                                      trainOrtest, Time)
            with open(path_temp, 'w', encoding='utf8') as f:
                f.write(str(flags.__dict__) + '\t' + flags.diff_path + '\t' + '\n')
                # for i in range(len(scores)):  # i: N = samples number    #j: 5类不同cond输出
                f.write('\n'.join(['{:.5g}'.format(e) for e in scores]))

            output_paths.append(path_temp)


