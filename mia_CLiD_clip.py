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
import torch.nn.functional as F

timestart = time.strftime('%m%d_%H%M%S', time.localtime()).split()[0]
device = "cuda" if torch.cuda.is_available() else "cpu"
print('device:', device)

class Flag(object):
    pass

'''
coco_ori; coco_split1; 
coco_asdoc_ori; coco_asdoc_split1
pokemon_ori; pokemon_split1
pokemon_ori_aug; pokemon_split1_aug
coco_asdoc_ori_5001; coco_asdoc_split1_5001
coco_asdoc_ori_25001; coco_asdoc_split1_25001
coco_asdoc_ori_50001; coco_asdoc_split1_50001
coco_asdoc_ori_delete3 ; coco_asdoc_split1_delete3
coco_asdoc_ori_delete1 ; coco_asdoc_split1_delete1
pretrain_mia_1; pretrain_mia_2; pretrain_sd_v15
pretrain_sd_v15_wacv;
coco_asdoc_ori_delete5 ; coco_asdoc_split1_delete5
'''

Use_data_model_name = "flick_real_split1"

flags = Flag
diff_path = {
   "xx":"xx"
}[Use_data_model_name]


flags.diff_path = diff_path

train_data_dict = {
    "xx": "xx"

}

if 'coco' in Use_data_model_name:
    if 'ori' in Use_data_model_name:
        flags.dataset_train_name = train_data_dict['coco_ori']
    elif 'split1' in Use_data_model_name:
        flags.dataset_train_name = train_data_dict['coco_split1']

else:
    flags.dataset_train_name = train_data_dict[Use_data_model_name]

test_data_dict = {
    "xx": "xx"
}

if 'coco' in Use_data_model_name:
    if 'ori' in Use_data_model_name:
        flags.dataset_test_name = test_data_dict['coco_ori']
    elif 'split1' in Use_data_model_name:
        flags.dataset_test_name = test_data_dict['coco_split1']

else:
    flags.dataset_test_name = test_data_dict[Use_data_model_name]

### LOAD MODEL
vae = AutoencoderKL.from_pretrained(
    diff_path, subfolder='vae', use_auth_token=True)
vae = vae.to(device)
print('vae loaded.')
# vae = vae.float()

tokenizer = CLIPTokenizer.from_pretrained(diff_path, subfolder="tokenizer", )
text_encoder = CLIPTextModel.from_pretrained(diff_path, subfolder="text_encoder", )
print('tokenizer, textencoder loaded.')

unet = UNet2DConditionModel.from_pretrained(
    diff_path,
    subfolder='unet', )  
print('unet loaded.')

scheduler = DDIMScheduler.from_pretrained(diff_path, subfolder="scheduler")
print('sch loaded.', scheduler)

vae = vae.to(device)
text_encoder = text_encoder.to(device)
unet = unet.to(device)
# unet(noisy_latents, timesteps, encoder_hidden_states).sample
vae.eval()
unet.eval()

flags.attack = 'clid_clip'
assert flags.attack in ['clid_clip', 'sec','pia', 'clid_impt','pfami']

# flags.ifcond = True
# flags.stps = [1]  # , 5, 10, 70, 100]
flags.T = 1000
flags.even_num = 10  ##
flags.max_n_samples = 3  ##
flags.max_clid_samples = 3  ##

flags.trials_eacht = 1

flags.train_batch_size = 8
flags.dataloader_num_workers = 0
flags.resolution = 512
flags.image_column = "image"
flags.caption_column = "text"
flags.t_sec = 100
flags.timestep = 10
flags.stpsnumi = 1
flags.outdir = 'outputs'
model_name = Use_data_model_name  # diff_path.split('/')[-1]



Template_name = flags.dataset_train_name.split('/')[-1].replace('train', '').replace('test', '')
Time = timestart

print(str(flags.__dict__) + '\n' + diff_path + '\n' + flags.attack + '\n' + Template_name + '-------' + '\n')


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
        input_ids_1 = torch.stack([example["input_ids_1"] for example in examples])
        input_ids_2 = torch.stack([example["input_ids_2"] for example in examples])
        input_ids_3 = torch.stack([example["input_ids_3"] for example in examples])
        input_ids_null = torch.stack([example["input_ids_null"] for example in examples])

        return {"pixel_values": pixel_values, "input_ids": input_ids, "input_ids_1": input_ids_1,
                "input_ids_2": input_ids_2, "input_ids_3": input_ids_3, "input_ids_null": input_ids_null, }

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

    print('---caption_column:', caption_column, '---')

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
        examples["input_ids"], examples["input_ids_1"], examples["input_ids_2"], examples["input_ids_3"], examples[
            "input_ids_null"] = tokenize_captions_multi(examples)
        return examples

    test_dataset = dataset["train"].with_transform(preprocess_train_multi)

    subset_indices = range(2500)
    from torch.utils.data import Subset
    subset_dataset = Subset(test_dataset, subset_indices)

    test_dataloader = torch.utils.data.DataLoader(
        subset_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=flags.train_batch_size,
        num_workers=flags.dataloader_num_workers,
    )

    return test_dataloader




@torch.no_grad()
def mi_mtcl_denoise(model, batch, vae, text_encoder, device, ):  # x_sec_list_s, x_sec_recon_list_s):
    global Noise
    global Noise_usedidx

    batch["pixel_values"] = batch["pixel_values"].to(device)
    latents = vae.encode(batch["pixel_values"].to(torch.float32)).latent_dist.sample()
    # print('latents.shape:', latents.shape)
    # print('vae.config.scaling_factor:', vae.config.scaling_factor)  ## vae.config.scaling_factor = 0.18215
    latents = latents * vae.config.scaling_factor
    # print('latents.shape 2:', latents.shape)
    # x = latents
    # embd_cond = text_encoder(batch["input_ids"].to(device))[0]

    T = flags.T
    even_num = flags.even_num
    max_n_samples = flags.max_n_samples
    # print('flags.max_n_samples: ', flags.max_n_samples)
    max_clid_samples = flags.max_clid_samples

    start = T // 2 - (even_num * max_n_samples // 2)
    t_to_eval = list(range(start, T, even_num))[:max_n_samples]
    # print('t_to_eval:', t_to_eval, )
    start_idx = len(t_to_eval) // 2 - max_clid_samples // 2
    t_to_eval = np.array(t_to_eval)
    t_clid_to_eval = list(t_to_eval[[start_idx + i for i in list(range(max_clid_samples))]])

    
    # print('t_clid_to_eval:', t_clid_to_eval)

    noise = None

    batch_loss = {"cond0": [], "cond1_dif": [], "cond2_dif": [], 'cond3_dif': [], "condNull_dif": []}

    for latent, input_ids, input_ids_1, input_ids_2, input_ids_3, input_ids_null in zip(latents, batch["input_ids"],
                                                                                        batch['input_ids_1'],
                                                                                        batch['input_ids_2'],
                                                                                        batch['input_ids_3'],
                                                                                        batch['input_ids_null']):
        assert latent.shape[-3:] == (4, 64, 64)  ###te

        ts = torch.tensor(np.concatenate([t_to_eval] * flags.trials_eacht)).long()  ### flags.trials_eacht=1
        ts_other = torch.tensor(np.concatenate([t_clid_to_eval] * flags.trials_eacht)).long()  ### flags.trials_eacht=1

        pixel_mtcl = latent.view(-1, 4, 64, 64).expand(len(t_to_eval), 4, 64, 64)
        noise = Noise[Noise_usedidx: Noise_usedidx + len(t_to_eval)]
        noise_other = noise[[start_idx + i for i in list(range(max_clid_samples))]]

        x_mtcl = scheduler.add_noise(pixel_mtcl.to(device), noise.to(device), ts.to(device))

        input_id_mtcl = input_ids.expand(len(t_to_eval), -1)
        emd_mtcl = text_encoder(input_id_mtcl.to(device))[0]

        noise_pred_emd_ori = model(x_mtcl, ts.to(device), emd_mtcl).sample
        loss_emd_ori = F.mse_loss(noise_pred_emd_ori.float(), noise.float().to(device), reduction="mean")

        batch_loss["cond0"].append(float(loss_emd_ori.detach().cpu()))

        for input_ids_other, dict_name in zip([input_ids_1, input_ids_2, input_ids_3, input_ids_null],
                                              ['cond1_dif', 'cond2_dif', 'cond3_dif', 'condNull_dif']):
            pixel_mtcl_other = latent.view(-1, 4, 64, 64).expand(len(t_clid_to_eval), 4, 64, 64)
            x_mtcl_other = scheduler.add_noise(pixel_mtcl_other.to(device), noise_other.to(device), ts_other.to(device))
            input_id_mtcl_other = input_ids_other.expand(len(t_clid_to_eval), -1)
            emd_mtcl_other = text_encoder(input_id_mtcl_other.to(device))[0]

            noise_pred_emd_other = model(x_mtcl_other, ts_other.to(device), emd_mtcl_other).sample
            loss_emd_other = F.mse_loss(noise_pred_emd_other.float(), noise_other.float().to(device), reduction="mean")

            batch_loss[dict_name].append(float(loss_emd_other.detach().cpu()) - float(loss_emd_ori.detach().cpu()))

        Noise_usedidx += len(t_to_eval)

    return batch_loss



torch.no_grad()


for Max_n_samples in [3, ]:  # 5, 7, 9]:

    flags.max_n_samples = Max_n_samples
    flags.max_clid_samples = Max_n_samples

    T = flags.T
    even_num = flags.even_num
    max_n_samples = flags.max_n_samples
    start = T // 2 - (even_num * max_n_samples // 2)
    t_to_eval = list(range(start, T, even_num))[:max_n_samples]
    print('\n ***************   Max_n_samples, t_to_eval', Max_n_samples, t_to_eval,
          '\nflags.max_n_samples, flags.max_clid_samples', flags.max_n_samples, flags.max_clid_samples)

    Noise = torch.randn(5000 * 40, 4, 64, 64)

    Noise_usedidx = 0
    print('Noise.shape:', Noise.shape)

    loader_flag = 0
    output_paths = []
    for data_name in [flags.dataset_train_name, flags.dataset_test_name]:
        # x_sec_list_s = [[], [], [], [], []]
        # x_sec_recon_list_s = [[], [], [], [], []]

        if loader_flag == 0:
            trainOrtest = "train"
            loader_flag += 1
        else:
            trainOrtest = "test"

        loader = get_data(flags, data_name)
        assert flags.max_n_samples * len(loader) * 2 < Noise.shape[0]

        print("*** trainOrtest ***  ", trainOrtest)
        dataset_loss_dict = {"cond0": [], "cond1_dif": [], "cond2_dif": [], 'cond3_dif': [], "condNull_dif": []}
        for step, batch in enumerate(tqdm(loader)):
            # if step>3:break
            model = unet
            if flags.attack == 'mydenoise':
                batch_loss = mi_mtcl_denoise(model, batch, vae, text_encoder, device)
                # print('batch_loss:', batch_loss)
                for key, value in batch_loss.items():
                    dataset_loss_dict[key].extend(value)
                # print('dataset_loss_dict:', dataset_loss_dict)

            else:
                print('Error, No implement!', flags.attack)
                exit()

        # name = Template_name.replace('[1]', '1')
        path_temp = '/Atk_{}_M_{}_DATA_{}_TRTE_{}_MAXsmp_{}_T_{}.txt'.format(flags.attack, model_name, Template_name,
                                                                             trainOrtest, flags.max_n_samples, Time)
        output_paths.append(path_temp)
        with open(flags.outdir + path_temp, 'w', encoding='utf8') as f:
            f.write(str(flags.__dict__) + '\t' + diff_path + '\t' + '\n')
            # print('\n\n---------------\n-------dataset_loss_dict', dataset_loss_dict)
            lines = ['\t'.join(map(lambda x: "{:.5g}".format(x), values)) for values in
                     zip(*dataset_loss_dict.values())]
            f.write('\n'.join(lines))

            

        print('save in', flags.outdir + path_temp)


    
