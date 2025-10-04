import torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

from torch.utils.tensorboard import SummaryWriter

from prepare_msvd_dataset import MSVDDataset
from prepare_msrvtt_dataset import MSRVTTDataset

from data_utils import get_loader_and_vocab
import math
from tqdm import tqdm
import json
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

from torch.optim import AdamW

import re

from qformer_bridge import QFormerBridge

torch.manual_seed(3)
torch.cuda.manual_seed(3)
torch.cuda.manual_seed_all(3)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#dt = MSRVTTDataset()
dt = MSVDDataset()

PAD_TOKEN = "pos"
BOS_TOKEN = "bos"
EOS_TOKEN = "eos"
UNK_TOKEN = "unk"


gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
gpt2_tokenizer.pad_token_id = gpt2_tokenizer.eos_token_id
# gpt2_tokenizer.pad_token = PAD_TOKEN
# gpt2_tokenizer.bos_token = BOS_TOKEN
# gpt2_tokenizer.eos_token = EOS_TOKEN
# gpt2_tokenizer.unk_token = UNK_TOKEN

if dt.name == "MSRVTT":
    train_visual_feature_folder = dt.train_visual_features_folder
    test_visual_feature_folder = dt.test_visual_features_folder    
elif dt.name == "MSVD":
    train_visual_feature_folder = dt.train_visual_features_folder
    test_visual_feature_folder = dt.train_visual_features_folder
   

#train_loader, val_loader, test_loader, vocab = get_loader_and_vocab(dt, train_visual_feature_folder, test_visual_feature_folder)
train_loader, val_loader, test_loader = get_loader_and_vocab(dt, tokenizer=gpt2_tokenizer, train_visual_features_folder= train_visual_feature_folder, test_visual_features_folder=test_visual_feature_folder)

coco = COCO()
with open(dt.test_annotations, 'r') as f:
    test = json.load(f)
new_test = []
new_test_images = []
for data in test:
    new_test.append({
        "image_id": data["video_id"],
        "id": data["video_id"],
        "caption": data["caption"]
    })
    new_test_images.append({
        "id": data["video_id"]
    })

test_dataset = dict()
test_dataset["annotations"] = new_test
test_dataset["images"] = new_test_images
coco.dataset = test_dataset
coco.createIndex()

DEVICE = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
config = GPT2Config.from_pretrained('gpt2', add_cross_attention=True)

gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2', config=config)
gpt2_model = gpt2_model.to(DEVICE)

VISION_FEATURE_DIM = 768
bridge = QFormerBridge(
    hidden_size=config.n_embd,
    num_query_tokens=16,
    num_layers=3,
    num_heads=config.n_head,
    dropout=0.1,
    vision_hidden_size=VISION_FEATURE_DIM,
).to(DEVICE)

optimizer = AdamW(
    list(gpt2_model.parameters()) + list(bridge.parameters()),
    lr=5e-5,
)
writer = SummaryWriter(comment=f"______|vit|gpt_2|{dt.name}|")
criterion = torch.nn.CrossEntropyLoss(ignore_index=gpt2_tokenizer.pad_token_id)

def train_epoch(model, bridge_module, optimizer):
    model.train()
    bridge_module.train()
    losses = 0

    for i, (image_feature, input_ids, attention_mask) in tqdm(enumerate(train_loader)):
        
        image_feature = image_feature.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        input_ids = input_ids.to(DEVICE)

        vision_tokens = bridge_module(image_feature)
        encoder_attention_mask = torch.ones(
            vision_tokens.size()[:2],
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
            encoder_hidden_states=vision_tokens,
            encoder_attention_mask=encoder_attention_mask,
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses += loss.item()
    epoch_loss = losses / (i + 1)
    return epoch_loss

def clean_caption_regex(caption, bos_token=gpt2_tokenizer.bos_token, eos_token=gpt2_tokenizer.eos_token, pad_token=gpt2_tokenizer.pad_token):
    pattern = f"({bos_token}|{eos_token}|{pad_token})"
    clean = re.sub(pattern, '', caption)
    clean = clean.strip()
    return clean

def generate_captions(model, bridge_module, src):
    max_len = 30
    batch_size = src.shape[0]
    encoding = gpt2_tokenizer([BOS_TOKEN] * batch_size, return_tensors='pt')
    generated = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)
    vision_tokens = bridge_module(src)
    encoder_attention_mask = torch.ones(
        vision_tokens.size()[:2],
        dtype=attention_mask.dtype,
        device=src.device,
    )
    for _ in range(max_len):
        outputs = model(
            input_ids=generated,
            encoder_hidden_states=vision_tokens,
            encoder_attention_mask=encoder_attention_mask,
            attention_mask=attention_mask,
        )
        predictions = outputs.logits
        next_token_logits = predictions[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        generated = torch.cat((generated, next_token), dim=1)
        # Update the attention mask to include the new token
        new_attention_mask = torch.ones((batch_size, 1), dtype=torch.long, device=generated.device)
        attention_mask = torch.cat((attention_mask, new_attention_mask), dim=1)
    generated_texts = [gpt2_tokenizer.decode(g, skip_special_tokens=True) for g in generated]
    return generated_texts
    

def test_epoch(model, bridge_module, best_score, epoch):
    model.eval()
    bridge_module.eval()
    data = []
    with torch.no_grad():
        for i, (src, ids) in tqdm(enumerate(test_loader)):  
            src = src.to(DEVICE)


            captions = generate_captions(model, bridge_module, src)

            if dt.name == "MSRVTT":
                for caption, id in zip(captions, ids):
                    data.append({
                        "image_id": id.item(),
                        "caption" : caption[4:]
                    })
            elif dt.name == "MSVD":
                for caption, id in zip(captions, ids):
                    data.append({
                        "image_id": id,
                        "caption" : caption[4:]
                    })
    
    
    json_file = f"results/test_result_{dt.name}.json"
    with open(json_file, "w") as file:
        json.dump(data, file)
    
    coco_result = coco.loadRes(json_file)
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.params['image_id'] = coco_result.getImgIds()
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')
        writer.add_scalar(f'{metric} Val Score', score, epoch)
        if metric == "CIDEr":
            if score > best_score:
                best_score = score
                json_file = f"results/best_test_result_{dt.name}.json"
                with open(json_file, "w") as file:
                    json.dump(data, file)
        writer.add_scalar(f'{metric}', score, epoch)

    return best_score
from timeit import default_timer as timer
NUM_EPOCHS = 80
BEST_CIDER_SCORE = 0.0
for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    #train_loss = train_epoch(gpt2_model, bridge, optimizer)
    #writer.add_scalar(f'Train loss', train_loss, epoch)
    end_time = timer()
    BEST_CIDER_SCORE = test_epoch(gpt2_model, bridge, BEST_CIDER_SCORE, epoch)
    #print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    with open('best_cider_score.txt', 'w') as file:
        file.write(f"Best CIDEr Score: {BEST_CIDER_SCORE}")
