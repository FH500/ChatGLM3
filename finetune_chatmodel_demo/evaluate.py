import argparse
from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--pt-checkpoint", type=str, default=None, help="The checkpoint path")
parser.add_argument("--model", type=str, default=None, help="main model weights")
parser.add_argument("--tokenizer", type=str, default=None, help="main model weights")
parser.add_argument("--pt-pre-seq-len", type=int, default=128, help="The pre-seq-len used in p-tuning")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--max-new-tokens", type=int, default=128)
parser.add_argument("--dev-path", type=str, default=None)

args = parser.parse_args()

if args.tokenizer is None:
    args.tokenizer = args.model

if args.pt_checkpoint:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True, pre_seq_len=args.pt_pre_seq_len)
    model = AutoModel.from_pretrained(args.model, config=config, trust_remote_code=True).cuda()
    prefix_state_dict = torch.load(os.path.join(args.pt_checkpoint, "pytorch_model.bin"))
    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        if k.startswith("transformer.prefix_encoder."):
            new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True)

# model = model.to(args.device)

devfilepath = args.dev_path
dev_data = []
with open(devfilepath, "r", encoding="utf-8") as f:
    for line in f.readlines():
        x = json.loads(line)
        dev_data.append(x)

count = 0
full_path = os.path.join(args.pt_checkpoint, "evaluate.json")
f = open(full_path, "w")
while True:
    if count >= len(dev_data):
        break
    prompt = "<|system|>\n<|user|>\n"
    userin = dev_data[count]['prompt']
    prompt += userin
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to(args.device)
    response = model.generate(input_ids=inputs["input_ids"], max_length=inputs["input_ids"].shape[-1] + args.max_new_tokens, temperature = 0.8, top_p = 0.8, do_sample = True)
    response = response[0, inputs["input_ids"].shape[-1]:]
    infer = tokenizer.decode(response, skip_special_tokens=True)
    expect = dev_data[count]['response']
    sjson = {"prompt" : userin,"infer" : infer, "expect" : expect}
    out = json.dumps(sjson, indent=5, separators=(",", " : "), ensure_ascii=False)
    f.write(out)
    print("Progress: ", count, "/", len(dev_data))
    count += 1
f.close()