import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.model import convert_weights
from tqdm import tqdm
from dassl.utils import save_checkpoint

_tokenizer = _Tokenizer()
class Feature_Trans_Module_two_layer(nn.Module):
    def __init__(self, input_dim=100, out_dim=256):
        super(Feature_Trans_Module_two_layer, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, out_dim, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, 1)
        )
    def forward(self, input_feat): # input_feat:[B, d] [B, N, d]
        
        final_feat = self.conv1(input_feat.unsqueeze(-1).unsqueeze(-1))
        
        return final_feat.squeeze(-1).squeeze(-1)


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model, device):
        super().__init__()
        self.device = device  # 保存设备
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.token_embedding = clip_model.token_embedding
        self.dtype = clip_model.dtype
        

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    # zero-shot clip
    def encode_text(self, text):
         text = text.to(self.device)  # 替换.text.cuda()
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
                
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init


        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]


        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS
        self.register_buffer("tokenized_prompts", tokenized_prompts)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        #self.text_encoder = TextEncoder(clip_model)
        self.text_encoder = TextEncoder(clip_model, device=device)  # 传入device
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
        self.mid_image_trans = Feature_Trans_Module_two_layer(512, 512)
        # 修正后：使用动态设备（需传入trainer的self.device）
        self.mid_image_trans = self.mid_image_trans.to(device)  # device从外部传入
        convert_weights(self.mid_image_trans)
        
        
       
    def forward(self, image):
        image_features, image_fine = self.image_encoder(image.type(self.dtype), all_layer_outputs=True)

        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()

        image_fine_list = []
        top_image_fine_list = []
        B, d = image_features.shape
        image_fine_features = [x[0] for x in image_fine] # B,N,d
        image_fine_attns =  [x[1] for x in image_fine] # B,1,N
        layers=[-1] 
        _, _, before_d = image_fine_features[0].shape
        loss = 0.0
        for i, layer in enumerate(layers):
            x = image_fine_features[layer]
            x = x.reshape(-1, before_d)

            if self.training:
                x = self.mid_image_trans(x)
                x = x.reshape(B, -1, d)

                image_fine_feature = x
                image_fine_list.append(image_fine_feature)
                
                
                k = 5
                _, indices = torch.topk(image_fine_attns[layer], k = k ,dim=-1)
                indices += 1
                indices = torch.cat((torch.zeros(B,1,dtype=torch.int64).cuda(),indices), dim=1)
                top_image_fine_feature = torch.gather(x, dim=1, index=indices.unsqueeze(-1).expand(B, k+1, d))
                avg_image_feature = torch.mean(x, dim = 1, keepdim = True)
                top_image_fine_feature = torch.cat((top_image_fine_feature, avg_image_feature), dim = 1)

                top_image_fine_list.append(top_image_fine_feature.reshape(-1, d))


        
        if len(image_fine_list) > 0:
            image_fine_list = torch.cat(image_fine_list)
            top_image_fine_list = torch.cat(top_image_fine_list)
        
        return text_features, image_features, logit_scale, image_fine_list, top_image_fine_list


class Memory(nn.Module):
    def __init__(self, clip_model,feature_dim = 512, memory_size = 25, reduction = 4, frozen_text_embedding=None, alpha=0.2,momentum=0.8,device=None):
        super().__init__()
        # 修正1：正确获取设备（优先用传入的device，否则从clip_model取）
        self.device = device if device is not None else clip_model.logit_scale.device
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.text_fine_cache = F.normalize(torch.rand(self.memory_size, feature_dim), dim = -1)
        self.text_fine_cache = self.text_fine_cache.to(self.device)
        
        self.alpha = alpha
        self.momentum = momentum

        if frozen_text_embedding is not None:
            self.frozen_text_embedding = frozen_text_embedding.to(self.device)  # 确保embedding在正确设备

        
        self.extractor = nn.Linear(2 * feature_dim , feature_dim, bias=False)
        self.extractor = self.extractor.to(self.device)
        


        self.writeTF = lambda x: x.clone()
        
    def forward(self, text_token=None, image_token=None):
        fine_feature = self.read(text_token)

        text_fine_feature = torch.cat((text_token, fine_feature), dim = -1)
        text_fine_feature = self.alpha * self.extractor(text_fine_feature) + text_token
        if self.training:
            _ = self.write(image_token)
            normalized_text_features = F.normalize(text_fine_feature, dim = -1)
            loss = F.l1_loss(normalized_text_features, text_token, reduction='mean') 
        else:
            loss = 0.0
        return text_fine_feature, loss

    def get_score(self, query, mem):
        score = query @ mem.t() 
        score_query = F.softmax(score, dim = 0)
        score_mem = F.softmax(score, dim = 1)
        return score_query, score_mem
    
    def read(self, x):
        base_features = F.normalize(x, dim = -1) 
        C, d = x.shape
        if self.training:
            self.text_fine_cache = self.text_fine_cache.detach()
        _, softmax_score_cache = self.get_score(base_features, self.text_fine_cache)
        fine_feature = softmax_score_cache @ self.text_fine_cache  # (N, d)
        
        return fine_feature

    def write(self, x):
        m, d = self.text_fine_cache.shape
        ratio = 0.2
        base_features = x.clone()
        base_features = self.writeTF(base_features)


        base_features = base_features.reshape(-1, d) # (B * P, d)
        base_features = F.normalize(base_features, dim = -1) 


        softmax_score_query, softmax_score_cache = self.get_score(base_features, self.text_fine_cache) #(B*P, 50)
        _, updating_indices = torch.topk(softmax_score_cache, 1, dim=1)
        

        updated_cache = self.text_fine_cache.clone().detach()
        for i in range(m):
            idx = torch.nonzero(updating_indices.squeeze(1) == i)
            a, _ = idx.size()
            if a != 0:
                score = (softmax_score_query[idx, i] / torch.max(softmax_score_query[:, i]))
                updated_cache[i] = self.momentum * self.text_fine_cache[i] + (1 - self.momentum) * torch.sum(score * base_features[idx.squeeze(1)], dim=0)
        
        updated_cache = F.normalize(updated_cache, dim = -1)

        loss = 0.0
        self.text_fine_cache = updated_cache.to(self.device)
        return loss
    def diversityloss(self, mem):
        # it is same with orthonomal constraints.
        cos_sim = torch.matmul(mem,torch.t(mem))
        margin = 0 
        cos_sim_pos = cos_sim-margin
        cos_sim_pos[cos_sim_pos<0]=0
        loss = (torch.sum(cos_sim_pos)-torch.trace(cos_sim_pos))/(self.memory_size*(self.memory_size-1))
        return loss
    


@TRAINER_REGISTRY.register()
class TextRefiner(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, device=self.device)  # 新增device参数

        print("Building memory")
        self.memory = Memory(clip_model,feature_dim=512, memory_size=cfg.TRAINER.TF.MEMORY_SIZE, alpha=cfg.TRAINER.TF.ALPHA,device=self.device  # 传入trainer的设备)
        if cfg.TRAINER.TF.BALANCE:
            self.balance = cfg.TRAINER.TF.BALANCE
        if cfg.TRAINER.TF.DISTILL:
            self.distill = cfg.TRAINER.TF.DISTILL
        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" in name or 'mid_image_trans' in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
            
        print("check hyper-parameter balance: ", self.balance)
        print("check hyper-parameter distill: ", self.distill)
        print("check hyper-parameter alpha: ", self.memory.alpha)
        print("check hyper-parameter momentum: ", self.memory.momentum)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        
        # NOTE: only give prompt_learner to the optimizer
        self.trainable_list = nn.ModuleList([])
        self.trainable_list.append(self.model)
        self.trainable_list.append(self.memory)
        self.optim = build_optimizer(self.trainable_list, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        self.register_model("prompt_learner", self.model, self.optim, self.sched)
        self.register_model("Memory", self.memory, self.optim, self.sched)

        enabled = set()
        for name, param in self.trainable_list.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print(f"Parameters count: {len(enabled)}")

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)


    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            text_features, image_features, logit_scale, image_fine, top_image_fine = self.model(image)
            
            fine_text_features, loss3 = self.memory(text_features, image_fine)

            output = logit_scale * image_features @ fine_text_features.t()
            top_local_logit = logit_scale * top_image_fine @ text_features.t()

            loss1 = F.cross_entropy(output, label)
            
            loss2 = F.cross_entropy(top_local_logit, label.repeat_interleave(dim=0, repeats=7))
            
            loss = loss1 + loss2 / self.balance + loss3 * self.distill

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        loss_summary = {
            "loss": loss.item(),
            "global": loss1.item(),
            "local": loss2.item() / self.balance,
            "feature": loss3.item() * self.distill,
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.sched.step()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]
            
            if "prompt_learner.tokenized_prompts" in state_dict:
                del state_dict["prompt_learner.tokenized_prompts"]

            if "frozen_text_embedding" in state_dict:
                del state_dict["frozen_text_embedding"]

            # Ignore abstract layer
            keys=[key for key in state_dict.keys() if 'mid_image_trans' in key]
            state_dict = {k:v for k,v in state_dict.items() if k not in keys}

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
            if 'Mem' in name:
                #self._models[name].text_fine_cache = checkpoint["memory_item"]
                self._models[name].text_fine_cache = checkpoint["memory_item"].to(self.device)  # 新增.to(self.device)

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            image, label = self.parse_batch_test(batch)

            text_features, image_features, logit_scale, image_fine, _ = self.model(image)
            fine_text_features, _ = self.memory(text_features, image_fine)

            output = logit_scale * image_features @ fine_text_features.t()
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)
        

        return list(results.values())[0]
    
    def set_model_mode(self, mode="train", names=None):
        names = self.get_model_names(names)

        for name in names:
            if mode == "train":
                self._models[name].train()
                self.training = True
            elif mode in ["test", "eval"]:
                self._models[name].eval()
                self.training = False
            else:
                raise KeyError
            
    def save_model(
        self, epoch, directory, is_best=False, val_result=None, model_name=""
    ):
        names = self.get_model_names()

        for name in names:
            model_dict = self._models[name].state_dict()

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()

            if 'Mem' in name:
                print("save memory item...")
                save_checkpoint(
                    {
                        "memory_item": self._models[name].text_fine_cache,
                        "state_dict": model_dict,
                        "epoch": epoch + 1,
                        "optimizer": optim_dict,
                        "scheduler": sched_dict,
                        "val_result": val_result
                    },
                    osp.join(directory, name),
                    is_best=is_best,
                    model_name=model_name,
                )
            else:
                save_checkpoint(
                    {
                        "state_dict": model_dict,
                        "epoch": epoch + 1,
                        "optimizer": optim_dict,
                        "scheduler": sched_dict,
                        "val_result": val_result
                    },
                    osp.join(directory, name),
                    is_best=is_best,
                    model_name=model_name,
                )
