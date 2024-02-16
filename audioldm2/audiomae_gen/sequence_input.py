import torch
import torch.nn as nn
from audioldm2.latent_diffusion.util import (
    instantiate_from_config,
)

# from latent_diffusion.modules.encoders.modules import CLAPAudioEmbeddingClassifierFreev2
from transformers import GPT2Config, GPT2Model
import torch.optim.lr_scheduler as lr_scheduler

# Sequence2AudioMAE 클래스를 정의. 이 클래스는 PyTorch의 nn.Module을 상속받아 생성.
class Sequence2AudioMAE(nn.Module):
    def __init__(
        self,
        base_learning_rate,
        sequence_gen_length,
        sequence_input_key,
        sequence_input_embed_dim,
        cond_stage_config,
        optimizer_type="AdamW",
        use_warmup=True,
        use_ar_gen_loss=False,
        use_audiomae_linear=False,
        target_tokens_mask_ratio=0.0,
        random_mask_ratio=False,
        **kwargs
    ):
        super().__init__()  # 부모 클래스인 nn.Module의 생성자를 호출
        assert use_audiomae_linear == False # use_audiomae_linear 플래그가 False인 것을 확인. 현재 선형 오디오 MAE는 사용되지 않는다는 가정.
        self.random_mask_ratio = random_mask_ratio # 학습률을 설정
        self.learning_rate = base_learning_rate # 조건 단계의 설정을 저장
        self.cond_stage_config = cond_stage_config
        self.use_audiomae_linear = use_audiomae_linear
        self.optimizer_type = optimizer_type  # 옵티마이저의 유형을 설정
        self.use_warmup = use_warmup # 학습률 예열 사용 여부를 설정
        self.use_ar_gen_loss = use_ar_gen_loss
        # Even though the LDM can be conditioned on mutliple pooling rate
        # Our model always predict the higest pooling rate

        # self.time_pool = max(self.cond_stage_config["crossattn_audiomae_pooled"]["params"]["time_pooling_factors"])
        # self.freq_pool = max(self.cond_stage_config["crossattn_audiomae_pooled"]["params"]["freq_pooling_factors"])
        # self.mae_token_num = int(512/(self.time_pool*self.freq_pool))

        self.mae_token_num = sequence_gen_length # 생성할 시퀀스의 길이를 설정
        self.sequence_input_key = sequence_input_key # 입력 시퀀스의 키를 설정
        self.sequence_input_embed_dim = sequence_input_embed_dim  # 입력 임베딩 차원을 설정
        self.target_tokens_mask_ratio = target_tokens_mask_ratio # 타겟 토큰 마스킹 비율을 설정

        # SOS(Start of Sequence)와 EOS(End of Sequence) 토큰을 위한 임베딩 레이어를 생성
        self.start_of_sequence_tokens = nn.Embedding(32, 768)
        self.end_of_sequence_tokens = nn.Embedding(32, 768)

        # 입력 시퀀스 임베딩을 위한 선형 레이어를 포함하는 모듈 리스트를 생성
        self.input_sequence_embed_linear = nn.ModuleList([])
        self.initial_learning_rate = None

        # 입력 임베딩 차원에 따라 선형 레이어를 추가.
        for dim in self.sequence_input_embed_dim:
            self.input_sequence_embed_linear.append(nn.Linear(dim, 768))

        # 조건 단계 모델을 위한 모듈 리스트를 생성하고, 이를 초기화
        self.cond_stage_models = nn.ModuleList([])
        self.instantiate_cond_stage(cond_stage_config)
        self.initialize_param_check_toolkit()

        # configuration = GPT2Config(n_layer=1) # TODO
        # self.model=GPT2Model(configuration)
        ###################
        # self.model=nn.Linear(768,768, bias=False) # TODO change the model
        # with torch.no_grad():
        #     self.model.weight.copy_(torch.eye(768))
        ###################

        # GPT-2 모델을 구성하고 인스턴스화. 여기서는 사전 훈련된 "gpt2" 모델을 사용
        self.model = GPT2Model(GPT2Config.from_pretrained("gpt2"))
        ###################
        # self.model = nn.LSTM(input_size=768, hidden_size=768, num_layers=1,bias=False) # TODO

        # self.loss_fn = nn.MSELoss()

        # 손실 함수를 L1 손실로 설정.
        self.loss_fn = nn.L1Loss()

        # 로거 관련 설정을 초기화
        self.logger_save_dir = None
        self.logger_exp_name = None
        self.logger_exp_group_name = None
        self.logger_version = None

    """set_log_dir 메소드
    로깅을 위한 디렉토리 설정을 담당.
    save_dir: 로그 파일을 저장할 경로.
    exp_group_name: 실험 그룹의 이름을 지정. 여러 실험을 그룹화할 때 사용.
    exp_name: 개별 실험의 이름을 지정.
    """
    def set_log_dir(self, save_dir, exp_group_name, exp_name):
        self.logger_save_dir = save_dir
        self.logger_exp_group_name = exp_group_name
        self.logger_exp_name = exp_name

    """cfg_uncond 메소드
    무조건적(unconditional) 조건 설정을 생성.
    각 조건 스테이지 모델의 무조건적 조건을 수집하여 unconditional_conditioning 딕셔너리에 저장.
    batch_size: 생성할 조건의 배치 크기.
    이 메소드는 "crossattn_audiomae_pooled"을 포함하는지 확인하며, 이는 오디오MAE의 특징을 CLAP을 통해 오디오MAE 특징으로 변환하는 데 사용.
    """
    def cfg_uncond(self, batch_size):
        unconditional_conditioning = {}
        for key in self.cond_stage_model_metadata:
            model_idx = self.cond_stage_model_metadata[key]["model_idx"]
            unconditional_conditioning[key] = self.cond_stage_models[
                model_idx
            ].get_unconditional_condition(batch_size)
        assert (
            "crossattn_audiomae_pooled" in unconditional_conditioning.keys()
        ), "The module is not initialized with AudioMAE"
        unconditional_conditioning[
            "crossattn_clap_to_audiomae_feature"
        ] = unconditional_conditioning["crossattn_audiomae_pooled"]
        return unconditional_conditioning
    
    """
    configure_optimizers 메소드
    모델의 옵티마이저와 스케줄러를 설정.
    lr: 학습률을 설정.
    params: 모델의 파라미터들.
    opt: 옵티마이저를 설정. optimizer_type에 따라 동적으로 옵티마이저 타입이 결정.
    scheduler: 학습률 스케줄러를 설정. 여기서는 StepLR을 사용하여 특정 스텝마다 학습률을 감소.
    """

    def configure_optimizers(self):
        lr = float(self.learning_rate)
        # params = list(self.model.parameters()) + list(self.input_sequence_embed_linear.parameters())
        params = list(self.parameters())

        # opt = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.98), eps=1e-9)
        opt = eval(self.optimizer_type)(params, lr=lr)
        scheduler = lr_scheduler.StepLR(opt, step_size=10, gamma=0.8)
        return [opt], [scheduler]
    
    """
    add_sos_eos_tokens 메소드
    입력 시퀀스에 시작(SOS) 토큰과 종료(EOS) 토큰을 추가.
    _id: 시퀀스 키의 ID.
    sequence: 원본 입력 시퀀스.
    attn_mask: 입력 시퀀스에 대한 어텐션 마스크.
    이 메소드는 각 시퀀스의 시작과 끝에 특별한 토큰을 추가하여 시퀀스를 변형.
    """

    def add_sos_eos_tokens(self, _id, sequence, attn_mask):
        batchsize = sequence.size(0)

        new_attn_mask_step = torch.ones((batchsize, 1)).to(sequence.device)
        key_id = torch.tensor([_id]).to(sequence.device)

        # Add two more steps to attn mask
        new_attn_mask = torch.cat(
            [new_attn_mask_step, attn_mask, new_attn_mask_step], dim=1
        )

        # Add two more tokens in the sequence
        sos_token = self.start_of_sequence_tokens(key_id).expand(batchsize, 1, -1)
        eos_token = self.end_of_sequence_tokens(key_id).expand(batchsize, 1, -1)
        new_sequence = torch.cat([sos_token, sequence, eos_token], dim=1)
        return new_sequence, new_attn_mask
    
    """
    truncate_sequence_and_mask 메소드
    입력 시퀀스와 마스크를 최대 길이로 자름.
    sequence, mask: 입력 시퀀스와 어텐션 마스크.
    max_len: 시퀀스의 최대 길이. 이보다 긴 시퀀스는 잘림.
    get_input_sequence_and_mask 메소드
    조건 딕셔너리로부터 입력 시퀀스와 어텐션 마스크를 생성.
    cond_dict: 조건 딕셔너리. 각 조건은 시퀀스 키에 매핑.
    이 메소드는 각 시퀀스 키에 대한 입력 임베딩과 어텐션 마스크를 생성, 필요한 경우 SOS/EOS 토큰을 추가.
    """

    def truncate_sequence_and_mask(self, sequence, mask, max_len=512):
        if sequence.size(1) > max_len:
            print(
                "The input sequence length to GPT-2 model is too long:",
                sequence.size(1),
            )
            return sequence[:, :max_len], mask[:, :max_len]
        else:
            return sequence, mask
        
    """
    get_input_sequence_and_mask 메소드
    조건 딕셔너리로부터 입력 시퀀스와 어텐션 마스크를 생성.
    cond_dict: 조건 딕셔너리. 각 조건은 시퀀스 키에 매핑.
    이 메소드는 각 시퀀스 키에 대한 입력 임베딩과 어텐션 마스크를 생성하고, 필요한 경우 SOS/EOS 토큰을 추가.
    """

    def get_input_sequence_and_mask(self, cond_dict):
        input_embeds = None
        input_embeds_attn_mask = None
        for _id, sequence_key in enumerate(self.sequence_input_key):
            assert sequence_key in cond_dict.keys(), (
                "Invalid sequence key %s" % sequence_key
            )
            cond_embed = cond_dict[sequence_key]
            if isinstance(cond_embed, list):
                assert (
                    len(cond_embed) == 2
                ), "The crossattn returned list should have length 2, including embed and attn_mask"
                item_input_embeds, item_attn_mask = cond_embed

                item_input_embeds = self.input_sequence_embed_linear[_id](
                    item_input_embeds
                )

                item_input_embeds, item_attn_mask = self.add_sos_eos_tokens(
                    _id, item_input_embeds, item_attn_mask
                )

                if input_embeds is None and input_embeds_attn_mask is None:
                    input_embeds, input_embeds_attn_mask = (
                        item_input_embeds,
                        item_attn_mask,
                    )
                else:
                    input_embeds = torch.cat(
                        [input_embeds, item_input_embeds], dim=1
                    )  # The 1-st dimension is time steps
                    input_embeds_attn_mask = torch.cat(
                        [input_embeds_attn_mask, item_attn_mask], dim=1
                    )  # The 1-st dimension is time steps
            else:
                assert isinstance(cond_embed, torch.Tensor)
                cond_embed = self.input_sequence_embed_linear[_id](cond_embed)
                attn_mask = torch.ones((cond_embed.size(0), cond_embed.size(1))).to(
                    cond_embed.device
                )

                item_input_embeds, item_attn_mask = self.add_sos_eos_tokens(
                    _id, cond_embed, attn_mask
                )

                if input_embeds is None and input_embeds_attn_mask is None:
                    input_embeds, input_embeds_attn_mask = (
                        item_input_embeds,
                        item_attn_mask,
                    )
                else:
                    input_embeds, input_embeds_attn_mask = torch.cat(
                        [input_embeds, item_input_embeds], dim=1
                    ), torch.cat([input_embeds_attn_mask, item_attn_mask], dim=1)

        assert input_embeds is not None and input_embeds_attn_mask is not None

        input_embeds, input_embeds_attn_mask = self.truncate_sequence_and_mask(
            input_embeds, input_embeds_attn_mask, int(1024 - self.mae_token_num)
        )
        cond_sequence_end_time_idx = input_embeds.size(
            1
        )  # The index that we start to collect the output embeds

        return input_embeds, input_embeds_attn_mask, cond_sequence_end_time_idx
    
    """
    warmup_step 메소드
    학습 초기에 학습률을 점진적으로 증가시키는 웜업(warmup) 단계를 구현.
    global_step을 사용하여 현재 스텝에 따라 학습률을 조절.
    """

    def warmup_step(self):
        if self.initial_learning_rate is None:
            self.initial_learning_rate = float(self.learning_rate)

        # Only the first parameter group
        if self.global_step <= 1000:
            if self.global_step == 0:
                print(
                    "Warming up learning rate start with %s"
                    % self.initial_learning_rate
                )
            self.trainer.optimizers[0].param_groups[0]["lr"] = (
                self.global_step / 1000
            ) * self.initial_learning_rate
        else:
            # TODO set learning rate here
            self.trainer.optimizers[0].param_groups[0][
                "lr"
            ] = self.initial_learning_rate

    """
    mask_target_sequence 메소드
    타겟 시퀀스에 마스크를 적용.
    target_tokens_mask_ratio에 따라 시퀀스의 일부 토큰을 마스킹.
    마스킹은 훈련 과정에서 모델이 데이터의 다양한 부분에 초점을 맞추도록 도움.
    """

    def mask_target_sequence(self, target_embeds, target_embeds_attn_mask):
        time_seq_mask = None
        if self.target_tokens_mask_ratio > 1e-4:
            batchsize, time_seq_len, embed_dim = target_embeds.size()
            _, time_seq_len = target_embeds_attn_mask.size()
            # Generate random mask
            if self.random_mask_ratio:
                mask_ratio = torch.rand(1).item() * self.target_tokens_mask_ratio
            else:
                mask_ratio = self.target_tokens_mask_ratio

            time_seq_mask = (torch.rand((batchsize, time_seq_len)) > mask_ratio).to(
                target_embeds.device
            )
            # Mask the target embedding
            target_embeds = target_embeds * time_seq_mask.unsqueeze(-1)
            target_embeds_attn_mask = target_embeds_attn_mask * time_seq_mask
        return target_embeds, target_embeds_attn_mask, time_seq_mask
    
    """
    generate_partial 및 generate 메소드
    주어진 조건에 따라 부분적 또는 전체 오디오 시퀀스를 생성.
    cond_dict에서 조건을 가져와 GPT-2 모델 등을 사용하여 시퀀스를 생성
    """

    def generate_partial(self, batch, cond_dict=None, no_grad=False):
        if cond_dict is None:
            cond_dict = self.get_input(batch)

        print("Generate partially prompted audio with in-context learning")
        # self.model.train()
        # assert self.model.training==True

        target_embeds, target_embeds_attn_mask = (
            cond_dict["crossattn_audiomae_pooled"][0],
            cond_dict["crossattn_audiomae_pooled"][1],
        )

        target_time_steps = target_embeds.size(1)

        (
            input_embeds,
            input_embeds_attn_mask,
            cond_sequence_end_time_idx,
        ) = self.get_input_sequence_and_mask(cond_dict)

        model_input = torch.cat(
            [input_embeds, target_embeds[:, : target_time_steps // 4, :]], dim=1
        )
        model_input_mask = torch.cat(
            [
                input_embeds_attn_mask,
                target_embeds_attn_mask[:, : target_time_steps // 4],
            ],
            dim=1,
        )

        steps = self.mae_token_num

        for _ in range(3 * steps // 4):
            output = self.model(
                inputs_embeds=model_input, attention_mask=model_input_mask
            )["last_hidden_state"]
            # Update the model input
            model_input = torch.cat([model_input, output[:, -1:, :]], dim=1)
            # Update the attention mask
            attention_mask_new_step = torch.ones((model_input_mask.size(0), 1)).to(
                model_input.device
            )
            model_input_mask = torch.cat(
                [model_input_mask, attention_mask_new_step], dim=1
            )

        output = model_input[:, cond_sequence_end_time_idx:]

        return output, cond_dict

    def generate(self, batch, cond_dict=None, no_grad=False):
        if cond_dict is None:
            cond_dict = self.get_input(batch)

        # self.model.train()
        # print("!!!!!!!!!!!!!train")

        (
            input_embeds,
            input_embeds_attn_mask,
            cond_sequence_end_time_idx,
        ) = self.get_input_sequence_and_mask(cond_dict)
        model_input = input_embeds
        model_input_mask = input_embeds_attn_mask

        steps = self.mae_token_num

        for _ in range(steps):
            output = self.model(
                inputs_embeds=model_input, attention_mask=model_input_mask
            )["last_hidden_state"]
            # Update the model input
            model_input = torch.cat([model_input, output[:, -1:, :]], dim=1)
            # Update the attention mask
            attention_mask_new_step = torch.ones((model_input_mask.size(0), 1)).to(
                model_input.device
            )
            model_input_mask = torch.cat(
                [model_input_mask, attention_mask_new_step], dim=1
            )

        return model_input[:, cond_sequence_end_time_idx:], cond_dict
    
    """
    get_input_item 및 get_input 메소드
    배치에서 필요한 입력 항목을 추출.
    get_input_item은 개별 항목을, get_input은 전체 조건 딕셔너리를 생성.
    """

    def get_input_item(self, batch, k):
        fname, text, waveform, stft, fbank = (
            batch["fname"],
            batch["text"],
            batch["waveform"],
            batch["stft"],
            batch["log_mel_spec"],
        )
        ret = {}

        ret["fbank"] = (
            fbank.unsqueeze(1).to(memory_format=torch.contiguous_format).float()
        )
        ret["stft"] = stft.to(memory_format=torch.contiguous_format).float()
        # ret["clip_label"] = clip_label.to(memory_format=torch.contiguous_format).float()
        ret["waveform"] = waveform.to(memory_format=torch.contiguous_format).float()
        ret["text"] = list(text)
        ret["fname"] = fname

        for key in batch.keys():
            if key not in ret.keys():
                ret[key] = batch[key]

        return ret[k]

    def get_input(self, batch):
        cond_dict = {}
        if len(self.cond_stage_model_metadata.keys()) > 0:
            unconditional_cfg = False

            for cond_model_key in self.cond_stage_model_metadata.keys():
                cond_stage_key = self.cond_stage_model_metadata[cond_model_key][
                    "cond_stage_key"
                ]

                # if(not self.training):
                #     if(isinstance(self.cond_stage_models[self.cond_stage_model_metadata[cond_model_key]["model_idx"]], CLAPAudioEmbeddingClassifierFreev2)):
                #         assert cond_stage_key == "text" # CLAP model should use text for evaluation

                # The original data for conditioning
                xc = self.get_input_item(batch, cond_stage_key)
                if type(xc) == torch.Tensor:
                    xc = xc.to(self.device)

                c = self.get_learned_conditioning(
                    xc, key=cond_model_key, unconditional_cfg=unconditional_cfg
                )
                cond_dict[cond_model_key] = c

        return cond_dict
    
    """instantiate_cond_stage 메소드
    조건 스테이지 모델들을 인스턴스화.
    config에서 제공된 설정에 따라 모델을 생성하고 초기화."""

    def instantiate_cond_stage(self, config):
        self.cond_stage_model_metadata = {}

        for i, cond_model_key in enumerate(config.keys()):
            model = instantiate_from_config(config[cond_model_key])
            self.cond_stage_models.append(model)
            self.cond_stage_model_metadata[cond_model_key] = {
                "model_idx": i,
                "cond_stage_key": config[cond_model_key]["cond_stage_key"],
                "conditioning_key": config[cond_model_key]["conditioning_key"],
            }

    """
    get_learned_conditioning 메소드
    학습된 조건을 생성.
    주어진 입력에 대해 조건 스테이지 모델을 사용하여 조건을 생성.
    """

    def get_learned_conditioning(self, c, key, unconditional_cfg):
        assert key in self.cond_stage_model_metadata.keys()

        # Classifier-free guidance
        if not unconditional_cfg:
            c = self.cond_stage_models[
                self.cond_stage_model_metadata[key]["model_idx"]
            ](c)
        else:
            if isinstance(c, torch.Tensor):
                batchsize = c.size(0)
            elif isinstance(c, list):
                batchsize = len(c)
            else:
                raise NotImplementedError()
            c = self.cond_stage_models[
                self.cond_stage_model_metadata[key]["model_idx"]
            ].get_unconditional_condition(batchsize)

        return c
    
    """
    initialize_param_check_toolkit 및 statistic_require_grad_tensor_number 메소드
    모델 파라미터와 학습 가능한 텐서의 수를 추적 및 출력.
    모델의 파라미터 상태를 모니터링하는 데 사용.
    """

    def initialize_param_check_toolkit(self):
        self.tracked_steps = 0
        self.param_dict = {}

    def statistic_require_grad_tensor_number(self, module, name=None):
        requires_grad_num = 0
        total_num = 0
        require_grad_tensor = None
        for p in module.parameters():
            if p.requires_grad:
                requires_grad_num += 1
                if require_grad_tensor is None:
                    require_grad_tensor = p
            total_num += 1
        print(
            "Module: [%s] have %s trainable parameters out of %s total parameters (%.2f)"
            % (name, requires_grad_num, total_num, requires_grad_num / total_num)
        )
        return require_grad_tensor
