"""
Implementation based on Multimodal Diffusion Transformer https://github.com/intuitive-robots/mdt_policy.git
"""
from typing import Callable, Union
import math
from collections import OrderedDict, deque
from packaging.version import parse as parse_version
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
# requires diffusers==0.11.1
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel

import robomimic.models.obs_nets as ObsNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo

import random
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import os

from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
import open_clip
import cv2
import copy
from .transformers.transformer_blocks import TransformerEncoder, TransformerFiLMDecoder

import random

@register_algo_factory_func("mdt")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """


    return MDT, {}

class MDT(PolicyAlgo):
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        # set up different observation groups for @MIMO_MLP
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)
        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder)
        
        obs_encoder = ObsNets.ObservationGroupEncoder(
            observation_group_shapes=observation_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )


        # IMPORTANT!
        # replace all BatchNorm with GroupNorm to work with EMA
        # performance will tank if you forget to do this!
        obs_encoder = replace_bn_with_gn(obs_encoder)
        
        obs_dim = obs_encoder.output_shape()[0]

        # create network object
        noise_pred_net = MDTTransformer(
            input_dim=self.ac_dim,
            diffusion_step_embed_dim=self.algo_config.diffusion_step_embed_dim,
            embed_dim=self.algo_config.embed_dim,
            n_heads=self.algo_config.n_heads,
            attn_pdrop=self.algo_config.attn_pdrop,
            resid_pdrop=self.algo_config.resid_pdrop,
            mlp_pdrop=self.algo_config.mlp_pdrop,
            n_enc_layers=self.algo_config.n_enc_layers,
            n_dec_layers=self.algo_config.n_dec_layers,
            block_size=self.algo_config.block_size,
            mlp_size=self.algo_config.mlp_size,
        )

        # the final arch has 2 parts
        nets = nn.ModuleDict({
            'policy': nn.ModuleDict({
                'obs_encoder': obs_encoder,
                'noise_pred_net': noise_pred_net
            })
        })
        
        nets = nets.float().to(self.device)
        
        # setup noise scheduler
        noise_scheduler = None
        if self.algo_config.ddpm.enabled:
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.algo_config.ddpm.num_train_timesteps,
                beta_schedule=self.algo_config.ddpm.beta_schedule,
                clip_sample=self.algo_config.ddpm.clip_sample,
                prediction_type=self.algo_config.ddpm.prediction_type
            )
        elif self.algo_config.ddim.enabled:
            noise_scheduler = DDIMScheduler(
                num_train_timesteps=self.algo_config.ddim.num_train_timesteps,
                beta_schedule=self.algo_config.ddim.beta_schedule,
                clip_sample=self.algo_config.ddim.clip_sample,
                set_alpha_to_one=self.algo_config.ddim.set_alpha_to_one,
                steps_offset=self.algo_config.ddim.steps_offset,
                prediction_type=self.algo_config.ddim.prediction_type
            )
        else:
            raise RuntimeError()
                
        # set attrs
        self.nets = nets
        self.noise_scheduler = noise_scheduler
        self.action_check_done = False
        self.obs_queue = None
        self.action_queue = None
        self.encode_lang = None
        
        if self.global_config.experiment.ckpt_path is not None:
            from robomimic.utils.file_utils import maybe_dict_from_checkpoint
            ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=self.global_config.experiment.ckpt_path)
            self.deserialize(ckpt_dict["model"])
            
            policy = nets['policy']
            
            for key in policy.keys():
            
                linear_layers = []

                for name, module in policy[key].named_modules():
                    if isinstance(module, torch.nn.Linear):
                        linear_layers.append(name)
                        
                lora_config = LoraConfig(
                    r=32,
                    lora_alpha=16,
                    lora_dropout=0.0,
                    target_modules=linear_layers,
                    init_lora_weights="gaussian",
                )
                
                nets['policy'][key] = get_peft_model(nets['policy'][key], lora_config)
        
        self.accelerator.prepare(nets)
        
        self.nets = nets
        
        # setup EMA
        ema = None
        if self.algo_config.ema.enabled:
            ema = EMAModel(model=nets, power=self.algo_config.ema.power)
        self.ema = ema
        
    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training 
        """
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon

        input_batch = dict()

        ## Semi-hacky fix which does the filtering for raw language which is just a list of lists of strings
        input_batch["obs"] = {k: batch["obs"][k][:, :To, :] for k in batch["obs"] if "raw" not in k }
        if "lang_fixed/language_raw" in batch["obs"].keys():
            str_ls = list(batch['obs']['lang_fixed/language_raw'][0])
            input_batch["obs"]["lang_fixed/language_raw"] = [str_ls] * To

        input_batch["actions"] = batch["actions"][:, :Tp, :]
        
        # check if actions are normalized to [-1,1]
        if not self.action_check_done:
            actions = input_batch["actions"]
            in_range = (-1 <= actions) & (actions <= 1)
            all_in_range = torch.all(in_range).item()
            if not all_in_range:
                raise ValueError('"actions" must be in range [-1,1] for Diffusion Policy! Check if hdf5_normalize_action is enabled.')
            self.action_check_done = True

        for key in input_batch["obs"]:
            input_batch["obs"][key] = torch.nan_to_num(input_batch["obs"][key], nan=0.0, posinf=0.0, neginf=0.0)
        input_batch["actions"] = torch.nan_to_num(input_batch["actions"], nan=0.0, posinf=0.0, neginf=0.0)
        
        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)
        
    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        B = batch['actions'].shape[0]

        
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(MDT, self).train_on_batch(batch, epoch, validate=validate)
            actions = batch['actions']
            
            # encode obs
            inputs = {
                'obs': batch["obs"],
            }
            for k in self.obs_shapes:
                ## Shape assertion does not apply to list of strings for raw language
                if "raw" in k:
                    continue
                # first two dimensions should be [B, T] for inputs
                assert inputs['obs'][k].ndim - 2 == len(self.obs_shapes[k])
            
            obs_features = TensorUtils.time_distributed({"obs":inputs["obs"]}, self.nets['policy']['obs_encoder'], inputs_as_kwargs=True)
            # assert obs_features.ndim == 3  # [B, T, D]
            # obs_cond = obs_features.flatten(start_dim=1)
            # obs_cond = obs_features
            obs_cond = []
            obs_cond.append(obs_features['lang'][:, 0, :].unsqueeze(1))
            for key in obs_features:
                if key == 'lang':
                    continue
                obs_cond.append(obs_features[key].flatten(start_dim=1, end_dim=2))
            # import pdb; pdb.set_trace()
            obs_cond = torch.cat(obs_cond, dim=1)
            num_noise_samples = self.algo_config.noise_samples

            # sample noise to add to actions
            noise = torch.randn([num_noise_samples] + list(actions.shape), device=self.device)
            
            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (B,), device=self.device
            ).long()
            
            # add noise to the clean actions according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = torch.cat([self.noise_scheduler.add_noise(
                            actions, noise[i], timesteps)
                            for i in range(len(noise))], dim=0)

            # obs_cond = obs_cond.repeat(num_noise_samples, 1)
            obs_cond =obs_cond.repeat(num_noise_samples, 1, 1)
            timesteps = timesteps.repeat(num_noise_samples)
            # timesteps = timesteps.unsqueeze(0).repeat(num_noise_samples, self.algo_config.horizon.obs_horizon)
            
            # predict the noise residual
            noise_pred = self.nets['policy']['noise_pred_net'](
                noisy_actions, timesteps, global_cond=obs_cond)
            
            # L2 loss
            noise = noise.view(noise.size(0) * noise.size(1), *noise.size()[2:])
            loss = F.mse_loss(noise_pred, noise)
            
            # logging
            losses = {
                'l2_loss': loss
            }
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                # gradient step
                policy_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets,
                    optim=self.optimizers["policy"],
                    loss=loss,
                    accelerator=self.accelerator,
                )
                
                # update Exponential Moving Average of the model weights
                if self.ema is not None:
                    self.ema.step(self.nets)
                
                step_info = {
                    'policy_grad_norms': policy_grad_norms
                }
                info.update(step_info)

        return info
    
    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(MDT, self).log_info(info)
        log["Loss"] = info["losses"]["l2_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log
    
    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        # setup inference queues
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        obs_queue = deque(maxlen=To)
        action_queue = deque(maxlen=Ta)
        self.obs_queue = obs_queue
        self.action_queue = action_queue
        
    def get_action(self, obs_dict, goal_mode=None, eval_mode=False):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation [1, Do]
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor [1, Da]
        """

        # obs_dict: key: [1,D]
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon

        if eval_mode:
            from droid.misc.parameters import hand_camera_id, varied_camera_1_id, varied_camera_2_id
            root_path = os.path.join(os. getcwd(), "eval_params")

            if goal_mode is not None:
                # Read in goal images
                goal_hand_camera_left_image = torch.FloatTensor((cv2.cvtColor(cv2.imread(os.path.join(root_path, f"{hand_camera_id}_left.png")), cv2.COLOR_BGR2RGB) / 255.0)).cuda().permute(2, 0, 1).unsqueeze(0).repeat([1, 1, 1, 1]).unsqueeze(0)
                goal_hand_camera_right_image = torch.FloatTensor((cv2.cvtColor(cv2.imread(os.path.join(root_path, f"{hand_camera_id}_right.png")), cv2.COLOR_BGR2RGB) / 255.0)).cuda().permute(2, 0, 1).unsqueeze(0).repeat([1, 1, 1, 1]).unsqueeze(0)
                goal_varied_camera_1_left_image = torch.FloatTensor((cv2.cvtColor(cv2.imread(os.path.join(root_path, f"{varied_camera_1_id}_left.png")), cv2.COLOR_BGR2RGB) / 255.0)).cuda().permute(2, 0, 1).unsqueeze(0).repeat([1, 1, 1, 1]).unsqueeze(0)
                goal_varied_camera_1_right_image = torch.FloatTensor((cv2.cvtColor(cv2.imread(os.path.join(root_path, f"{varied_camera_1_id}_right.png")), cv2.COLOR_BGR2RGB) / 255.0)).cuda().permute(2, 0, 1).unsqueeze(0).repeat([1, 1, 1, 1]).unsqueeze(0)
                goal_varied_camera_2_left_image = torch.FloatTensor((cv2.cvtColor(cv2.imread(os.path.join(root_path, f"{varied_camera_2_id}_left.png")), cv2.COLOR_BGR2RGB) / 255.0)).cuda().permute(2, 0, 1).unsqueeze(0).repeat([1, 1, 1, 1]).unsqueeze(0)
                goal_varied_camera_2_right_image = torch.FloatTensor((cv2.cvtColor(cv2.imread(os.path.join(root_path, f"{varied_camera_2_id}_right.png")), cv2.COLOR_BGR2RGB) / 255.0)).cuda().permute(2, 0, 1).unsqueeze(0).repeat([1, 1, 1, 1]).unsqueeze(0)

                obs_dict['camera/image/hand_camera_left_image'] = torch.cat([obs_dict['camera/image/hand_camera_left_image'], goal_hand_camera_left_image.repeat(1, To, 1, 1, 1)], dim=2) 
                obs_dict['camera/image/hand_camera_right_image'] = torch.cat([obs_dict['camera/image/hand_camera_right_image'], goal_hand_camera_right_image.repeat(1, To, 1, 1, 1)], dim=2) 
                obs_dict['camera/image/varied_camera_1_left_image'] = torch.cat([obs_dict['camera/image/varied_camera_1_left_image'], goal_varied_camera_1_left_image.repeat(1, To, 1, 1, 1)], dim=2) 
                obs_dict['camera/image/varied_camera_1_right_image'] = torch.cat([obs_dict['camera/image/varied_camera_1_right_image'] , goal_varied_camera_1_right_image.repeat(1, To, 1, 1, 1)], dim=2) 
                obs_dict['camera/image/varied_camera_2_left_image'] = torch.cat([obs_dict['camera/image/varied_camera_2_left_image'] , goal_varied_camera_2_left_image.repeat(1, To, 1, 1, 1)], dim=2) 
                obs_dict['camera/image/varied_camera_2_right_image'] = torch.cat([obs_dict['camera/image/varied_camera_2_right_image'], goal_varied_camera_2_right_image.repeat(1, To, 1, 1, 1)], dim=2) 
            # Note: currently assumes that you are never doing both goal and language conditioning
            else:
                # Reads in current language instruction from file and fills the appropriate obs key, only will
                # actually use it if the policy uses language instructions
                with open(os.path.join(root_path, "lang_command.txt"), 'r') as file:
                    raw_lang = file.read()
                
                # Encode language
                if self.encode_lang is None:
                    model, _, preprocess = open_clip.create_model_and_transforms('EVA02-E-14-plus', pretrained='laion2b_s9b_b144k')
                    model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
                    tokenizer = open_clip.get_tokenizer('EVA02-E-14-plus')
                    encoded_lang = tokenizer(raw_lang)
                    encoded_lang = model.encode_text(encoded_lang)
                    encoded_lang = encoded_lang.unsqueeze(0).repeat(1, 2, 1)
                    self.encode_lang = encoded_lang
                    del model

                obs_dict["lang_fixed/language_distilbert"] = self.encode_lang.type(torch.float32).to('cuda')

        ###############################

        # TODO: obs_queue already handled by frame_stack
        # make sure we have at least To observations in obs_queue
        # if not enough, repeat
        # if already full, append one to the obs_queue
        # n_repeats = max(To - len(self.obs_queue), 1)
        # self.obs_queue.extend([obs_dict] * n_repeats)
        
        if len(self.action_queue) == 0:
            # no actions left, run inference
            action_sequence = self._get_action_trajectory(obs_dict=obs_dict)
            
            # put actions into the queue
            self.action_queue.extend(action_sequence[0])
        
        # has action, execute from left to right
        # [Da]
        action = self.action_queue.popleft()
        
        # [1,Da]
        action = action.unsqueeze(0)
        return action
        
    def _get_action_trajectory(self, obs_dict):
        assert not self.nets.training
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        if self.algo_config.ddpm.enabled is True:
            num_inference_timesteps = self.algo_config.ddpm.num_inference_timesteps
        elif self.algo_config.ddim.enabled is True:
            num_inference_timesteps = self.algo_config.ddim.num_inference_timesteps
        else:
            raise ValueError
        
        # select network
        nets = self.nets
        if self.ema is not None:
            nets = self.ema.averaged_model
        
        # encode obs
        inputs = {
            'obs': obs_dict,
        }
        for k in self.obs_shapes:
            ## Shape assertion does not apply to list of strings for raw language
            if "raw" in k:
                continue
            # first two dimensions should be [B, T] for inputs
            assert inputs['obs'][k].ndim - 2 == len(self.obs_shapes[k])
        obs_features = TensorUtils.time_distributed({"obs":inputs["obs"]}, nets['policy']['obs_encoder'], inputs_as_kwargs=True)
        obs_cond = []
        obs_cond.append(obs_features['lang'][:, 0, :].unsqueeze(1))
        for key in obs_features:
            if key == 'lang':
                continue
            obs_cond.append(obs_features[key].flatten(start_dim=1, end_dim=2))
        obs_cond = torch.cat(obs_cond, dim=1)

        B = obs_cond.shape[0]
        
        # initialize action from Guassian noise
        noisy_action = torch.randn(
            (B, Tp, action_dim), device=self.device)
        naction = noisy_action
        
        # init scheduler
        self.noise_scheduler.set_timesteps(num_inference_timesteps)

        for k in self.noise_scheduler.timesteps:
            # predict noise
            noise_pred = nets['policy']['noise_pred_net'](
                sample=naction, 
                timestep=k,
                global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample

        # process action using Ta
        start = To - 1
        end = start + Ta
        action = naction[:,start:end]
        return action

    def serialize(self):
        """
        Get dictionary of current model parameters.
        """
        return {
            "nets": self.nets.state_dict(),
            "ema": self.ema.averaged_model.state_dict() if self.ema is not None else None,
        }

    def deserialize(self, model_dict):
        """
        Load model from a checkpoint.

        Args:
            model_dict (dict): a dictionary saved by self.serialize() that contains
                the same keys as @self.network_classes
        """
        self.nets.load_state_dict(model_dict["nets"])
        if model_dict.get("ema", None) is not None and self.ema is not None:
            self.ema.averaged_model.load_state_dict(model_dict["ema"])

    
            
            

# =================== Vision Encoder Utils =====================
def replace_submodules(
        root_module: nn.Module, 
        predicate: Callable[[nn.Module], bool], 
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    if parse_version(torch.__version__) < parse_version('1.9.0'):
        raise ImportError('This function requires pytorch >= 1.9.0')

    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module, 
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group, 
            num_channels=x.num_features)
    )
    return root_module

# =================== UNet for Diffusion ==============
import warnings
import torch
from torch.nn import Linear
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
from torch.nn import Module
from torch.nn import functional as F




class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class MDTTransformer(nn.Module):
    def __init__(self, 
        input_dim,
        diffusion_step_embed_dim=1024,
        embed_dim=1024,
        n_heads = 16,
        attn_pdrop = 0.3,
        resid_pdrop = 0.1,
        mlp_pdrop = 0.05,
        n_enc_layers = 18,
        n_dec_layers = 24,
        block_size = 2 * 2 + 1 * 2 + 16 + 1,
        mlp_size = 2048,
        ):
        """
        input_dim: Dim of actions.
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        embed_dim: Size of the embedding for the transformer, all observation tokens and action tokens are embedded to this size.
        n_heads: Number of attention heads
        attn_pdrop: Dropout probability for attention
        resid_pdrop: Dropout probability for residual connections
        mlp_pdrop: Dropout probability for MLP
        n_enc_layers: Number of encoder layers
        n_dec_layers: Number of decoder layers
        block_size: Size of the input block, this is the number of tokens in the input sequence.
        mlp_size: Size of the MLP in the transformer
        """

        super().__init__()

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        
        self.action_emb = nn.Linear(10, embed_dim)
        
        self.action_pred = nn.Sequential(
            nn.Linear(embed_dim, mlp_size),
            nn.GELU(),
            nn.Linear(mlp_size, input_dim),
        )
        
        self.encoder = TransformerEncoder(
            embed_dim=embed_dim,
            n_heads=n_heads,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            n_layers=n_enc_layers,
            block_size=block_size,
            bias=False,
            use_rot_embed=False,
            rotary_xpos=False,
            mlp_pdrop=mlp_pdrop,)
        self.decoder = TransformerFiLMDecoder(
            embed_dim=embed_dim,
            n_heads=n_heads,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            n_layers=n_dec_layers,
            film_cond_dim=embed_dim,
            block_size=block_size,
            bias=False,
            use_rot_embed=False,
            rotary_xpos=False,
            mlp_pdrop=mlp_pdrop,
            use_cross_attention=False,
        )
        
        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            global_cond=None):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        sample = self.action_emb(sample)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0]).unsqueeze(1)
        context = self.encoder(global_cond)
        emb_t = self.diffusion_step_encoder(timesteps)
        context = torch.mean(context, dim=1)
        context = context.unsqueeze(1)
        cond = context + emb_t
        
        pred_sample = self.action_pred(self.decoder(sample, cond))
        
        return pred_sample