from __future__ import annotations

from omegaconf import OmegaConf

import numpy as np
import torch
from copy import deepcopy
from tqdm import tqdm

from verl import DataProto
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from .ray_trainer import (
    compute_response_mask,
    apply_kl_penalty,
    compute_advantage,
    AdvantageEstimator,
    pad_dataproto_to_divisor,
    unpad_dataproto,
)
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from .ray_trainer import RayPPOTrainer
from .utils import Role
from verl.single_controller.ray import RayClassWithInitArgs


def compute_ref_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["ref_responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["ref_attention_mask"]
    return attention_mask[:, -response_length:]


class RayPPOTrainerCoreward(RayPPOTrainer):
    """
    A trainer variant that co-locates actor, rollout and ref in the SAME worker
    by spawning a single ActorRolloutRefWorker with role="actor_rollout_ref".

    - No separate RefPolicy worker group is created.
    - This enables the worker to initialize both rollout and ref models so that
      the rollout sharding manager can switch weights between actor/ref for rollout.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Force co-location mode for this trainer
        self._coreward_co_locate_ref: bool = True

        # Make downstream logic use the in-actor ref path
        self.ref_in_actor = True
        
        # Initialize sliding average parameters from config
        sliding_config = self.config.actor_rollout_ref.ref.get("sliding_average", {})
        self.sliding_average_enable = sliding_config.get("enable", False)
        self.sliding_update_interval = sliding_config.get("update_interval", 30)
        self.sliding_alpha = sliding_config.get("alpha", 0.0)
        self.sliding_alpha_start = sliding_config.get("alpha_start", 0.3)
        self.sliding_alpha_end = sliding_config.get("alpha_end", 0.05)
        if self.sliding_average_enable:
            print(f"sliding: update_interval={self.sliding_update_interval}, alpha_start={self.sliding_alpha_start}, alpha_end={self.sliding_alpha_end}")

        if hasattr(self.reward_fn, "set_temp_schedule"):
            reward_config = self.config.reward_model
            self.reward_temp_start = reward_config.reward_kwargs.temp_start
            self.reward_temp_end = reward_config.reward_kwargs.temp_end
            self.k_pseudo = reward_config.reward_kwargs.k_pseudo
            print(f"Initialize reward manager: temp_start={self.reward_temp_start}, temp_end={self.reward_temp_end}, k_pseudo={self.k_pseudo}")
        
    def init_workers(self):
        """Initialize distributed training workers with co-located ref rollout.

        Spawns only one worker group for actor+rollout+ref using role="actor_rollout_ref".
        Critic and RM creation remain unchanged.
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor+rollout(+ref) in one WG
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout_ref",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = OmegaConf.to_container(self.config.critic, resolve=True)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # Do NOT create separate RefPolicy WG since it is co-located in actor_rollout

        # create a reward model if reward_fn is None
        if self.use_rm:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup(s)
        all_wg = {}
        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        from verl.single_controller.ray.base import create_colocated_worker_cls

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # Initialize the co-located actor+rollout+ref last (so inference gets a better KV-estimate)
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()
        
        self.async_rollout_mode = False

    def fit(self):
        """
        Training loop with optional ref rollout. This overrides the parent to
        insert a ref-model rollout step and union its outputs when enabled.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            from pprint import pprint as _p

            _p(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            from verl.utils.rollout_skip import RolloutSkip as _RolloutSkip

            rollout_skip = _RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        # 设置计算reward的参数
        if hasattr(self.reward_fn, "set_temp_schedule"):
            print("Setting reward manager")
            self.reward_fn.set_temp_schedule(
                temp_start=self.reward_temp_start,
                temp_end=self.reward_temp_end,
                total_steps=self.total_training_steps
            )

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                if self.config.data.use_aug:
                    batch: DataProto = DataProto.from_single_dict(batch_dict["ori"])
                    batch_aug: DataProto = DataProto.from_single_dict(batch_dict["aug"])
                    print("batch_aug keys", batch_aug.batch.keys(), batch_aug.non_tensor_batch.keys())
                else:
                    batch: DataProto = DataProto.from_single_dict(batch_dict)
                    batch_aug = None
                batch.non_tensor_batch["uid"] = np.array(
                    [str(__import__("uuid").uuid4()) for _ in range(len(batch.batch))], dtype=object
                )
                if batch_aug is not None:
                    batch_aug.non_tensor_batch["uid"] = batch.non_tensor_batch["uid"]

                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                if batch_aug is not None:
                    gen_batch_aug = self._get_gen_batch(batch_aug)
                    gen_batch_aug.meta_info["global_steps"] = self.global_steps
                    gen_batch_aug = gen_batch_aug.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # policy rollout
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    # Check if we need to update ref model with sliding average
                    # Use modulo operation to check if it's time to update, but skip when global_steps = 0
                    should_update_ref = (self.sliding_average_enable and 
                                       self.global_steps > 0 and 
                                       self.global_steps % self.sliding_update_interval == 0)

                    # optional ref rollout
                    ref_gen_batch_output = None
                    with marked_timer("gen_ref", timing_raw, color="green"):
                        if batch_aug is not None:
                            # 将滑动平均参数放在meta_info中传递
                            gen_batch_aug.meta_info["update_ref_model"] = should_update_ref
                            gen_batch_aug.meta_info["sliding_alpha_start"] = self.sliding_alpha_start
                            gen_batch_aug.meta_info["sliding_alpha_end"] = self.sliding_alpha_end
                            gen_batch_aug.meta_info["total_steps"] = self.total_training_steps
                            gen_batch_aug.meta_info["current_step"] = self.global_steps
                            
                            ref_gen_batch_output = self.actor_rollout_wg.generate_ref_sequences(gen_batch_aug)
                        else:
                            # 将滑动平均参数放在meta_info中传递
                            gen_batch.meta_info["update_ref_model"] = should_update_ref
                            gen_batch.meta_info["sliding_alpha_start"] = self.sliding_alpha_start
                            gen_batch.meta_info["sliding_alpha_end"] = self.sliding_alpha_end
                            gen_batch.meta_info["total_steps"] = self.total_training_steps
                            gen_batch.meta_info["current_step"] = self.global_steps
                            
                            ref_gen_batch_output = self.actor_rollout_wg.generate_ref_sequences(gen_batch)
                        timing_raw.update(ref_gen_batch_output.meta_info["timing"])
                        ref_gen_batch_output.meta_info.pop("timing", None)

                    # merge generated outputs
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)
                    # Prefix all ref rollout keys with ref_ and merge into batch to avoid collisions
                    if ref_gen_batch_output is not None:
                        ref_tensors = {f"ref_{k}": v for k, v in ref_gen_batch_output.batch.items()}
                        if len(ref_tensors) > 0:
                            ref_prefixed = DataProto.from_dict(tensors=ref_tensors)
                            batch = batch.union(ref_prefixed)
                        if "ref_response_mask" not in batch.batch.keys():
                            batch.batch["ref_response_mask"] = compute_ref_response_mask(batch)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # reward
                    with marked_timer("reward", timing_raw, color="yellow"):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        else:
                            if hasattr(self.reward_fn, "update_temp"):
                                self.reward_fn.update_temp(self.global_steps)
                                print(f"Updating reward temp={self.reward_fn.freq_temp}")
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                            if "metrics" in reward_extra_infos_dict.keys():
                                metrics.update(reward_extra_infos_dict["metrics"])
                                reward_extra_infos_dict.pop("metrics")

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                                                
                        from .ray_trainer import agg_loss as _agg_loss

                        entropy_agg = _agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                    # reference log_prob
                    if self.use_reference_policy:
                        with marked_timer("ref", timing_raw, color="olive"):
                            # co-located path
                            ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    # advantage
                    with marked_timer("adv", timing_raw, color="brown"):
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = __import__("ray").get(future_reward)  # type: ignore
                        batch.batch["token_level_scores"] = reward_tensor
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # critic update
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # actor update（支持 warmup）
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)
                    
                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            sample_gts = [
                                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                                for item in batch
                            ]

                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )

                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                gts=sample_gts,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )
                            
                    ref_rollout_data_dir = self.config.trainer.get("ref_rollout_data_dir", None)
                    if ref_rollout_data_dir:
                        with marked_timer("dump_ref_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["ref_prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["ref_responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            sample_gts = [
                                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                                for item in batch
                            ]

                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )

                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                gts=sample_gts,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=ref_rollout_data_dir,
                            )                    
                
                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics) 
                    
                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()            
    
                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                metrics.update({"training/global_step": self.global_steps, "training/epoch": epoch})

                from .ray_trainer import compute_data_metrics, compute_timing_metrics, compute_throughout_metrics

                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                from verl.experimental.dataset.sampler import AbstractCurriculumSampler

                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    from pprint import pprint as _p

                    _p(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                if hasattr(self.train_dataset, "on_batch_end"):
                    self.train_dataset.on_batch_end(batch=batch)


