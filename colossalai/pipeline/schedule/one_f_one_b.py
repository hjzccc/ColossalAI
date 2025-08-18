from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import torch
from torch.nn import Module
from torch.utils._pytree import tree_map
import os
from colossalai.accelerator import get_accelerator
from colossalai.interface import ModelWrapper, OptimizerWrapper
from colossalai.pipeline.p2p import PipelineP2PCommunication, create_send_metadata
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.quantization.fp8 import cast_from_fp8_pipeline, cast_to_fp8_pipeline
from colossalai.utils import get_current_device, global_step_counter

from ._utils import (
    detach,
    get_batch_size,
    get_micro_batch,
    merge_batch,
    model_forward,
    retain_grad,
    to_device,
    tree_map_hf,
)
from .base import PipelineSchedule

import torch.distributed as dist

class OneForwardOneBackwardSchedule(PipelineSchedule):
    def __init__(
        self,
        stage_manager: PipelineStageManager,
        num_microbatches: Optional[int] = None,
        microbatch_size: Optional[int] = None,
        enable_metadata_cache: bool = True,
        fp8_communication: bool = False,
    ) -> None:
        """1F1B pipeline schedule.

        Args:
            stage_manager (PipelineStageManager): Pipeline stage manager
            num_microbatches (Optional[int], optional): The number of microbatches. If not provided, it will be derived from microbatch size. Defaults to None.
            microbatch_size (Optional[int], optional): Microbatch size. If num_microbatches is provided, this will be ignored. Defaults to None.
        """
        super().__init__(stage_manager)
        assert (
            num_microbatches is not None or microbatch_size is not None
        ), "Either num_microbatches or microbatch_size should be provided"

        self.comm = PipelineP2PCommunication(stage_manager, overlap_p2p=False)

        self.num_microbatches = num_microbatches
        self.microbatch_size = microbatch_size
        self.batch: Optional[Any] = None
        self.batch_size: Optional[int] = None
        self.last_batch_size: Optional[int] = None
        self.microbatch_offset: Optional[int] = None

        # P2PMeta cache
        self.enable_metadata_cache = enable_metadata_cache
        self.send_tensor_metadata = True
        self.send_grad_metadata = True
        self.tensor_metadata_recv = None
        self.grad_metadata_recv = None

        self.fp8_communication = fp8_communication
    def _merge_tensors(self, tensor_list):
        """Merge list of tensors/dicts along batch dimension."""
        if not tensor_list or tensor_list[0] is None:
            return None
        if isinstance(tensor_list[0], dict):
            merged = {}
            for key in tensor_list[0].keys():
                if isinstance(tensor_list[0][key], torch.Tensor):
                    merged[key] = torch.cat([t[key] for t in tensor_list], dim=0)
                else:
                    merged[key] = tensor_list[0][key]
            return merged
        elif isinstance(tensor_list[0], torch.Tensor):
            return torch.cat(tensor_list, dim=0)
        return tensor_list[0]
    def _run_exact_merged_recovery(
        self,
        model: Module,
        criterion: Callable[..., Any],
        optimizer: OptimizerWrapper,
        checkpoint_data: dict,
        num_warmup_microbatches: int,
        num_microbatches_remaining: int,
        return_loss: bool,
        return_outputs: bool,
    ) -> Dict:
        """Run EXACT recovery with merged batches using saved microbatches."""
        print(f"\n[{dist.get_rank()}] ===== EXACT MERGED RECOVERY =====")
        print(f"  Stage: {self.stage_manager.stage}")
        print(f"  Warmup: {num_warmup_microbatches}, Steady: {num_microbatches_remaining}")
        # Monitor memory
        initial_mem = torch.cuda.memory_allocated() / 1e9
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Initial memory: {initial_mem:.2f}GB / {total_mem:.2f}GB")

        accum_loss = None
        if return_loss and self.stage_manager.is_last_stage():
            accum_loss = torch.scalar_tensor(0, device=get_current_device())
        outputs = [] if return_outputs and self.stage_manager.is_last_stage() else None

        # Load saved data
        intermediate = checkpoint_data['intermediate']
        saved_micro_batches = checkpoint_data['micro_batches']

        print(f"  Loaded {len(saved_micro_batches)} microbatches")
        print(f"  Loaded {len(intermediate)} intermediate tensors")

        # Parse intermediate list based on EXACT order from run_forward_backward
        idx = 0

        # 1. Warmup forward inputs
        warmup_inputs = []
        for i in range(num_warmup_microbatches):
            if idx < len(intermediate):
                warmup_inputs.append(intermediate[idx])
                idx += 1

        # 2. Steady state first input (if exists)
        steady_first_input = None
        if num_microbatches_remaining > 0 and idx < len(intermediate):
            steady_first_input = intermediate[idx]
            idx += 1

        # 3. Steady state: INTERLEAVED gradients and inputs
        steady_grads = []
        steady_remaining_inputs = []

        for i in range(num_microbatches_remaining):
            # First comes the gradient
            if idx < len(intermediate):
                steady_grads.append(intermediate[idx])
                idx += 1

            # Then comes the input (except for last iteration)
            if i < num_microbatches_remaining - 1:  # Not last iteration
                if idx < len(intermediate):
                    steady_remaining_inputs.append(intermediate[idx])
                    idx += 1

        # 4. Cooldown gradients
        cooldown_grads = []
        for i in range(num_warmup_microbatches):
            if idx < len(intermediate):
                cooldown_grads.append(intermediate[idx])
                idx += 1

        print(f"  Parsed: {len(warmup_inputs)} warmup inputs, "
                f"{1 if steady_first_input else 0} steady first, "
                f"{len(steady_grads)} steady grads, "
                f"{len(steady_remaining_inputs)} steady remaining inputs, "
                f"{len(cooldown_grads)} cooldown grads")
        print(f"  Total parsed: {idx}, Remaining: {len(intermediate) - idx}")

        # Build ordered list of ALL input activations for forward pass
        all_input_activations = []

        # Add warmup inputs
        all_input_activations.extend(warmup_inputs)

        # Add steady state inputs in correct order
        if num_microbatches_remaining > 0:
            all_input_activations.append(steady_first_input)
            all_input_activations.extend(steady_remaining_inputs)

        # Build ordered list of ALL gradients for backward pass
        all_gradients = steady_grads + cooldown_grads

        print(f"  Total: {len(all_input_activations)} input activations, {len(all_gradients)} gradients")
        print(f"  Expected: {self.num_microbatches} activations, {self.num_microbatches} gradients")

        # Merge tensors for batch processing
        merged_micro_batch = self._merge_tensors(saved_micro_batches)
        merged_input = self._merge_tensors(all_input_activations) if not self.stage_manager.is_first_stage() else None
        merged_grad = self._merge_tensors(all_gradients) if all_gradients else None

        # Debug shapes
        if merged_micro_batch:
            if isinstance(merged_micro_batch, dict):
                for k, v in merged_micro_batch.items():
                    if isinstance(v, torch.Tensor):
                        print(f"  Merged batch '{k}': shape={v.shape}")
                        break
            elif isinstance(merged_micro_batch, torch.Tensor):
                print(f"  Merged batch: shape={merged_micro_batch.shape}")

        if merged_input:
            if isinstance(merged_input, dict):
                for k, v in merged_input.items():
                    if isinstance(v, torch.Tensor):
                        print(f"  Merged input '{k}': shape={v.shape}")
                        break
            elif isinstance(merged_input, torch.Tensor):
                print(f"  Merged input: shape={merged_input.shape}")

        if merged_grad:
            if isinstance(merged_grad, dict):
                for k, v in merged_grad.items():
                    if isinstance(v, torch.Tensor):
                        print(f"  Merged grad '{k}': shape={v.shape}")
                        break
            elif isinstance(merged_grad, torch.Tensor):
                print(f"  Merged grad: shape={merged_grad.shape}")

        # FORWARD PASS
        print(f"\n[{dist.get_rank()}] Running merged forward pass...")
        mem_before_fwd = torch.cuda.memory_allocated() / 1e9

        output_obj = model_forward(model, merged_micro_batch, merged_input)

        mem_after_fwd = torch.cuda.memory_allocated() / 1e9
        print(f"  Forward memory: {mem_before_fwd:.2f}GB -> {mem_after_fwd:.2f}GB")

        # BACKWARD PASS
        print(f"\n[{dist.get_rank()}] Running merged backward pass...")
        mem_before_bwd = torch.cuda.memory_allocated() / 1e9

        if merged_input is not None:
            tree_map(retain_grad, merged_input)

        # optimizer.zero_grad()

        if self.stage_manager.is_last_stage():
            loss = criterion(output_obj, merged_micro_batch)
            optimizer.backward(loss)
            if accum_loss is not None:
                accum_loss.add_(loss.data)
        else:
            if merged_grad is not None:
                if isinstance(output_obj, dict) and isinstance(merged_grad, dict):
                    keys = output_obj.get("backward_tensor_keys",
                                        [k for k in output_obj.keys() if k in merged_grad])
                    tensors_to_backward = []
                    grads_to_backward = []
                    for k in keys:
                        if k in output_obj and k in merged_grad:
                            tensors_to_backward.append(output_obj[k])
                            grads_to_backward.append(merged_grad[k])

                    if len(tensors_to_backward) == 1:
                        optimizer.backward_by_grad(tensors_to_backward[0], grads_to_backward[0])
                    elif len(tensors_to_backward) > 0:
                        optimizer.backward_by_grad(tensors_to_backward, grads_to_backward)
                elif isinstance(output_obj, torch.Tensor) and isinstance(merged_grad, torch.Tensor):
                    optimizer.backward_by_grad(output_obj, merged_grad)

        mem_after_bwd = torch.cuda.memory_allocated() / 1e9
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Backward memory: {mem_before_bwd:.2f}GB -> {mem_after_bwd:.2f}GB")
        print(f"  Peak memory: {peak_mem:.2f}GB / {total_mem:.2f}GB ({peak_mem/total_mem*100:.1f}%)")

        # Handle outputs if needed
        if outputs is not None:
            if isinstance(output_obj, dict):
                batch_size = next(v.shape[0] for v in output_obj.values() if isinstance(v, torch.Tensor))
            else:
                batch_size = output_obj.shape[0] if isinstance(output_obj, torch.Tensor) else self.num_microbatches

            mb_size = batch_size // self.num_microbatches
            for i in range(self.num_microbatches):
                if isinstance(output_obj, dict):
                    mb_output = {k: v[i*mb_size:(i+1)*mb_size] if isinstance(v, torch.Tensor) else v
                                for k, v in output_obj.items()}
                else:
                    mb_output = output_obj[i*mb_size:(i+1)*mb_size] if isinstance(output_obj, torch.Tensor) else output_obj
                outputs.append(tree_map_hf(detach, mb_output))

            if isinstance(model, ModelWrapper):
                model = model.unwrap()
            batch_size_dim = getattr(model, "batch_size_dim", 0)
            outputs = merge_batch(outputs, batch_size_dim)

        print(f"[{dist.get_rank()}] Exact merged recovery completed!\n")
        global_step_counter.increment()
        return {"loss": accum_loss, "outputs": outputs}


    def load_batch(self, data_iter: Iterable, device: Optional[torch.device] = None) -> None:
        """Load a batch from data iterator.

        Args:
            data_iter (Iterable): Data iterator.
            device (Optional[torch.device], optional): Target device. Defaults to None.
        """
        batch = next(data_iter)
        if device is not None:
            batch = tree_map(partial(to_device, device=device), batch)

        self.microbatch_offset = 0
        self.batch = batch
        self.batch_size = get_batch_size(batch)

        if self.microbatch_size is None:
            assert self.batch_size % self.num_microbatches == 0, "Batch size should divided by # microbatches"
            self.microbatch_size = self.batch_size // self.num_microbatches
        if self.num_microbatches is None:
            assert self.batch_size % self.microbatch_size == 0, "Batch size should divided by the microbatch size"
            self.num_microbatches = self.batch_size // self.microbatch_size

        if not self.forward_only:
            assert self.last_batch_size is None or self.last_batch_size == self.batch_size
            assert self.batch_size == self.microbatch_size * self.num_microbatches

            assert (
                self.num_microbatches >= self.stage_manager.num_stages
            ), "Number of microbatch should be larger than number of stages"

        if self.forward_only:
            self.num_microbatches = (self.batch_size - 1) // self.microbatch_size + 1
            # NOTE: disable metadata cache when batch size changes (not valid anymore)
            if self.batch_size != self.last_batch_size:
                self.enable_metadata_cache = False
                self.send_tensor_metadata = True
                self.send_grad_metadata = True
                self.tensor_metadata_recv = None
                self.grad_metadata_recv = None

        self.last_batch_size = self.batch_size

    def load_micro_batch(self) -> Any:
        """Load a micro batch from the current batch.

        Returns:
            Any: Micro batch.
        """
        assert self.microbatch_offset <= self.batch_size, "Microbatches exhausted"
        micro_batch = get_micro_batch(self.batch, self.microbatch_offset, self.microbatch_size)
        self.microbatch_offset += self.microbatch_size
        return tree_map(partial(to_device, device=get_accelerator().get_current_device()), micro_batch)

    def recv_forward(self, prev_rank: int = None) -> Any:
        """Copy the forward output from the previous stage in pipeline as the input tensor of this stage.
           For 1F1B.

        Args:
            prev_rank (int, optional): The rank of the source of the tensor.

        Returns:
            Any: The input tensor or input tensor list.
        """
        if not self.stage_manager.is_first_stage():
            input_tensor, _ = self.comm.recv_forward(prev_rank, metadata_recv=self.tensor_metadata_recv)
            if self.enable_metadata_cache and self.tensor_metadata_recv is None:
                self.tensor_metadata_recv = create_send_metadata(input_tensor)

            if self.fp8_communication:
                cast_from_fp8_pipeline(input_tensor)
            return input_tensor

    def recv_backward(self, next_rank: int = None) -> Any:
        """Copy the gradient tensor from the next stage in pipeline as the input gradient of this stage.
           For 1F1B.

        Args:
            next_rank (int, optional): The rank of the source of the tensor.

        Returns:
            Any: The input gradient tensor or gradient tensor list.
        """
        if not self.stage_manager.is_last_stage():
            output_tensor_grad, _ = self.comm.recv_backward(next_rank, metadata_recv=self.grad_metadata_recv)
            if self.fp8_communication:
                cast_from_fp8_pipeline(output_tensor_grad)
            if self.enable_metadata_cache and self.grad_metadata_recv is None:
                self.grad_metadata_recv = create_send_metadata(output_tensor_grad)

            return output_tensor_grad

    def send_forward(self, output_tensor: Any, next_rank: int = None) -> None:
        """Sends the input tensor to the next stage in pipeline.
           For 1F1B.

        Args:
            output_object (Any): Object to be sent.
            next_rank (int, optional): The rank of the recipient of the tensor.
        """
        # if not self.stage_manager.is_last_stage():
        #   # ADD THIS: Print activations before sending
        #   print(f"\nStage {self.stage_manager.stage} sending activations:")
        #   if isinstance(output_tensor, dict):
        #       for k, v in output_tensor.items():
        #           if isinstance(v, torch.Tensor):
        #               print(f"  {k}: shape={v.shape}, mean={v.mean().item():.6f}")
        #   elif isinstance(output_tensor, torch.Tensor):
        #       print(f"  shape={output_tensor.shape}, mean={output_tensor.mean().item():.6f}")

        if not self.stage_manager.is_last_stage():
            if self.fp8_communication:
                cast_to_fp8_pipeline(output_tensor)
            self.comm.send_forward(output_tensor, next_rank, send_metadata=self.send_tensor_metadata)
            self.send_tensor_metadata = not self.enable_metadata_cache

            if self.fp8_communication:
                cast_from_fp8_pipeline(output_tensor, del_metadata=False)

    def send_backward(self, input_tensor_grad: Any, prev_rank: int = None) -> None:
        """Sends the gradient tensor to the previous stage in pipeline.
           For 1F1B.

        Args:
            input_object (Any): Object to be sent.
            prev_rank (int, optional): The rank of the recipient of the tensor
        """
        if not self.stage_manager.is_first_stage():
            if self.fp8_communication:
                cast_to_fp8_pipeline(input_tensor_grad)
            self.comm.send_backward(input_tensor_grad, prev_rank, send_metadata=self.send_grad_metadata)
            self.send_grad_metadata = not self.enable_metadata_cache
            if self.fp8_communication:
                cast_from_fp8_pipeline(input_tensor_grad, del_metadata=False)

    def send_forward_recv_backward(self, output_tensor: Any, send_first: Optional[bool] = None) -> Any:
        """Sends the input tensor to the next stage and copy the gradient tensor from the next stage in pipeline.
           For 1F1B.

        Args:
            output_object (Any): Object to be sent.
            next_rank (int, optional): The rank of the recipient of the tensor.
        """
        if not self.stage_manager.is_last_stage():
            # print(f"\nStage {self.stage_manager.stage} sending activations:")
            # if isinstance(output_tensor, dict):
            #     for k, v in output_tensor.items():
            #         if isinstance(v, torch.Tensor):
            #             print(f"  {k}: shape={v.shape}, mean={v.mean().item():.6f}")
            # elif isinstance(output_tensor, torch.Tensor):
            #     print(f"  shape={output_tensor.shape}, mean={output_tensor.mean().item():.6f}")
            if not self.send_tensor_metadata and self.grad_metadata_recv is not None:
                send_first = None
            if self.fp8_communication:
                cast_to_fp8_pipeline(output_tensor)
            output_tensor_grad, _ = self.comm.send_forward_recv_backward(
                output_tensor,
                send_metadata=self.send_tensor_metadata,
                metadata_recv=self.grad_metadata_recv,
                send_first=send_first,
            )
            self.send_tensor_metadata = not self.enable_metadata_cache
            if self.enable_metadata_cache and self.grad_metadata_recv is None:
                self.grad_metadata_recv = create_send_metadata(output_tensor_grad)
            if self.fp8_communication:
                cast_from_fp8_pipeline(output_tensor, del_metadata=False)
                cast_from_fp8_pipeline(output_tensor_grad)

            # # ADD THIS: Print received gradients
            # print(f"\nStage {self.stage_manager.stage} received gradients:")
            # if isinstance(output_tensor_grad, dict):
            #     for k, v in output_tensor_grad.items():
            #         if isinstance(v, torch.Tensor):
            #             print(f"  {k}: shape={v.shape}, mean={v.mean().item():.6f}")
            # elif isinstance(output_tensor_grad, torch.Tensor):
            #     print(f"  shape={output_tensor_grad.shape}, mean={output_tensor_grad.mean().item():.6f}")
            return output_tensor_grad

    def send_backward_recv_forward(self, input_tensor_grad: Any, send_first: Optional[bool] = None) -> Any:
        """Sends the gradient tensor to the previous stage and copy the input tensor from the previous stage in pipeline.
           For 1F1B.

        Args:
            output_object (Any): Object to be sent.
            prev_rank (int, optional): The rank of the recipient of the tensor.
        """
        if not self.stage_manager.is_first_stage():
            if not self.send_grad_metadata and self.tensor_metadata_recv is not None:
                send_first = None  # must not fallback
            if self.fp8_communication:
                cast_to_fp8_pipeline(input_tensor_grad)
            input_tensor, _ = self.comm.send_backward_recv_forward(
                input_tensor_grad,
                send_metadata=self.send_grad_metadata,
                metadata_recv=self.tensor_metadata_recv,
                send_first=send_first,
            )
            self.send_grad_metadata = not self.enable_metadata_cache
            if self.enable_metadata_cache and self.tensor_metadata_recv is None:
                self.tensor_metadata_recv = create_send_metadata(input_tensor)
            if self.fp8_communication:
                cast_from_fp8_pipeline(input_tensor)
                cast_from_fp8_pipeline(input_tensor_grad, del_metadata=False)

            return input_tensor

    def forward_step(
        self,
        model: Module,
        input_obj: Optional[dict],
        criterion: Callable,
        accum_loss: Optional[torch.Tensor] = None,
        outputs: Optional[List[Any]] = None,
        return_micro_batch: bool = False,  # Add this parameter
    ) -> Union[torch.Tensor, dict]:
        """Forward one step of the pipeline

        Args:
            model (Module): Model to be run
            input_obj (Optional[dict]): The output from the previous stage. If it is the first stage, the `input_obj` is None.
            criterion (Callable): Criterion to calculate loss.
            accum_loss (Optional[torch.Tensor], optional): Accumulated loss. Defaults to None.
            outputs (Optional[List[Any]], optional): List to store the output of the last stage (final output). Defaults to None.

        Returns:
            Union[torch.Tensor, dict]: The intermediate output (dict) of the current stage. If it is the last stage, the output is the loss (Tensor).
        """
        micro_batch = self.load_micro_batch()
        # for the first stage, input_obj is None
        # for the non-first stage, input_obj is the output of the previous stage and it's must be a dict
        output_obj = model_forward(model, micro_batch, input_obj)
        if self.stage_manager.is_last_stage():
            loss = criterion(output_obj, micro_batch) / self.num_microbatches

            if accum_loss is not None:
                accum_loss.add_(loss.data)
            if outputs is not None:
                outputs.append(tree_map_hf(detach, output_obj))
            return (loss, micro_batch) if return_micro_batch else loss
        else:
            return (output_obj, micro_batch) if return_micro_batch else output_obj

    def backward_step(
        self,
        optimizer: OptimizerWrapper,
        input_obj: Optional[dict],
        output_obj: Union[dict, torch.Tensor],
        output_obj_grad: Optional[dict],
    ) -> Optional[dict]:
        """Backward one step of the pipeline

        Args:
            optimizer (OptimizerWrapper): Optimizer to update the model
            input_obj (Optional[dict]): Output of the previous stage. If it is the first stage, the `input_obj` is None.
            output_obj (Union[dict, torch.Tensor]): Output of the current stage. If it is the last stage, the output is the loss (Tensor).
            output_obj_grad (dict): Gradient of the `output_obj`. If it is the last stage, the `output_obj_grad` is None.

        Returns:
            Optional[dict]: Gradient of the `input_obj`. If it is the first stage, the `input_obj_grad` is None.
        """

        # Retain the grad on the input_obj.
        tree_map(retain_grad, input_obj)
        # Backward pass.
        if output_obj_grad is None:
            optimizer.backward(output_obj)
        else:
            keys = output_obj.get("backward_tensor_keys", output_obj_grad.keys())
            tensors_to_backward = []
            grads_to_backward = []
            for k in keys:
                tensors_to_backward.append(output_obj[k])
                grads_to_backward.append(output_obj_grad[k])
            if len(tensors_to_backward) == 1:
                optimizer.backward_by_grad(tensors_to_backward[0], grads_to_backward[0])
            else:
                optimizer.backward_by_grad(tensors_to_backward, grads_to_backward)

        # Collect the grad of the input_obj.
        input_obj_grad = None
        if input_obj is not None:
            input_obj_grad = {}
            for k, v in input_obj.items():
                if isinstance(v, torch.Tensor) and v.grad is not None:
                    input_obj_grad[k] = v.grad
        return input_obj_grad

    def run_forward_only(
        self,
        model: Module,
        data_iter: Iterable,
        criterion: Callable[..., Any],
        return_loss: bool = False,
        return_outputs: bool = False,
    ) -> Dict:
        """
        Runs forward only schedule, with communication between pipeline stages.
        """
        assert self.forward_only

        self.load_batch(data_iter)

        accum_loss = None
        if return_loss and self.stage_manager.is_last_stage():
            accum_loss = torch.scalar_tensor(0, device=get_accelerator().get_current_device())
        outputs = [] if return_outputs and self.stage_manager.is_last_stage() else None

        for _ in range(self.num_microbatches):
            input_obj = self.recv_forward()
            output_obj = self.forward_step(model, input_obj, criterion, accum_loss, outputs)
            self.send_forward(output_obj)

        if outputs is not None:
            if isinstance(model, ModelWrapper):
                model = model.unwrap()
            batch_size_dim = getattr(model, "batch_size_dim", 0)
            outputs = merge_batch(outputs, batch_size_dim)
        return {"loss": accum_loss, "outputs": outputs}

    def run_forward_backward(
        self,
        model: Module,
        data_iter: Iterable,
        criterion: Callable[..., Any],
        optimizer: Optional[OptimizerWrapper] = None,
        return_loss: bool = False,
        return_outputs: bool = False,
    ) -> Dict:
        """
        Runs non-interleaved 1F1B schedule, with communication between pipeline stages.
        """

        print(f"\n[{dist.get_rank()}]=========================================")
        print(f"\t ====== Step[{global_step_counter.get()}] ======")

        assert dist.get_rank() == self.stage_manager.stage, "rank != stage ?"
        import time

        assert not self.forward_only

        self.load_batch(data_iter)
        # num_warmup_microbatches is the step when not all the processes are working
        print(self.stage_manager.num_stages, self.stage_manager.stage)
        num_warmup_microbatches = self.stage_manager.num_stages - self.stage_manager.stage - 1
        print(num_warmup_microbatches, self.stage_manager.stage)
        num_warmup_microbatches = min(num_warmup_microbatches, self.num_microbatches)
        print(num_warmup_microbatches, self.stage_manager.stage)
        num_microbatches_remaining = self.num_microbatches - num_warmup_microbatches
        print(num_microbatches_remaining, self.stage_manager.stage)

        # Input, output tensors only need to be saved when doing backward passes
        input_objs, output_objs = [], []

        overriding_intermediates = False
        logging_intermediate = False

        intermediate = []
        saved_micro_batches = []
        
        f_checkpoint = f"checkpoint.step{global_step_counter.get()}.stage{self.stage_manager.stage}.pt"
        if os.path.exists(f_checkpoint):
            print("=====using intermediate tensors=====")
            overriding_intermediates = True
            checkpoint_data = torch.load(f_checkpoint)
            intermediate = checkpoint_data['intermediate']
            # Restore RNG states
            torch.set_rng_state(checkpoint_data['torch_rng_state'])
            torch.cuda.set_rng_state(checkpoint_data['cuda_rng_state'])
            # Check if we have saved microbatches for exact recovery
            if 'micro_batches' in checkpoint_data:
                print(f"=====Found {len(checkpoint_data['micro_batches'])} saved microbatches - EXACT recovery possible=====")
                use_merged_recovery = True  # Set to False for sequential recovery

                if use_merged_recovery:
                    try:
                        return self._run_exact_merged_recovery(
                            model, criterion, optimizer, checkpoint_data,
                            num_warmup_microbatches, num_microbatches_remaining,
                            return_loss, return_outputs
                        )
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"ERROR: Out of memory with merged recovery! Falling back to  sequential.")
                            torch.cuda.empty_cache()
                            # Fall through to sequential recovery
                        else:
                            raise e
            else:
                print("WARNING: No saved microbatches in checkpoint - cannot do exact recovery")
                print("Falling back to sequential recovery with new data (results may differ!)")
        else:
            logging_intermediate = True
            print("=====regular computation=====")

        accum_loss = None
        if return_loss and self.stage_manager.is_last_stage():
            accum_loss = torch.scalar_tensor(0, device=get_current_device())
        outputs = [] if return_outputs and self.stage_manager.is_last_stage() else None

        # Run warmup forward passes.
        for i in range(num_warmup_microbatches):

            if not overriding_intermediates:
                input_obj = self.recv_forward()  # original code

                print_obj(input_obj, f"warmup_{i}", self.recv_forward)
                if logging_intermediate:
                    intermediate.append(input_obj)
            else:
                    input_obj = intermediate.pop(0)

            if logging_intermediate:
                (output_obj, micro_batch) = self.forward_step(model, input_obj, criterion, accum_loss, outputs, return_micro_batch=True)
                if micro_batch is not None:
                    saved_micro_batches.append(micro_batch)
            else:
                output_obj = self.forward_step(model, input_obj, criterion, accum_loss, outputs)

            print_itr(f"warmup_{i}", self.forward_step)
            # time.sleep(1)

            if not overriding_intermediates:
                self.send_forward(output_obj)  # original code

                print_obj(output_obj, f"warmup_{i}", self.send_forward)

            input_objs.append(input_obj)
            output_objs.append(output_obj)

        # Before running 1F1B, need to receive first forward tensor.
        # If all microbatches are run in warmup / cooldown phase, then no need to
        # receive this tensor here.
        if num_microbatches_remaining > 0:
            
            if not overriding_intermediates:
                input_obj = self.recv_forward()
                
                print_obj(input_obj, "warmup_fin", self.recv_forward)      
                if logging_intermediate:
                    intermediate.append(input_obj)
            else:
                    input_obj = intermediate.pop(0)    

        # Run 1F1B in steady state.
        for i in range(num_microbatches_remaining):
            last_iteration = i == (num_microbatches_remaining - 1)
            if logging_intermediate:
                (output_obj, micro_batch) = self.forward_step(model, input_obj, criterion, accum_loss, outputs, return_micro_batch=True)
                if micro_batch is not None:
                    saved_micro_batches.append(micro_batch)
            else:
                output_obj = self.forward_step(model, input_obj, criterion, accum_loss, outputs)

            print_itr(f"steady_{i}", self.forward_step)
            # time.sleep(1)

            if not overriding_intermediates:
                output_obj_grad = self.send_forward_recv_backward(output_obj, send_first=self.stage_manager.stage % 2 == 0)
            
                print_obj(output_obj, f"steady_{i}", self.send_forward_recv_backward)
                print_obj(output_obj_grad, f"steady_{i}", self.send_forward_recv_backward, "(grad)")
                if logging_intermediate:
                    intermediate.append(output_obj_grad)
            else:
                output_obj_grad = intermediate.pop(0)

            # Add input_obj and output_obj to end of list.
            input_objs.append(input_obj)
            output_objs.append(output_obj)

            # Pop output_obj and output_obj from the start of the list for
            # the backward pass.
            input_obj = input_objs.pop(0)
            output_obj = output_objs.pop(0)
            input_obj_grad = self.backward_step(optimizer, input_obj, output_obj, output_obj_grad)

            print_itr(f"steady_{i}", self.backward_step)
            # time.sleep(1)

            if last_iteration:
                if not overriding_intermediates:
                    self.send_backward(input_obj_grad)

                    print_obj(input_obj_grad, f"steady_{i}", self.send_backward)
    
            else:
                if not overriding_intermediates:
                    input_obj = self.send_backward_recv_forward(
                        input_obj_grad, send_first=self.stage_manager.stage % 2 == 0
                    )

                    print_obj(input_obj_grad, f"steady_{i}", self.send_backward_recv_forward, "(grad)")
                    print_obj(input_obj, f"steady_{i}", self.send_backward_recv_forward)
                    if logging_intermediate:
                        intermediate.append(input_obj)
                else:
                    input_obj = intermediate.pop(0)

        # Run cooldown backward passes.
        for i in range(num_warmup_microbatches):
            input_obj = input_objs.pop(0)
            output_obj = output_objs.pop(0)

            if not overriding_intermediates:
                output_obj_grad = self.recv_backward()
            
                print_obj(output_obj_grad, f"cooldown_{i}", self.recv_backward, "(grad)")
                if logging_intermediate:
                    intermediate.append(output_obj_grad)
            else:
                output_obj_grad = intermediate.pop(0)
            
            input_obj_grad = self.backward_step(optimizer, input_obj, output_obj, output_obj_grad)

            print_itr(f"cooldown_{i}", self.backward_step)
            # time.sleep(1)

            if not overriding_intermediates:
                self.send_backward(input_obj_grad)

                print_obj(input_obj_grad, f"cooldown_{i}", self.send_backward, "(grad)")

        assert all(len(v) == 0 for v in input_objs) and all(len(v) == 0 for v in output_objs)

        if outputs is not None:
            if isinstance(model, ModelWrapper):
                model = model.unwrap()
            batch_size_dim = getattr(model, "batch_size_dim", 0)
            outputs = merge_batch(outputs, batch_size_dim)


        print(f"[{dist.get_rank()}]----------------------------------")
        if logging_intermediate:
            # Save both intermediate tensors and RNG states
            checkpoint_data = {
                'intermediate': intermediate,
                "micro_batches": saved_micro_batches,
                'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state()
            }
            torch.save(checkpoint_data, f_checkpoint)
            print(f"  Saved {len(saved_micro_batches)} microbatches to checkpoint")
            
        global_step_counter.increment()
        return {"loss": accum_loss, "outputs": outputs}

    def forward_backward_step(
        self,
        model: Module,
        data_iter: Iterable,
        criterion: Callable[..., Any],
        optimizer: Optional[OptimizerWrapper] = None,
        return_loss: bool = False,
        return_outputs: bool = False,
    ) -> dict:
        """
        Args:
            model (Module): Model to be trained.
            data_iter (Iterable): Data iterator.
            criterion (Callable[[Any, Any], Tensor]): Criterion to be used. It should take two arguments: model outputs and inputs, and returns loss tensor.
            optimizer (OptimizerWrapper, optional): Optimizer to be used. Can be None when only forward is executed. Defaults to None.
            return_loss (bool, optional): Whether to return loss. Defaults to False. Whether to return loss.
            return_outputs (bool, optional): Whether to return model outputs. Defaults to False. Whether to return model outputs.

        Returns:
            dict: Dictionary containing loss and outputs.
        """

        self.forward_only = not torch.is_grad_enabled()
        if optimizer is None:
            assert self.forward_only, "Optimizer should be passed when doing backward."

        if self.forward_only:
            result = self.run_forward_only(model, data_iter, criterion, return_loss, return_outputs)
        else:
            result = self.run_forward_backward(model, data_iter, criterion, optimizer, return_loss, return_outputs)

        return result

def print_itr(microbatch, func):
    print(f"\t[COMP] Device [{dist.get_rank()}] Microbatch [{microbatch}] Operation [{func.__name__}]")

def print_obj(obj, microbatch, func, suffix=""):
    if type(obj) == dict:
        for k, v in obj.items():
            print(f"\t[COMM] Device [{dist.get_rank()}] Microbatch [{microbatch}] Operation [{func.__name__}] \n\t\t{k}: shape={v.shape}, dtype={v.dtype}, mean={v.mean().item():.6f} {suffix}", flush=True)
    elif type(obj) == torch.Tensor:
        k, v = "send tensor", obj
        print(f"\t[COMM] Device [{dist.get_rank()}] Microbatch [{microbatch}] Operation [{func.__name__}] \n\t\t{k}: shape={v.shape}, dtype={v.dtype}, mean={v.mean().item():.6f} {suffix}", flush=True)
    else:
        print(f"\t[COMM] Device [{dist.get_rank()}] Microbatch [{microbatch}] Operation [{func.__name__}] \n\t\t is type={type(obj)} {suffix}", flush=True)