import os
from collections import OrderedDict
from lib.train.trainers import BaseTrainer
from lib.train.admin import AverageMeter, StatValue
from lib.train.admin import TensorboardWriter
import torch
import time
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import lib.utils.misc as misc
import lib.train.data_recorder as data_recorder


class LTRTrainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None, use_amp=False, log_save=False):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
            use_amp - Use Automatic Mixed Precision for faster training if True
            log_save - Whether to save data to data_recorder (default: False)
        """
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)

        # Initialize statistics variables
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        # Initialize tensorboard
        if settings.local_rank in [-1, 0]:
            tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
            if not os.path.exists(tensorboard_writer_dir):
                os.makedirs(tensorboard_writer_dir)
            self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])

        # ----- Modification Start: Define and Create Checkpoint Directory -----
        print("--- Modifying ltr_trainer: Defining checkpoint directory ---")
        self.checkpoint_dir = os.path.join(self.settings.env.workspace_dir, self.settings.project_path, "checkpoints")
        if self.settings.local_rank in [-1, 0]:  # Only main process creates directory
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
                print(f"--- ltr_trainer: Created checkpoint directory at: {self.checkpoint_dir} ---")
            else:
                print(f"--- ltr_trainer: Checkpoint directory already exists at: {self.checkpoint_dir} ---")
        # ----- Modification End: Define and Create Checkpoint Directory -----

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)
        self.settings = settings
        self.use_amp = use_amp
        if use_amp:
            self.scaler = GradScaler()

        # ----- NEW: Initialize iteration counter for Excel logging frequency -----
        self.iteration_counter = 0

        # ----- NEW: Add log_save parameter -----
        self.log_save = log_save

    def cycle_dataset(self, loader):
        """Do a cycle of training or validation."""
        print('start tracking...')
        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)
        self._init_timing()
        print('epoch no.= ', self.epoch)

        # Get samples_stats_save_permission array and determine current permission
        samples_stats_save_permission = getattr(self.settings, 'samples_stats_save_permission', [False, True])
        current_epoch_idx = self.epoch - 1  # Convert to 0-based index
        
        # Determine if we should save stats for this epoch
        if current_epoch_idx < len(samples_stats_save_permission):
            should_save_stats = samples_stats_save_permission[current_epoch_idx]
        else:
            should_save_stats = samples_stats_save_permission[-1]  # Use last value for any additional epochs
            
        # Get SAVE_GRADIENTS array and determine if we should save gradients this epoch
        save_gradients = getattr(self.settings, 'SAVE_GRADIENTS', [False, True])
        if current_epoch_idx < len(save_gradients):
            should_save_gradients = save_gradients[current_epoch_idx] and loader.training
        else:
            should_save_gradients = save_gradients[-1] and loader.training
            
        print(f"Epoch {self.epoch}:")
        print(f"  - samples_stats_save_permission = {should_save_stats}")
        print(f"  - SAVE_GRADIENTS = {should_save_gradients}")

        # Initialize gradient saving if enabled for this epoch
        self._save_gradients = False
        if should_save_gradients:
            try:
                self._grad_output_dir = os.path.join(self.settings.env.workspace_dir, 'gradients')
                print(f"Gradient saving is ENABLED for this epoch. Gradients will be saved to: {self._grad_output_dir}")
                self._save_gradients = True
            except Exception as e:
                print(f"Error initializing gradient saving: {e}")

        # Initialize timing
        self.last_time_print = time.time()
        self.iteration_counter = 0

        for i, data in enumerate(loader, 1):
            self.iteration_counter += 1

            data_info = data[1]
            sample_index = data[2]
            data = data[0]

            if self.move_data_to_gpu:
                data = data.to(self.device)

            data['epoch'] = self.epoch
            data['iteration'] = i
            data['time'] = time.time()

            # Forward pass
            loss, stats = self.actor(data)
            
            # Save sample statistics if enabled for this epoch
            if should_save_stats and (i % getattr(self.settings, 'log_sample_stats_interval', 200) == 0 or i == len(loader)):
                try:
                    import lib.train.data_recorder as data_recorder
                    data_recorder.samples_stats_save(
                        sample_index=sample_index,
                        data_info=data_info,
                        stats=stats
                    )
                    print(f"Sample statistics saved at iteration {self.iteration_counter}")
                except Exception as e:
                    print(f"Error saving sample statistics: {e}")

            # Backward pass and parameter updates (only if not in stats saving mode)
            if loader.training and not should_save_stats:
                self.optimizer.zero_grad()
                if not self.use_amp:
                    loss.backward()
                    
                    # Save gradients if enabled for this iteration
                    if self._save_gradients:
                        try:
                            import lib.train.data_recorder as data_recorder
                            data_recorder.save_gradients(
                                model=self.actor.net,
                                sample_index=sample_index,
                                epoch=self.epoch,
                                output_dir=self._grad_output_dir
                            )
                            print(f"Saved gradients for sample {sample_index}")
                        except Exception as e:
                            print(f"Error saving gradients: {e}")
                    
                    if self.settings.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.settings.grad_clip_norm)
                    self.optimizer.step()
                else:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            torch.cuda.synchronize()

            # Update statistics
            batch_size = data['template_images'].shape[loader.stack_dim]
            self._update_stats(stats, batch_size, loader)

            # Print statistics
            self._print_stats(i, loader, batch_size)

    def train_epoch(self):
        # Set the current epoch in the data recorder at the beginning of each epoch
        samples_stats_save_permission = getattr(self.settings, 'samples_stats_save_permission', [False, True])
        current_epoch_idx = self.epoch - 1  # Convert to 0-based index
        should_save_stats = samples_stats_save_permission[min(current_epoch_idx, len(samples_stats_save_permission) - 1)]
        
        if should_save_stats:
            data_recorder.set_epoch(self.epoch)

        for loader in self.loaders:
            self.cycle_dataset(loader)

        self._stats_new_epoch()

    def _stats_new_epoch(self):
        # Finalize data recording for the current epoch (save remaining buffer, merge chunks)
        # Ensure this runs only on the main process to avoid race conditions during merge
        samples_stats_save_permission = getattr(self.settings, 'samples_stats_save_permission', [False, True])
        current_epoch_idx = self.epoch - 1  # Convert to 0-based index
        should_save_stats = samples_stats_save_permission[min(current_epoch_idx, len(samples_stats_save_permission) - 1)]
        
        if should_save_stats and self.settings.local_rank in [-1, 0]:
            print(f"--- ltr_trainer: Finalizing data recording for epoch {self.epoch} ---")
            data_recorder.finalize_epoch(self.epoch)
            print(f"--- ltr_trainer: Data recording finalized for epoch {self.epoch} ---")

        # Record learning rate
        for loader in self.loaders:
            if loader.training:
                try:
                    # Correct way to get LR in newer PyTorch versions
                    lr_list = [param_group['lr'] for param_group in self.optimizer.param_groups]
                except AttributeError:
                    # Fallback for older versions or different schedulers
                    try:
                        lr_list = self.lr_scheduler.get_lr()
                    except AttributeError:
                        # Handle cases where scheduler might not have get_lr or _get_lr
                        try:
                            lr_list = self.lr_scheduler._get_lr(self.epoch)
                        except Exception as e:
                            print(f"Could not retrieve learning rate: {e}")
                            lr_list = [self.optimizer.param_groups[0]['lr']]  # Default to first group LR

                for i, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i)
                    if loader.name not in self.stats or self.stats[loader.name] is None:
                        self.stats[loader.name] = OrderedDict()
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.script_name, self.settings.description)

        self.tensorboard_writer.write_epoch(self.stats, self.epoch)

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        self.prev_time = current_time

        # Define ss_print_interval once at the beginning
        ss_print_interval = getattr(self.settings, 'ss_print_interval', 50)

        if i % ss_print_interval == 0 or i == loader.__len__():
            # ===== NEW CODE BLOCK START =====
            # Calculate progress information
            total_samples_per_epoch = loader.__len__()
            samples_completed = i
            samples_left = total_samples_per_epoch - i
            progress_ratio = samples_completed / total_samples_per_epoch
            samples_left_ratio = samples_left / total_samples_per_epoch

            # Time calculations for current epoch
            time_used_seconds = current_time - self.start_time
            time_used_hours = time_used_seconds / 3600

            # Estimate time left for current epoch
            if progress_ratio > 0:
                estimated_total_epoch_time = time_used_seconds / progress_ratio
                time_left_epoch_seconds = estimated_total_epoch_time - time_used_seconds
                time_left_epoch_hours = time_left_epoch_seconds / 3600
            else:
                time_left_epoch_hours = 0

            # Time for last completed epoch (if not first epoch)
            if hasattr(self, 'last_epoch_time'):
                last_epoch_time_hours = self.last_epoch_time / 3600
            else:
                last_epoch_time_hours = 0.0

            # Total time since training start
            if hasattr(self, 'training_start_time'):
                total_training_time_seconds = current_time - self.training_start_time
                total_training_time_hours = total_training_time_seconds / 3600
            else:
                # First epoch, initialize training start time
                self.training_start_time = self.start_time
                total_training_time_hours = time_used_hours

            # Comprehensive progress line
            progress_info = (f"[{loader.name}: Epoch {self.epoch}, {i}/{total_samples_per_epoch}] "
                             f"Samples Left: {samples_left} ({samples_left_ratio:.1%}) | "
                             f"Current Epoch: {time_used_hours:.2f}h used, {time_left_epoch_hours:.2f}h left | "
                             f"Last Epoch: {last_epoch_time_hours:.2f}h | "
                             f"Total Training: {total_training_time_hours:.2f}h | "
                             f"FPS: {average_fps:.1f} ({batch_fps:.1f})")

            # Add loss statistics to the same line
            stats_str = ""
            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats):
                    if hasattr(val, 'avg'):
                        stats_str += f'{name}: {val.avg:.5f}, '

            # Combine progress info with stats
            if stats_str:
                full_line = progress_info + " | " + stats_str[:-2]  # Remove last ", "
            else:
                full_line = progress_info

            print(full_line)

            # Log to file
            log_str = full_line + '\n'
            # ===== NEW CODE BLOCK END =====

            if misc.is_main_process():
                # Ensure log file path is correctly handled
                log_file_path = getattr(self.settings, 'log_file', None)
                if log_file_path:
                    try:
                        with open(log_file_path, 'a') as f:
                            f.write(log_str)
                    except Exception as e:
                        print(f"Error writing to log file {log_file_path}: {e}")
                else:
                    print("Log file path not configured in settings.")

    # ----- Modification Start: Save Checkpoint Conditionally -----
    # Save checkpoint only for the first 10 epochs as requested for the initial stage
    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.script_name, self.settings.description)

        self.tensorboard_writer.write_epoch(self.stats, self.epoch)

        if self.epoch <= 10:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{self.epoch}.pt")
            print(f"--- ltr_trainer: Attempting to save checkpoint for epoch {self.epoch} to {checkpoint_path} ---")
            # Save the network's state_dict (all parameters)
            torch.save(self.actor.net.state_dict(), checkpoint_path)
            print(f"--- ltr_trainer: Successfully saved checkpoint for epoch {self.epoch} to {checkpoint_path} ---")
    # ----- Modification End: Save Checkpoint Conditionally -----
