from unet import *
from random import random
from beartype.typing import List, Union
from beartype import beartype
from tqdm.auto import tqdm
from functools import partial
from contextlib import contextmanager, nullcontext
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch import nn
from torch.cuda.amp import autocast
import torchvision.transforms as T
import kornia.augmentation as K
from einops import rearrange, repeat, reduce, pack, unpack


class Imagen(nn.Module):
    def __init__(
            self,
            unets,
            *,
            image_sizes,  # for cascading ddpm, image size at each stage
            text_encoder_name='t5_encoder',
            text_embed_dim=None,
            channels=3,
            timesteps=1000,
            cond_drop_prob=0.1,
            loss_type='l2',
            noise_schedules='cosine',
            pred_objectives='noise',
            random_crop_sizes=None,
            lowres_noise_schedule='linear',
            lowres_sample_noise_level=0.2,
            # in the paper, they present a new trick where they noise the lowres conditioning image, and at sample time, fix it to a certain level (0.1 or 0.3) - the unets are also made to be conditioned on this noise level
            per_sample_random_aug_noise_level=False,
            # unclear when conditioning on augmentation noise level, whether each batch element receives a random aug noise value - turning off due to @marunine's find
            condition_on_text=True,
            auto_normalize_img=True,
            # whether to take care of normalizing the image from [0, 1] to [-1, 1] and back automatically - you can turn this off if you want to pass in the [-1, 1] ranged image yourself from the dataloader
            dynamic_thresholding=True,
            dynamic_thresholding_percentile=0.95,  # unsure what this was based on perusal of paper
            only_train_unet_number=None,
            temporal_downsample_factor=1,
            resize_cond_video_frames=True,
            resize_mode='nearest',
            min_snr_loss_weight=True,  # https://arxiv.org/abs/2303.09556
            min_snr_gamma=5
    ):
        super().__init__()

        # loss

        if loss_type == 'l1':
            loss_fn = F.l1_loss
        elif loss_type == 'l2':
            loss_fn = F.mse_loss
        elif loss_type == 'huber':
            loss_fn = F.smooth_l1_loss
        else:
            raise NotImplementedError()

        self.loss_type = loss_type
        self.loss_fn = loss_fn

        # conditioning hparams

        self.condition_on_text = condition_on_text
        self.unconditional = not condition_on_text

        # channels

        self.channels = channels

        # automatically take care of ensuring that first unet is unconditional
        # while the rest of the unets are conditioned on the low resolution image produced by previous unet

        unets = cast_tuple(unets)
        num_unets = len(unets)

        # determine noise schedules per unet

        timesteps = cast_tuple(timesteps, num_unets)

        # make sure noise schedule defaults to 'cosine', 'cosine', and then 'linear' for rest of super-resoluting unets

        noise_schedules = cast_tuple(noise_schedules)
        noise_schedules = pad_tuple_to_length(noise_schedules, 2, 'cosine')
        noise_schedules = pad_tuple_to_length(noise_schedules, num_unets, 'linear')

        # construct noise schedulers

        noise_scheduler_klass = GaussianDiffusionContinuousTimes
        self.noise_schedulers = nn.ModuleList([])

        for timestep, noise_schedule in zip(timesteps, noise_schedules):
            noise_scheduler = noise_scheduler_klass(noise_schedule=noise_schedule, timesteps=timestep)
            self.noise_schedulers.append(noise_scheduler)

        # randomly cropping for upsampler training

        self.random_crop_sizes = cast_tuple(random_crop_sizes, num_unets)
        assert not exists(first(
            self.random_crop_sizes)), 'you should not need to randomly crop image during training for base unet, only for upsamplers - so pass in `random_crop_sizes = (None, 128, 256)` as example'

        # lowres augmentation noise schedule

        self.lowres_noise_schedule = GaussianDiffusionContinuousTimes(noise_schedule=lowres_noise_schedule)

        # ddpm objectives - predicting noise by default

        self.pred_objectives = cast_tuple(pred_objectives, num_unets)

        # get text encoder

        self.text_encoder_name = text_encoder_name
        self.text_embed_dim = 768#default(text_embed_dim, lambda: get_encoded_dim(text_encoder_name))

        self.encode_text = None #partial(t5_encode_text, name=text_encoder_name)

        # construct unets

        self.unets = nn.ModuleList([])

        self.unet_being_trained_index = -1  # keeps track of which unet is being trained at the moment
        self.only_train_unet_number = only_train_unet_number

        for ind, one_unet in enumerate(unets):
            assert isinstance(one_unet, (Unet, NullUnet))
            is_first = ind == 0

            one_unet = one_unet.cast_model_parameters(
                lowres_cond=not is_first,
                cond_on_text=self.condition_on_text,
                text_embed_dim=self.text_embed_dim if self.condition_on_text else None,
                channels=self.channels,
                channels_out=self.channels
            )

            self.unets.append(one_unet)

        # unet image sizes

        image_sizes = cast_tuple(image_sizes)
        self.image_sizes = image_sizes

        assert num_unets == len(
            image_sizes), f'you did not supply the correct number of u-nets ({len(unets)}) for resolutions {image_sizes}'

        self.sample_channels = cast_tuple(self.channels, num_unets)

        # determine whether we are training on images or video

        is_video = any([False for unet in self.unets])
        self.is_video = is_video

        self.right_pad_dims_to_datatype = partial(rearrange,
                                                  pattern=('b -> b 1 1 1' if not is_video else 'b -> b 1 1 1 1'))

        self.resize_to = resize_image_to
        self.resize_to = partial(self.resize_to, mode=resize_mode)

        # temporal interpolation

        temporal_downsample_factor = cast_tuple(temporal_downsample_factor, num_unets)
        self.temporal_downsample_factor = temporal_downsample_factor

        self.resize_cond_video_frames = resize_cond_video_frames
        self.temporal_downsample_divisor = temporal_downsample_factor[0]

        assert temporal_downsample_factor[-1] == 1, 'downsample factor of last stage must be 1'
        assert tuple(sorted(temporal_downsample_factor,
                            reverse=True)) == temporal_downsample_factor, 'temporal downsample factor must be in order of descending'

        # cascading ddpm related stuff

        lowres_conditions = tuple(map(lambda t: t.lowres_cond, self.unets))
        assert lowres_conditions == (False, *((True,) * (
                    num_unets - 1))), 'the first unet must be unconditioned (by low resolution image), and the rest of the unets must have `lowres_cond` set to True'

        self.lowres_sample_noise_level = lowres_sample_noise_level
        self.per_sample_random_aug_noise_level = per_sample_random_aug_noise_level

        # classifier free guidance

        self.cond_drop_prob = cond_drop_prob
        self.can_classifier_guidance = cond_drop_prob > 0.

        # normalize and unnormalize image functions

        self.normalize_img = normalize_neg_one_to_one if auto_normalize_img else identity
        self.unnormalize_img = unnormalize_zero_to_one if auto_normalize_img else identity
        self.input_image_range = (0. if auto_normalize_img else -1., 1.)

        # dynamic thresholding

        self.dynamic_thresholding = cast_tuple(dynamic_thresholding, num_unets)
        self.dynamic_thresholding_percentile = dynamic_thresholding_percentile

        # min snr loss weight

        min_snr_loss_weight = cast_tuple(min_snr_loss_weight, num_unets)
        min_snr_gamma = cast_tuple(min_snr_gamma, num_unets)

        assert len(min_snr_loss_weight) == len(min_snr_gamma) == num_unets
        self.min_snr_gamma = tuple(
            (gamma if use_min_snr else None) for use_min_snr, gamma in zip(min_snr_loss_weight, min_snr_gamma))

        # one temp parameter for keeping track of device

        self.register_buffer('_temp', torch.tensor([0.]), persistent=False)

        # default to device of unets passed in

        self.to(next(self.unets.parameters()).device)

    def force_unconditional_(self):
        self.condition_on_text = False
        self.unconditional = True

        for unet in self.unets:
            unet.cond_on_text = False

    @property
    def device(self):
        return self._temp.device

    def get_unet(self, unet_number):
        assert 0 < unet_number <= len(self.unets)
        index = unet_number - 1

        if isinstance(self.unets, nn.ModuleList):
            unets_list = [unet for unet in self.unets]
            delattr(self, 'unets')
            self.unets = unets_list

        if index != self.unet_being_trained_index:
            for unet_index, unet in enumerate(self.unets):
                unet.to(self.device if unet_index == index else 'cpu')

        self.unet_being_trained_index = index
        return self.unets[index]

    def reset_unets_all_one_device(self, device=None):
        device = default(device, self.device)
        self.unets = nn.ModuleList([*self.unets])
        self.unets.to(device)

        self.unet_being_trained_index = -1

    @contextmanager
    def one_unet_in_gpu(self, unet_number=None, unet=None):
        assert exists(unet_number) ^ exists(unet)

        if exists(unet_number):
            unet = self.unets[unet_number - 1]

        cpu = torch.device('cpu')

        devices = [module_device(unet) for unet in self.unets]

        self.unets.to(cpu)
        unet.to(self.device)

        yield

        for unet, device in zip(self.unets, devices):
            unet.to(device)

    # overriding state dict functions

    def state_dict(self, *args, **kwargs):
        self.reset_unets_all_one_device()
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        self.reset_unets_all_one_device()
        return super().load_state_dict(*args, **kwargs)

    # gaussian diffusion methods

    def p_mean_variance(
            self,
            unet,
            x,
            t,
            *,
            noise_scheduler,
            text_embeds=None,
            text_mask=None,
            cond_images=None,
            cond_video_frames=None,
            post_cond_video_frames=None,
            lowres_cond_img=None,
            self_cond=None,
            lowres_noise_times=None,
            cond_scale=1.,
            model_output=None,
            t_next=None,
            pred_objective='noise',
            dynamic_threshold=True
    ):
        assert not (
                    cond_scale != 1. and not self.can_classifier_guidance), 'imagen was not trained with conditional dropout, and thus one cannot use classifier free guidance (cond_scale anything other than 1)'

        video_kwargs = dict()
        if self.is_video:
            video_kwargs = dict(
                cond_video_frames=cond_video_frames,
                post_cond_video_frames=post_cond_video_frames,
            )

        pred = default(model_output, lambda: unet.forward_with_cond_scale(
            x,
            noise_scheduler.get_condition(t),
            text_embeds=text_embeds,
            text_mask=text_mask,
            cond_images=cond_images,
            cond_scale=cond_scale,
            lowres_cond_img=lowres_cond_img,
            self_cond=self_cond,
            lowres_noise_times=self.lowres_noise_schedule.get_condition(lowres_noise_times),
            **video_kwargs
        ))

        if pred_objective == 'noise':
            x_start = noise_scheduler.predict_start_from_noise(x, t=t, noise=pred)
        elif pred_objective == 'x_start':
            x_start = pred
        elif pred_objective == 'v':
            x_start = noise_scheduler.predict_start_from_v(x, t=t, v=pred)
        else:
            raise ValueError(f'unknown objective {pred_objective}')

        if dynamic_threshold:
            # following pseudocode in appendix
            # s is the dynamic threshold, determined by percentile of absolute values of reconstructed sample per batch element
            s = torch.quantile(
                rearrange(x_start, 'b ... -> b (...)').abs(),
                self.dynamic_thresholding_percentile,
                dim=-1
            )

            s.clamp_(min=1.)
            s = right_pad_dims_to(x_start, s)
            x_start = x_start.clamp(-s, s) / s
        else:
            x_start.clamp_(-1., 1.)

        mean_and_variance = noise_scheduler.q_posterior(x_start=x_start, x_t=x, t=t, t_next=t_next)
        return mean_and_variance, x_start

    @torch.no_grad()
    def p_sample(
            self,
            unet,
            x,
            t,
            *,
            noise_scheduler,
            t_next=None,
            text_embeds=None,
            text_mask=None,
            cond_images=None,
            cond_video_frames=None,
            post_cond_video_frames=None,
            cond_scale=1.,
            self_cond=None,
            lowres_cond_img=None,
            lowres_noise_times=None,
            pred_objective='noise',
            dynamic_threshold=True
    ):
        b, *_, device = *x.shape, x.device

        video_kwargs = dict()
        if self.is_video:
            video_kwargs = dict(
                cond_video_frames=cond_video_frames,
                post_cond_video_frames=post_cond_video_frames,
            )

        (model_mean, _, model_log_variance), x_start = self.p_mean_variance(
            unet,
            x=x,
            t=t,
            t_next=t_next,
            noise_scheduler=noise_scheduler,
            text_embeds=text_embeds,
            text_mask=text_mask,
            cond_images=cond_images,
            cond_scale=cond_scale,
            lowres_cond_img=lowres_cond_img,
            self_cond=self_cond,
            lowres_noise_times=lowres_noise_times,
            pred_objective=pred_objective,
            dynamic_threshold=dynamic_threshold,
            **video_kwargs
        )

        noise = torch.randn_like(x)
        # no noise when t == 0
        is_last_sampling_timestep = (t_next == 0) if isinstance(noise_scheduler,
                                                                GaussianDiffusionContinuousTimes) else (t == 0)
        nonzero_mask = (1 - is_last_sampling_timestep.float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred, x_start

    @torch.no_grad()
    def p_sample_loop(
            self,
            unet,
            shape,
            *,
            noise_scheduler,
            lowres_cond_img=None,
            lowres_noise_times=None,
            text_embeds=None,
            text_mask=None,
            cond_images=None,
            cond_video_frames=None,
            post_cond_video_frames=None,
            inpaint_images=None,
            inpaint_videos=None,
            inpaint_masks=None,
            inpaint_resample_times=5,
            init_images=None,
            skip_steps=None,
            cond_scale=1,
            pred_objective='noise',
            dynamic_threshold=True,
            use_tqdm=True
    ):
        device = self.device

        batch = shape[0]
        img = torch.randn(shape, device=device)

        # video

        is_video = len(shape) == 5
        frames = shape[-3] if is_video else None
        resize_kwargs = dict(target_frames=frames) if exists(frames) else dict()

        # for initialization with an image or video

        if exists(init_images):
            img += init_images

        # keep track of x0, for self conditioning

        x_start = None

        # prepare inpainting

        inpaint_images = default(inpaint_videos, inpaint_images)

        has_inpainting = exists(inpaint_images) and exists(inpaint_masks)
        resample_times = inpaint_resample_times if has_inpainting else 1

        if has_inpainting:
            inpaint_images = self.normalize_img(inpaint_images)
            inpaint_images = self.resize_to(inpaint_images, shape[-1], **resize_kwargs)
            inpaint_masks = self.resize_to(rearrange(inpaint_masks, 'b ... -> b 1 ...').float(), shape[-1],
                                           **resize_kwargs).bool()

        # time

        timesteps = noise_scheduler.get_sampling_timesteps(batch, device=device)

        # whether to skip any steps

        skip_steps = default(skip_steps, 0)
        timesteps = timesteps[skip_steps:]

        # video conditioning kwargs

        video_kwargs = dict()
        if self.is_video:
            video_kwargs = dict(
                cond_video_frames=cond_video_frames,
                post_cond_video_frames=post_cond_video_frames,
            )

        for times, times_next in tqdm(timesteps, desc='sampling loop time step', total=len(timesteps),
                                      disable=not use_tqdm):
            is_last_timestep = times_next == 0

            for r in reversed(range(resample_times)):
                is_last_resample_step = r == 0

                if has_inpainting:
                    noised_inpaint_images, *_ = noise_scheduler.q_sample(inpaint_images, t=times)
                    img = img * ~inpaint_masks + noised_inpaint_images * inpaint_masks

                self_cond = x_start if unet.self_cond else None

                img, x_start = self.p_sample(
                    unet,
                    img,
                    times,
                    t_next=times_next,
                    text_embeds=text_embeds,
                    text_mask=text_mask,
                    cond_images=cond_images,
                    cond_scale=cond_scale,
                    self_cond=self_cond,
                    lowres_cond_img=lowres_cond_img,
                    lowres_noise_times=lowres_noise_times,
                    noise_scheduler=noise_scheduler,
                    pred_objective=pred_objective,
                    dynamic_threshold=dynamic_threshold,
                    **video_kwargs
                )

                if has_inpainting and not (is_last_resample_step or torch.all(is_last_timestep)):
                    renoised_img = noise_scheduler.q_sample_from_to(img, times_next, times)

                    img = torch.where(
                        self.right_pad_dims_to_datatype(is_last_timestep),
                        img,
                        renoised_img
                    )

        img.clamp_(-1., 1.)

        # final inpainting

        if has_inpainting:
            img = img * ~inpaint_masks + inpaint_images * inpaint_masks

        unnormalize_img = self.unnormalize_img(img)
        return unnormalize_img

    @torch.no_grad()
    @eval_decorator
    @beartype
    def sample(
            self,
            texts: List[str] = None,
            text_masks=None,
            text_embeds=None,
            video_frames=None,
            cond_images=None,
            cond_video_frames=None,
            post_cond_video_frames=None,
            inpaint_videos=None,
            inpaint_images=None,
            inpaint_masks=None,
            inpaint_resample_times=5,
            init_images=None,
            skip_steps=None,
            batch_size=1,
            cond_scale=1.,
            lowres_sample_noise_level=None,
            start_at_unet_number=1,
            start_image_or_video=None,
            stop_at_unet_number=None,
            return_all_unet_outputs=False,
            return_pil_images=False,
            device=None,
            use_tqdm=True,
            use_one_unet_in_gpu=True
    ):
        device = default(device, self.device)
        self.reset_unets_all_one_device(device=device)

        cond_images = maybe(cast_uint8_images_to_float)(cond_images)

        if exists(texts) and not exists(text_embeds) and not self.unconditional:
            assert all([*map(len, texts)]), 'text cannot be empty'

            with autocast(enabled=False):
                text_embeds, text_masks = self.encode_text(texts, return_attn_mask=True)

            text_embeds, text_masks = map(lambda t: t.to(device), (text_embeds, text_masks))

        if not self.unconditional:
            assert exists(
                text_embeds), 'text must be passed in if the network was not trained without text `condition_on_text` must be set to `False` when training'

            text_masks = default(text_masks, lambda: torch.any(text_embeds != 0., dim=-1))
            batch_size = text_embeds.shape[0]

        # inpainting

        inpaint_images = default(inpaint_videos, inpaint_images)

        if exists(inpaint_images):
            if self.unconditional:
                if batch_size == 1:  # assume researcher wants to broadcast along inpainted images
                    batch_size = inpaint_images.shape[0]

            assert inpaint_images.shape[
                       0] == batch_size, 'number of inpainting images must be equal to the specified batch size on sample `sample(batch_size=<int>)``'
            assert not (self.condition_on_text and inpaint_images.shape[0] != text_embeds.shape[
                0]), 'number of inpainting images must be equal to the number of text to be conditioned on'

        assert not (self.condition_on_text and not exists(
            text_embeds)), 'text or text encodings must be passed into imagen if specified'
        assert not (not self.condition_on_text and exists(
            text_embeds)), 'imagen specified not to be conditioned on text, yet it is presented'
        assert not (exists(text_embeds) and text_embeds.shape[
            -1] != self.text_embed_dim), f'invalid text embedding dimension being passed in (should be {self.text_embed_dim})'

        assert not (exists(inpaint_images) ^ exists(
            inpaint_masks)), 'inpaint images and masks must be both passed in to do inpainting'

        outputs = []

        is_cuda = next(self.parameters()).is_cuda
        device = next(self.parameters()).device

        lowres_sample_noise_level = default(lowres_sample_noise_level, self.lowres_sample_noise_level)

        num_unets = len(self.unets)

        # condition scaling

        cond_scale = cast_tuple(cond_scale, num_unets)

        # add frame dimension for video

        if self.is_video and exists(inpaint_images):
            video_frames = inpaint_images.shape[2]

            if inpaint_masks.ndim == 3:
                inpaint_masks = repeat(inpaint_masks, 'b h w -> b f h w', f=video_frames)

            assert inpaint_masks.shape[1] == video_frames

        assert not (self.is_video and not exists(
            video_frames)), 'video_frames must be passed in on sample time if training on video'

        all_frame_dims = calc_all_frame_dims(self.temporal_downsample_factor, video_frames)

        frames_to_resize_kwargs = lambda frames: dict(target_frames=frames) if exists(frames) else dict()

        # for initial image and skipping steps

        init_images = cast_tuple(init_images, num_unets)
        init_images = [maybe(self.normalize_img)(init_image) for init_image in init_images]

        skip_steps = cast_tuple(skip_steps, num_unets)

        # handle starting at a unet greater than 1, for training only-upscaler training

        if start_at_unet_number > 1:
            assert start_at_unet_number <= num_unets, 'must start a unet that is less than the total number of unets'
            assert not exists(stop_at_unet_number) or start_at_unet_number <= stop_at_unet_number
            assert exists(start_image_or_video), 'starting image or video must be supplied if only doing upscaling'

            prev_image_size = self.image_sizes[start_at_unet_number - 2]
            prev_frame_size = all_frame_dims[start_at_unet_number - 2][0] if self.is_video else None
            img = self.resize_to(start_image_or_video, prev_image_size, **frames_to_resize_kwargs(prev_frame_size))

        # go through each unet in cascade

        for unet_number, unet, channel, image_size, frame_dims, noise_scheduler, pred_objective, dynamic_threshold, unet_cond_scale, unet_init_images, unet_skip_steps in tqdm(
                zip(range(1, num_unets + 1), self.unets, self.sample_channels, self.image_sizes, all_frame_dims,
                    self.noise_schedulers, self.pred_objectives, self.dynamic_thresholding, cond_scale, init_images,
                    skip_steps), disable=not use_tqdm):

            if unet_number < start_at_unet_number:
                continue

            assert not isinstance(unet, NullUnet), 'one cannot sample from null / placeholder unets'

            context = self.one_unet_in_gpu(unet=unet) if is_cuda and use_one_unet_in_gpu else nullcontext()

            with context:
                # video kwargs

                video_kwargs = dict()
                if self.is_video:
                    video_kwargs = dict(
                        cond_video_frames=cond_video_frames,
                        post_cond_video_frames=post_cond_video_frames,
                    )

                    video_kwargs = compact(video_kwargs)


                # low resolution conditioning

                lowres_cond_img = lowres_noise_times = None
                shape = (batch_size, channel, *frame_dims, image_size, image_size)

                resize_kwargs = dict(target_frames=frame_dims[0]) if self.is_video else dict()

                if unet.lowres_cond:
                    lowres_noise_times = self.lowres_noise_schedule.get_times(batch_size, lowres_sample_noise_level,
                                                                              device=device)

                    lowres_cond_img = self.resize_to(img, image_size, **resize_kwargs)

                    lowres_cond_img = self.normalize_img(lowres_cond_img)
                    lowres_cond_img, *_ = self.lowres_noise_schedule.q_sample(x_start=lowres_cond_img,
                                                                              t=lowres_noise_times,
                                                                              noise=torch.randn_like(lowres_cond_img))

                # init images or video

                if exists(unet_init_images):
                    unet_init_images = self.resize_to(unet_init_images, image_size, **resize_kwargs)

                # shape of stage

                shape = (batch_size, self.channels, *frame_dims, image_size, image_size)

                img = self.p_sample_loop(
                    unet,
                    shape,
                    text_embeds=text_embeds,
                    text_mask=text_masks,
                    cond_images=cond_images,
                    inpaint_images=inpaint_images,
                    inpaint_masks=inpaint_masks,
                    inpaint_resample_times=inpaint_resample_times,
                    init_images=unet_init_images,
                    skip_steps=unet_skip_steps,
                    cond_scale=unet_cond_scale,
                    lowres_cond_img=lowres_cond_img,
                    lowres_noise_times=lowres_noise_times,
                    noise_scheduler=noise_scheduler,
                    pred_objective=pred_objective,
                    dynamic_threshold=dynamic_threshold,
                    use_tqdm=use_tqdm,
                    **video_kwargs
                )

                outputs.append(img)

            if exists(stop_at_unet_number) and stop_at_unet_number == unet_number:
                break

        output_index = -1 if not return_all_unet_outputs else slice(
            None)  # either return last unet output or all unet outputs

        if not return_pil_images:
            return outputs[output_index]

        if not return_all_unet_outputs:
            outputs = outputs[-1:]

        assert not self.is_video, 'converting sampled video tensor to video file is not supported yet'

        pil_images = list(map(lambda img: list(map(T.ToPILImage(), img.unbind(dim=0))), outputs))

        return pil_images[
            output_index]  # now you have a bunch of pillow images you can just .save(/where/ever/you/want.png)

    @beartype
    def p_losses(
            self,
            unet: Union[Unet, NullUnet, DistributedDataParallel],
            x_start,
            times,
            *,
            noise_scheduler,
            lowres_cond_img=None,
            lowres_aug_times=None,
            text_embeds=None,
            text_mask=None,
            cond_images=None,
            noise=None,
            times_next=None,
            pred_objective='noise',
            min_snr_gamma=None,
            random_crop_size=None,
            **kwargs
    ):
        is_video = x_start.ndim == 5

        noise = default(noise, lambda: torch.randn_like(x_start))

        # normalize to [-1, 1]

        x_start = self.normalize_img(x_start)
        lowres_cond_img = maybe(self.normalize_img)(lowres_cond_img)

        # random cropping during training
        # for upsamplers

        if exists(random_crop_size):
            if is_video:
                frames = x_start.shape[2]
                x_start, lowres_cond_img, noise = map(lambda t: rearrange(t, 'b c f h w -> (b f) c h w'),
                                                      (x_start, lowres_cond_img, noise))

            aug = K.RandomCrop((random_crop_size, random_crop_size), p=1.)

            # make sure low res conditioner and image both get augmented the same way
            # detailed https://kornia.readthedocs.io/en/latest/augmentation.module.html?highlight=randomcrop#kornia.augmentation.RandomCrop
            x_start = aug(x_start)
            lowres_cond_img = aug(lowres_cond_img, params=aug._params)
            noise = aug(noise, params=aug._params)

            if is_video:
                x_start, lowres_cond_img, noise = map(lambda t: rearrange(t, '(b f) c h w -> b c f h w', f=frames),
                                                      (x_start, lowres_cond_img, noise))

        # get x_t

        x_noisy, log_snr, alpha, sigma = noise_scheduler.q_sample(x_start=x_start, t=times, noise=noise)

        # also noise the lowres conditioning image
        # at sample time, they then fix the noise level of 0.1 - 0.3

        lowres_cond_img_noisy = None
        if exists(lowres_cond_img):
            lowres_aug_times = default(lowres_aug_times, times)
            lowres_cond_img_noisy, *_ = self.lowres_noise_schedule.q_sample(x_start=lowres_cond_img, t=lowres_aug_times,
                                                                            noise=torch.randn_like(lowres_cond_img))

        # time condition

        noise_cond = noise_scheduler.get_condition(times)

        # unet kwargs

        unet_kwargs = dict(
            text_embeds=text_embeds,
            text_mask=text_mask,
            cond_images=cond_images,
            lowres_noise_times=self.lowres_noise_schedule.get_condition(lowres_aug_times),
            lowres_cond_img=lowres_cond_img_noisy,
            cond_drop_prob=self.cond_drop_prob,
            **kwargs
        )

        # self condition if needed

        # Because 'unet' can be an instance of DistributedDataParallel coming from the
        # ImagenTrainer.unet_being_trained when invoking ImagenTrainer.forward(), we need to
        # access the member 'module' of the wrapped unet instance.
        self_cond = unet.module.self_cond if isinstance(unet, DistributedDataParallel) else unet.self_cond

        if self_cond and random() < 0.5:
            with torch.no_grad():
                pred = unet.forward(
                    x_noisy,
                    noise_cond,
                    **unet_kwargs
                ).detach()

                x_start = noise_scheduler.predict_start_from_noise(x_noisy, t=times,
                                                                   noise=pred) if pred_objective == 'noise' else pred

                unet_kwargs = {**unet_kwargs, 'self_cond': x_start}

        # get prediction

        pred = unet.forward(
            x_noisy,
            noise_cond,
            **unet_kwargs
        )

        # prediction objective

        if pred_objective == 'noise':
            target = noise
        elif pred_objective == 'x_start':
            target = x_start
        elif pred_objective == 'v':
            # derivation detailed in Appendix D of Progressive Distillation paper
            # https://arxiv.org/abs/2202.00512
            # this makes distillation viable as well as solve an issue with color shifting in upresoluting unets, noted in imagen-video
            target = alpha * noise - sigma * x_start
        else:
            raise ValueError(f'unknown objective {pred_objective}')

        # losses

        losses = self.loss_fn(pred, target, reduction='none')
        losses = reduce(losses, 'b ... -> b', 'mean')

        # min snr loss reweighting

        snr = log_snr.exp()
        maybe_clipped_snr = snr.clone()

        if exists(min_snr_gamma):
            maybe_clipped_snr.clamp_(max=min_snr_gamma)

        if pred_objective == 'noise':
            loss_weight = maybe_clipped_snr / snr
        elif pred_objective == 'x_start':
            loss_weight = maybe_clipped_snr
        elif pred_objective == 'v':
            loss_weight = maybe_clipped_snr / (snr + 1)

        losses = losses * loss_weight
        return losses.mean()

    @beartype
    def forward(
            self,
            images,  # rename to images or video
            unet: Union[Unet, NullUnet, DistributedDataParallel] = None,
            texts: List[str] = None,
            text_embeds=None,
            text_masks=None,
            unet_number=None,
            cond_images=None,
            **kwargs
    ):
        if self.is_video and images.ndim == 4:
            images = rearrange(images, 'b c h w -> b c 1 h w')
            kwargs.update(ignore_time=True)

        assert images.shape[-1] == images.shape[
            -2], f'the images you pass in must be a square, but received dimensions of {images.shape[2]}, {images.shape[-1]}'
        assert not (len(self.unets) > 1 and not exists(
            unet_number)), f'you must specify which unet you want trained, from a range of 1 to {len(self.unets)}, if you are training cascading DDPM (multiple unets)'
        unet_number = default(unet_number, 1)
        assert not exists(
            self.only_train_unet_number) or self.only_train_unet_number == unet_number, 'you can only train on unet #{self.only_train_unet_number}'

        images = cast_uint8_images_to_float(images)
        cond_images = maybe(cast_uint8_images_to_float)(cond_images)

        assert images.dtype == torch.float, f'images tensor needs to be floats but {images.dtype} dtype found instead'

        unet_index = unet_number - 1

        unet = default(unet, lambda: self.get_unet(unet_number))

        assert not isinstance(unet, NullUnet), 'null unet cannot and should not be trained'

        noise_scheduler = self.noise_schedulers[unet_index]
        min_snr_gamma = self.min_snr_gamma[unet_index]
        pred_objective = self.pred_objectives[unet_index]
        target_image_size = self.image_sizes[unet_index]
        random_crop_size = self.random_crop_sizes[unet_index]
        prev_image_size = self.image_sizes[unet_index - 1] if unet_index > 0 else None

        b, c, *_, h, w, device, is_video = *images.shape, images.device, images.ndim == 5

        assert images.shape[1] == self.channels
        assert h >= target_image_size and w >= target_image_size

        frames = images.shape[2] if is_video else None
        all_frame_dims = tuple(
            safe_get_tuple_index(el, 0) for el in calc_all_frame_dims(self.temporal_downsample_factor, frames))
        ignore_time = kwargs.get('ignore_time', False)

        target_frame_size = all_frame_dims[unet_index] if is_video and not ignore_time else None
        prev_frame_size = all_frame_dims[unet_index - 1] if is_video and not ignore_time and unet_index > 0 else None
        frames_to_resize_kwargs = lambda frames: dict(target_frames=frames) if exists(frames) else dict()

        times = noise_scheduler.sample_random_times(b, device=device)

        if exists(texts) and not exists(text_embeds) and not self.unconditional:
            assert all([*map(len, texts)]), 'text cannot be empty'
            assert len(texts) == len(
                images), 'number of text captions does not match up with the number of images given'

            with autocast(enabled=False):
                text_embeds, text_masks = self.encode_text(texts, return_attn_mask=True)

            text_embeds, text_masks = map(lambda t: t.to(images.device), (text_embeds, text_masks))

        if not self.unconditional:
            text_masks = default(text_masks, lambda: torch.any(text_embeds != 0., dim=-1))

        assert not (self.condition_on_text and not exists(
            text_embeds)), 'text or text encodings must be passed into decoder if specified'
        assert not (not self.condition_on_text and exists(
            text_embeds)), 'decoder specified not to be conditioned on text, yet it is presented'

        assert not (exists(text_embeds) and text_embeds.shape[
            -1] != self.text_embed_dim), f'invalid text embedding dimension being passed in (should be {self.text_embed_dim})'

        # handle video frame conditioning

        # handle low resolution conditioning

        lowres_cond_img = lowres_aug_times = None
        if exists(prev_image_size):
            lowres_cond_img = self.resize_to(images, prev_image_size, **frames_to_resize_kwargs(prev_frame_size),
                                             clamp_range=self.input_image_range)
            lowres_cond_img = self.resize_to(lowres_cond_img, target_image_size,
                                             **frames_to_resize_kwargs(target_frame_size),
                                             clamp_range=self.input_image_range)

            if self.per_sample_random_aug_noise_level:
                lowres_aug_times = self.lowres_noise_schedule.sample_random_times(b, device=device)
            else:
                lowres_aug_time = self.lowres_noise_schedule.sample_random_times(1, device=device)
                lowres_aug_times = repeat(lowres_aug_time, '1 -> b', b=b)

        images = self.resize_to(images, target_image_size, **frames_to_resize_kwargs(target_frame_size))

        return self.p_losses(unet, images, times, text_embeds=text_embeds, text_mask=text_masks,
                             cond_images=cond_images, noise_scheduler=noise_scheduler, lowres_cond_img=lowres_cond_img,
                             lowres_aug_times=lowres_aug_times, pred_objective=pred_objective,
                             min_snr_gamma=min_snr_gamma, random_crop_size=random_crop_size, **kwargs)
