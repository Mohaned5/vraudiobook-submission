from .PanoGenerator import PanoGenerator
from ..modules.utils import tensor_to_image
from .MVGenModel import MultiViewBaseModel
import torch
import os
from PIL import Image
from external.Perspective_and_Equirectangular import e2p
from einops import rearrange
from lightning.pytorch.utilities import rank_zero_only
import lpips
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.transforms as T
import sys
from omegaconf import OmegaConf
import re
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.ssim import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image.inception import InceptionScore
from ..Real_ESRGAN.realesrgan_utils import initialize_realesrgan, upscale_image

class PanFusion(PanoGenerator):
    def __init__(
            self,
            use_pers_prompt: bool = True,
            use_pano_prompt: bool = True,
            copy_pano_prompt: bool = True,
            **kwargs
            ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.lpips_model = lpips.LPIPS(net='alex')
        self.lpips_model.eval()
        for param in self.lpips_model.parameters():
            param.requires_grad = False

        self.fid_metric = FrechetInceptionDistance(feature=2048, normalize=True)
        for param in self.fid_metric.parameters():
            param.requires_grad = False
        
        self.fid_transform_float = T.Compose([
            T.Resize(299),           
            T.CenterCrop(299),
            T.ToTensor(),             
        ])

        self.transform_uint8_scaled = T.Compose([
            T.Resize(299),           
            T.CenterCrop(299),
            T.ToTensor(),
            T.Lambda(lambda x: (x * 255).to(torch.uint8)),  # Convert to uint8
        ])

        self.transform_raw_uint8 = T.Compose([
            T.Resize(299),
            T.CenterCrop(299),
            T.PILToTensor(),  # Produces a torch.Tensor with dtype=torch.uint8
        ])
        self.img_transform_float = T.Compose([
            T.Resize(299),
            T.CenterCrop(299),
            T.ToTensor(),     # Produces a float tensor in [0,1]
        ])

        self.kid_metric = KernelInceptionDistance(subset_size=50) 
        self.ms_ssim_metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
        self.inception_score = InceptionScore()

        self._val_lpips = []

        self._test_lpips = []
        
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../id_embeddings")) 
        self.identity_embedding_paths = {
            "[man_1]": os.path.join(base_dir, "man/man_1.pt"),
            "[man_2]": os.path.join(base_dir, "man/man_2.pt"),
            "[man_3]": os.path.join(base_dir, "man/man_3.pt"),
            "[man_4]": os.path.join(base_dir, "man/man_4.pt"),
            "[man_5]": os.path.join(base_dir, "man/man_5.pt"),
            "[man_6]": os.path.join(base_dir, "man/man_6.pt"),
            "[man_7]": os.path.join(base_dir, "man/man_7.pt"),
            "[man_8]": os.path.join(base_dir, "man/man_8.pt"),
            "[man_9]": os.path.join(base_dir, "man/man_9.pt"),
            "[woman_1]": os.path.join(base_dir, "woman/woman_1.pt"),
            "[woman_2]": os.path.join(base_dir, "woman/woman_2.pt"),
            "[woman_3]": os.path.join(base_dir, "woman/woman_3.pt"),
            "[woman_4]": os.path.join(base_dir, "woman/woman_4.pt"),
            "[woman_5]": os.path.join(base_dir, "woman/woman_5.pt"),
            "[woman_6]": os.path.join(base_dir, "woman/woman_6.pt"),
            "[woman_7]": os.path.join(base_dir, "woman/woman_7.pt"),
            "[woman_8]": os.path.join(base_dir, "woman/woman_8.pt"),
            "[woman_9]": os.path.join(base_dir, "woman/woman_9.pt")
        }

        self.valid_ids = list(self.identity_embedding_paths.keys())

    def instantiate_model(self):
        pano_unet, cn = self.load_pano()
        unet, pers_cn = self.load_pers()
        self.mv_base_model = MultiViewBaseModel(unet, pano_unet, pers_cn, cn, self.hparams.unet_pad)
        if not self.hparams.layout_cond:
            self.trainable_params.extend(self.mv_base_model.trainable_parameters)

    def init_noise(self, bs, equi_h, equi_w, pers_h, pers_w, cameras, device):
        cameras = {k: rearrange(v, 'b m ... -> (b m) ...') for k, v in cameras.items()}
        m = len(cameras['FoV']) // bs
        pano_noise = torch.randn(
            bs, 1, 4, equi_h, equi_w, device=device)
        pano_noises = pano_noise.expand(-1, m, -1, -1, -1)
        pano_noises = rearrange(pano_noises, 'b m c h w -> (b m) c h w')
        noise = e2p(
            pano_noises,
            cameras['FoV'], cameras['theta'], cameras['phi'],
            (pers_h, pers_w), mode='nearest')
        noise = rearrange(noise, '(b m) c h w -> b m c h w', b=bs, m=m)
        # noise_sample = noise[0, 0, :3]
        # pano_noise_sample = pano_noise[0, 0, :3]
        return pano_noise, noise

    def embed_prompt(self, batch, num_cameras):
        if self.hparams.use_pers_prompt:
            pers_prompt = self.get_pers_prompt(batch)
            pers_prompt_embd = self.encode_text(pers_prompt)
            pers_prompt_embd = rearrange(pers_prompt_embd, '(b m) l c -> b m l c', m=num_cameras)
        else:
            pers_prompt = ''
            pers_prompt_embd = self.encode_text(pers_prompt)
            pers_prompt_embd = pers_prompt_embd[:, None].repeat(1, num_cameras, 1, 1)

        if self.hparams.use_pano_prompt:
            pano_prompt = self.get_pano_prompt(batch)
        else:
            pano_prompt = ''
        pano_prompt_embd = self.encode_text(pano_prompt)
        pano_prompt_embd = pano_prompt_embd[:, None]

        return pers_prompt_embd, pano_prompt_embd

    def training_step(self, batch, batch_idx):
        device = batch['images'].device
        latents = self.encode_image(batch['images'], self.vae)
        b, m, c, h, w = latents.shape

        pano_pad = self.pad_pano(batch['pano'])
        pano_latent_pad = self.encode_image(pano_pad, self.vae)
        pano_latent = self.unpad_pano(pano_latent_pad, latent=True)
        # # test encoded pano latent
        # pano_pad = ((pano_pad[0, 0] + 1) * 127.5).cpu().numpy().astype(np.uint8)
        # pano = ((batch['pano'][0, 0] + 1) * 127.5).cpu().numpy().astype(np.uint8)
        # pano_decode = self.decode_latent(pano_latent, self.vae)[0, 0]

        t = torch.randint(0, self.scheduler.config.num_train_timesteps,
                          (b,), device=latents.device).long()
        pers_prompt_embd, pano_prompt_embd = self.embed_prompt(batch, m)
        pano_noise, noise = self.init_noise(
            b, *pano_latent.shape[-2:], h, w, batch['cameras'], device)

        noise_z = self.scheduler.add_noise(latents, noise, t)
        pano_noise_z = self.scheduler.add_noise(pano_latent, pano_noise, t)
        t = t[:, None].repeat(1, m)

        denoise, pano_denoise = self.mv_base_model(
            noise_z, pano_noise_z, t, pers_prompt_embd, pano_prompt_embd, batch['cameras'],
            batch.get('images_layout_cond'), batch.get('pano_layout_cond'))

        # eps mode
        loss_pers = torch.nn.functional.mse_loss(denoise, noise)
        loss_pano = torch.nn.functional.mse_loss(pano_denoise, pano_noise)
        loss = loss_pers + loss_pano
        self.log('train/loss', loss, prog_bar=False)
        self.log('train/loss_pers', loss_pers, prog_bar=True)
        self.log('train/loss_pano', loss_pano, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        self.eval()
        torch.manual_seed(9999)

        total_loss = 0.0
        count = 0

        stabilized_loader = self.trainer.datamodule.train_stabilized_loader

        with torch.no_grad():
            for batch_idx, batch in enumerate(stabilized_loader):
                batch = self.transfer_batch_to_device(batch, self.device, dataloader_idx=0)
                
                loss = self.training_step(batch, batch_idx)
                total_loss += loss.item()
                count += 1

        avg_loss = total_loss / max(count, 1)
        self.log("train_stabilized", avg_loss, sync_dist=True)
        self.train()  

    @torch.no_grad()
    def forward_cls_free(self, latents, pano_latent, timestep, prompt_embd, pano_prompt_embd, batch, pano_layout_cond=None):
        latents, pano_latent, timestep, cameras, images_layout_cond, pano_layout_cond = self.gen_cls_free_guide_pair(
            latents, pano_latent, timestep, batch['cameras'],
            batch.get('images_layout_cond'), pano_layout_cond)

        noise_pred, pano_noise_pred = self.mv_base_model(
            latents, pano_latent, timestep, prompt_embd, pano_prompt_embd, cameras,
            images_layout_cond, pano_layout_cond)

        noise_pred, pano_noise_pred = self.combine_cls_free_guide_pred(noise_pred, pano_noise_pred)

        return noise_pred, pano_noise_pred

    def rotate_latent(self, pano_latent, cameras, degree=None):
        if degree is None:
            degree = self.hparams.rot_diff
        if degree % 360 == 0:
            return pano_latent, cameras

        pano_latent = super().rotate_latent(pano_latent, degree)
        cameras = cameras.copy()
        cameras['theta'] = (cameras['theta'] + degree) % 360
        return pano_latent, cameras
    

    def parse_prompt_for_identities(self, prompt):
        print("[DEBUG] Original prompt:", prompt)
        pattern = r'(' + '|'.join(map(re.escape, self.valid_ids)) + r')'
        found = re.findall(pattern, prompt)
        mapping = {}
        placeholder_index = 1
        for token in found:
            if token not in mapping:
                mapping[token] = (f'v{placeholder_index}*', f'v{placeholder_index+1}*')
                placeholder_index += 2

        def replace_func(match):
            token = match.group(0)
            if token in mapping:
                return " ".join(mapping[token])
            return token
                
        cleaned_prompt = re.sub(pattern, replace_func, prompt)
        print("[DEBUG] ", cleaned_prompt)
        return cleaned_prompt, mapping


    def inject_identity_embeddings(self, identity_embedding_path, placeholder_tokens):
        fixed_embedding = torch.load(identity_embedding_path).to(self.device)
        
        v1_emb = fixed_embedding[:, 0]
        v2_emb = fixed_embedding[:, 1]
        
        tokens = list(placeholder_tokens)  # e.g. ("v1*", "v2*")
        self.tokenizer.add_tokens(tokens)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        self.text_encoder.get_input_embeddings().weight.data[token_ids[0]] = v1_emb
        self.text_encoder.get_input_embeddings().weight.data[token_ids[1]] = v2_emb

    @torch.no_grad()
    def inference(self, batch):
        raw_prompt = batch['pano_prompt'][0]
        cleaned_prompt, id_mapping = self.parse_prompt_for_identities(raw_prompt)
        batch['pano_prompt'][0] = cleaned_prompt

        for id_token, placeholder_tokens in id_mapping.items():
            print(id_token, placeholder_tokens)
            embedding_path = self.identity_embedding_paths.get(id_token)
            if embedding_path is None:
                print("[WARNING] embedding id does not exist")
            else:
                self.inject_identity_embeddings(embedding_path, placeholder_tokens)

        bs, m = batch['cameras']['height'].shape[:2]
        h, w = batch['cameras']['height'][0, 0].item(), batch['cameras']['width'][0, 0].item()

        equi_h = int(batch['height'][0] // 8)
        equi_w = int(batch['width'][0] // 8)
        device = self.device

        pano_latent, latents = self.init_noise(
            bs, equi_h, equi_w, h//8, h//8, batch['cameras'], device)

        pers_prompt_embd, pano_prompt_embd = self.embed_prompt(batch, m)

        prompt_null = self.encode_text('')[:, None]
        pano_prompt_embd = torch.cat([prompt_null, pano_prompt_embd])
        prompt_null = prompt_null.repeat(1, m, 1, 1)
        pers_prompt_embd = torch.cat([prompt_null, pers_prompt_embd])

        self.scheduler.set_timesteps(self.hparams.diff_timestep, device=device)
        timesteps = self.scheduler.timesteps

        pano_layout_cond = batch.get('pano_layout_cond')

        curr_rot = 0
        for i, t in enumerate(timesteps):
            timestep = torch.cat([t[None, None]]*m, dim=1)

            pano_latent, batch['cameras'] = self.rotate_latent(pano_latent, batch['cameras'])
            curr_rot += self.hparams.rot_diff

            if self.hparams.layout_cond:
                pano_layout_cond = super().rotate_latent(pano_layout_cond)
            else:
                pano_layout_cond = None
            noise_pred, pano_noise_pred = self.forward_cls_free(
                latents, pano_latent, timestep, pers_prompt_embd, pano_prompt_embd, batch, pano_layout_cond)

            latents = self.scheduler.step(
                noise_pred, t, latents).prev_sample
            pano_latent = self.scheduler.step(
                pano_noise_pred, t, pano_latent).prev_sample

        pano_latent, batch['cameras'] = self.rotate_latent(pano_latent, batch['cameras'], -curr_rot)

        images_pred = self.decode_latent(latents, self.vae)
        images_pred = tensor_to_image(images_pred)
        print("[DEBUG inference] images_pred:", 
          f"shape={images_pred.shape}, dtype={images_pred.dtype}")

        pano_latent_pad = self.pad_pano(pano_latent, latent=True)
        pano_pred_pad = self.decode_latent(pano_latent_pad, self.vae)
        pano_pred = self.unpad_pano(pano_pred_pad)
        pano_pred = tensor_to_image(pano_pred)

        # # test encoded pano latent
        # img1 = self.decode_latent(pano_latent, self.vae).squeeze()
        # img1 = np.roll(img1, img1.shape[0]//2, axis=0)
        # img1 = np.roll(img1, img1.shape[1]//2, axis=1)
        # img2 = pano_pred.squeeze()
        # img2 = np.roll(img2, img2.shape[0]//2, axis=0)
        # img2 = np.roll(img2, img2.shape[1]//2, axis=1)

        upsampler = initialize_realesrgan(
            model_name='RealESRGAN_x4plus_anime_6B',
            tile=0,
            gpu_id=0
        )

        pano_pred_squeezed = pano_pred[0, 0]    # shape=(512,1024,3)
        upscaled_pano = upscale_image(pano_pred_squeezed, upsampler)

        # upscaled_images = []
        # b, m = images_pred.shape[:2]
        # for i in range(b):
        #     for j in range(m):
        #         single_img = images_pred[i, j]   # (256,256,3)
        #         upscaled_images.append(upscale_image(single_img, upsampler))

        return images_pred, upscaled_pano

        # return images_pred, pano_pred

    def to01(self, x: torch.Tensor) -> torch.Tensor:
        """
        Converts a tensor in [-1, 1] to [0, 1].
        """
        return (x + 1.) * 0.5

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        device = batch['images'].device
        
        latents = self.encode_image(batch['images'], self.vae)  # shape: (b, m, c, h, w)

        pano_pad = self.pad_pano(batch['pano'])
        pano_latent_pad = self.encode_image(pano_pad, self.vae)
        pano_latent = self.unpad_pano(pano_latent_pad, latent=True)

        b, m, c, h, w = latents.shape
        t = torch.randint(0, self.scheduler.config.num_train_timesteps,
                          (b,), device=device).long()

        pers_prompt_embd, pano_prompt_embd = self.embed_prompt(batch, m)
        pano_noise, noise = self.init_noise(
            b, pano_latent.shape[-2], pano_latent.shape[-1], h, w, batch['cameras'], device
        )

        noise_z = self.scheduler.add_noise(latents, noise, t)
        pano_noise_z = self.scheduler.add_noise(pano_latent, pano_noise, t)
        t = t[:, None].repeat(1, m) 

        denoise, pano_denoise = self.mv_base_model(
            noise_z, pano_noise_z, t, pers_prompt_embd, pano_prompt_embd,
            batch['cameras'], batch.get('images_layout_cond'), batch.get('pano_layout_cond')
        )

        loss_pers = torch.nn.functional.mse_loss(denoise, noise)
        loss_pano = torch.nn.functional.mse_loss(pano_denoise, pano_noise)
        val_loss = loss_pers + loss_pano

        self.log('val/loss_pers', loss_pers, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/loss_pano', loss_pano, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/loss', val_loss, prog_bar=True, on_step=False, on_epoch=True)

        images_pred, pano_pred = self.inference(batch)
        self.log_val_image(
            images_pred, batch['images'], pano_pred, batch['pano'], batch['pano_prompt'],
            batch.get('images_layout_cond'), batch.get('pano_layout_cond')
        )

        lpips_scores = []

        for i in range(b):
            for j in range(m):
                pred_img = images_pred[i, j] 
                gt_img   = batch['images'][i, j]  
                pred_img_torch = torch.from_numpy(pred_img).float()

                if pred_img_torch.ndim == 3 and pred_img_torch.shape[-1] == 3:
                    pred_img_torch = pred_img_torch.permute(2, 0, 1)

                pred_img_torch = pred_img_torch.to(device)
                gt_img_torch   = gt_img.to(device).float() 

                lpips_val = self.lpips_model(
                    pred_img_torch.unsqueeze(0), 
                    gt_img_torch.unsqueeze(0)     
                )
                lpips_scores.append(lpips_val.item())

                pred_img_01 = self.to01(pred_img_torch).clamp(0, 1).cpu()
                gt_img_01   = self.to01(gt_img_torch).clamp(0, 1).cpu()

                pred_pil = transforms.ToPILImage()(pred_img_01)
                gt_pil   = transforms.ToPILImage()(gt_img_01)

                pred_img_resized = self.transform_uint8_scaled(pred_pil)
                pred_img_resized = pred_img_resized.to(device) 
                gt_img_resized   = self.transform_uint8_scaled(gt_pil)
                gt_img_resized = gt_img_resized.to(device)

                self.fid_metric.update(pred_img_resized.unsqueeze(0), real=False)
                self.fid_metric.update(gt_img_resized.unsqueeze(0), real=True)
        
        if lpips_scores:
            avg_lpips_batch = sum(lpips_scores) / len(lpips_scores)
            self._val_lpips.append(avg_lpips_batch)
            # self.log('val/lpips_batch', avg_lpips_batch, on_step=False, on_epoch=True, sync_dist=True)
        else:
            avg_lpips_batch = float('nan')
            # self.log('val/lpips_batch', avg_lpips_batch, on_step=False, on_epoch=True, sync_dist=True)

        return val_loss

    
    def on_validation_epoch_end(self):
        if self._val_lpips:
            epoch_avg_lpips = sum(self._val_lpips) / len(self._val_lpips)
        else:
            epoch_avg_lpips = float('nan')
        self.log('val/lpips_epoch', epoch_avg_lpips, prog_bar=True)

        try:
            epoch_fid = self.fid_metric.compute().item()
        except Exception as e:
            self.print(f"Error computing FID: {e}")
            epoch_fid = float('nan')
        self.log('val/fid_epoch', epoch_fid, prog_bar=True)

        self._val_lpips.clear()
        self.fid_metric.reset()

        if self.trainer and self.trainer.is_global_zero:
            run_id = (
                self.trainer.logger.experiment.id
                if hasattr(self.trainer.logger, "experiment") and self.trainer.logger.experiment
                else "default_run"
            )
            
            runid_path = os.path.join("logs", run_id)
            checkpoint_dir = os.path.join(runid_path, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{self.current_epoch}.ckpt")
            self.trainer.save_checkpoint(checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")



    def inference_and_save(self, batch, output_dir, ext='png'):
        prompt_path = os.path.join(output_dir, 'prompt.txt')
        if os.path.exists(prompt_path):
            return

        _, pano_pred = self.inference(batch)

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"pano.{ext}")
        im = Image.fromarray(pano_pred)
        im.save(path)

        with open(prompt_path, 'w') as f:
            f.write(batch['pano_prompt'][0]+'\n')

    @torch.no_grad()
    @rank_zero_only
    def log_val_image(self, images_pred, images, pano_pred, pano, pano_prompt,
                      images_layout_cond=None, pano_layout_cond=None):
        log_dict = {f"val/{k}_pred": v for k, v in self.temp_wandb_images(
            images_pred, pano_pred, None, pano_prompt).items()}
        log_dict.update({f"val/{k}_gt": v for k, v in self.temp_wandb_images(
            images, pano, None, pano_prompt).items()})
        if images_layout_cond is not None and pano_layout_cond is not None:
            log_dict.update({f"val/{k}_layout_cond": v for k, v in self.temp_wandb_images(
                images_layout_cond, pano_layout_cond, None, pano_prompt).items()})
        self.logger.experiment.log(log_dict)

    def temp_wandb_images(self, images, pano, prompt=None, pano_prompt=None):
        log_dict = {}
        pers = []
        for m_i in range(images.shape[1]):
            pers.append(self.temp_wandb_image(
                images[0, m_i], prompt[m_i][0] if prompt else None))
        log_dict['pers'] = pers

        log_dict['pano'] = self.temp_wandb_image(
            pano[0, 0], pano_prompt[0] if pano_prompt else None)
        return log_dict

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        device = batch['images'].device
        
        images_pred, pano_pred = self.inference(batch)

        self.log_test_image(
            images_pred=images_pred,
            images=batch['images'],
            pano_pred=pano_pred,
            pano=batch['pano'],
            pano_prompt=batch['pano_prompt']
        )

        b, m = batch['images'].shape[:2]
        lpips_scores = []

        for i in range(b):
            for j in range(m):
                pred_img = images_pred[i, j]   # (H, W, C) in numpy
                gt_img   = batch['images'][i, j]  # (C, H, W) in torch [-1,1]
                
                # Convert predicted image to torch, shape (1, C, H, W)
                pred_img_torch = torch.from_numpy(pred_img).float()
                if pred_img_torch.ndim == 3 and pred_img_torch.shape[-1] == 3:
                    # (H, W, C) -> (C, H, W)
                    pred_img_torch = pred_img_torch.permute(2, 0, 1)
                pred_img_torch = pred_img_torch.to(device)
                gt_img_torch   = gt_img.to(device).float()

                lpips_val = self.lpips_model(
                    pred_img_torch.unsqueeze(0),
                    gt_img_torch.unsqueeze(0)
                )
                lpips_scores.append(lpips_val.item())

                pred_img_01 = self.to01(pred_img_torch).clamp(0, 1).cpu()
                gt_img_01   = self.to01(gt_img_torch).clamp(0, 1).cpu()

                pred_pil = transforms.ToPILImage()(pred_img_01)
                gt_pil   = transforms.ToPILImage()(gt_img_01)

                pred_img_resized = self.img_transform_float(pred_pil).to(device)
                gt_img_resized   = self.img_transform_float(gt_pil).to(device)


                pred_img_resized = self.fid_transform_float(pred_pil).to(device)
                gt_img_resized   = self.fid_transform_float(gt_pil).to(device)

                self.fid_metric.update(pred_img_resized_fid.unsqueeze(0), real=False)
                self.fid_metric.update(gt_img_resized_fid.unsqueeze(0),   real=True)
                pred_img_uint8 = self.transform_raw_uint8(pred_pil).to(device)
                gt_img_uint8   = self.transform_raw_uint8(gt_pil).to(device)
                self.kid_metric.update(pred_img_uint8.unsqueeze(0), real=False)
                self.kid_metric.update(gt_img_uint8.unsqueeze(0), real=True)
                self.inception_score.update(pred_img_uint8.unsqueeze(0))


                self.ms_ssim_metric.update(pred_img_resized.unsqueeze(0), gt_img_resized.unsqueeze(0))

        if lpips_scores:
            avg_lpips_batch = sum(lpips_scores) / len(lpips_scores)
            self._test_lpips.append(avg_lpips_batch)
            self.log('test/lpips_batch', avg_lpips_batch, on_epoch=True, sync_dist=True)
        else:
            self.log('test/lpips_batch', float('nan'), on_epoch=True)

        return torch.tensor(0.0, device=device)

    def on_test_epoch_end(self):
        if self._test_lpips:
            epoch_avg_lpips = sum(self._test_lpips) / len(self._test_lpips)
        else:
            epoch_avg_lpips = float('nan')
        self.log('test/lpips_epoch', epoch_avg_lpips, prog_bar=True)

        try:
            epoch_fid = self.fid_metric.compute().item()
        except Exception as e:
            self.print(f"Error computing FID on test set: {e}")
            epoch_fid = float('nan')
        self.log('test/fid_epoch', epoch_fid, prog_bar=True)

        kid_mean, kid_std = self.kid_metric.compute()
        self.log('val/kid_mean_epoch', kid_mean.item(), prog_bar=True)
        self.log('val/kid_std_epoch', kid_std.item(), prog_bar=False)

        ms_ssim_val = self.ms_ssim_metric.compute().item()
        self.log('val/ms_ssim_epoch', ms_ssim_val, prog_bar=True)

        score_mean, score_std = self.inception_score.compute()
        self.log('val/inception_score_mean', score_mean, prog_bar=True)
        self.log('val/inception_score_std', score_std, prog_bar=False)
        self.inception_score.reset()

        self.kid_metric.reset()
        self.ms_ssim_metric.reset()
        self._test_lpips.clear()
        self.fid_metric.reset()

    @torch.no_grad()
    @rank_zero_only
    def log_test_image(self, images_pred, images, pano_pred, pano, pano_prompt):
        log_dict = {f"test/{k}_pred": v for k, v in self.temp_wandb_images(
            images_pred, pano_pred, None, pano_prompt
        ).items()}
        log_dict.update({f"test/{k}_gt": v for k, v in self.temp_wandb_images(
            images, pano, None, pano_prompt
        ).items()})
        self.logger.experiment.log(log_dict)



