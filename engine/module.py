import torch, pdb, os
from sinabs import SNNAnalyzer
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import ModelSummary

from .loss import YoloLoss, EuclidianLoss, SpeckLoss, BboxLoss
from .models.spiking.lpf import LPFOnline
from .models.utils import get_spiking_threshold_list, compute_output_dim

class EyeTrackingModelModule(pl.LightningModule):
    def __init__(self, model, dataset_params, training_params):
        super().__init__()
        self.model = model.to(self.device)
        self.dataset_params = dataset_params
        self.training_params = training_params 

        # Input
        self.num_bins = dataset_params["num_bins"]
        self.input_channel = dataset_params["input_channel"]
        self.img_width = dataset_params["img_width"]
        self.img_height = dataset_params["img_height"]

        # Output 
        self.num_classes = training_params["num_classes"]
        self.num_boxes = training_params["num_boxes"]

        # learning rate
        self.lr_model = training_params["lr_model"] 

        # Initialize Low Pass Filter (LPF) 
        if self.training_params["arch_name"] =="retina_snn":
            self.model_lpf = LPFOnline(
                initial_scale=training_params["lpf_init"],
                tau_mem=training_params["lpf_tau_mem_syn"][0],
                tau_syn=training_params["lpf_tau_mem_syn"][1],
                path_to_image=os.path.join(training_params["out_dir"]),
                num_channels=compute_output_dim(training_params),
                kernel_size=min(training_params["lpf_kernel_size"], self.num_bins),
                train_scale=training_params["lpf_train"],
            ).to(self.device)

        # Loss initialization
        if self.training_params["arch_name"] == "3et":
            self.error = EuclidianLoss()
        elif self.training_params["arch_name"] == "retina_ann":
            self.error = BboxLoss(dataset_params, training_params) 
        elif self.training_params["arch_name"] == "retina_snn":
            self.error = YoloLoss(dataset_params, training_params)  
            
        if self.training_params["arch_name"] =="retina_snn":
            spiking_thresholds = get_spiking_threshold_list(self.model.spiking_model)
            self.speck_loss = SpeckLoss(
                sinabs_analyzer=SNNAnalyzer(self.model.spiking_model),
                spiking_thresholds=spiking_thresholds,
                synops_lim=training_params["synops_lim"],
                firing_lim=training_params["firing_lim"], 
                w_fire_loss=training_params["w_fire_loss"], 
                w_input_loss=training_params["w_input_loss"], 
                w_synap_loss=training_params["w_synap_loss"]
            )

    def forward(self, x):  
        
        if self.training_params["arch_name"][:6] =="retina":
            if self.training_params["arch_name"] =="retina_snn":
                B, T, C, H, W = x.shape

                # 1) 展平成 (B*T, C, H, W) 喂给 SNN
                x = x.view(B * T, C, H, W)

                # 2) SNN 前向
                outputs = self.model.spiking_model(x)   # (B*T, D)

                # 3) 还原时间维度 + 调整成 Conv1d 需要的形式
                outputs = outputs.view(B, T, -1)        # (B, T, D)
                outputs = outputs.permute(0, 2, 1)      # (B, D, T)

                # 4) LPF：在时间轴 T 上做 1D 卷积
                # 如果你的 LPF 接口允许传 padding_mode，这里用 "zeros" 或 "repeat"
                # 避免跨 batch 的“past”状态互相污染（因为训练时 shuffle 了）
                outputs = self.model_lpf(outputs)       # 仍然 (B, D, T)

                # 5) 再转回 (B*T, D)，方便后面统一算 loss
                outputs = outputs.permute(0, 2, 1)      # (B, T, D)
                outputs = outputs.reshape(B * T, -1)    # (B*T, D)
                return outputs   
        return self.model(x) 

    def on_train_start(self): 
        self.model = self.model.to(self.device)
        
    def training_step(self, batch, batch_idx):
        data, labels, _, _ = batch

        #print("data shape in training_step:", data.shape)
        
        data = data.to(self.device)
        labels = labels.to(self.device)

        # 1) 对 retina_snn：把 labels 从 (B, T, D) 展平到 (B*T, D)
        if self.training_params["arch_name"] == "retina_snn":
            # labels: (B, T, label_dim)
            def reset_if_possible(m):
                if hasattr(m, "reset_states"):
                    m.reset_states()
                elif hasattr(m, "reset_state"):
                    m.reset_state()

            for m in self.model.spiking_model.modules():
                reset_if_possible(m)
           

            B, T = data.shape[:2]
            # print("labels shape:", labels.shape, "numel:", labels.numel(), "B*T:", B*T)

            # labels: (B,4,4,5) -> (B,T,4,4,5)
            labels = labels.unsqueeze(1).expand(B, T, 4, 4, 5)

            # -> (B*T, 80) 以匹配 outputs (B*T, 80)
            labels = labels.reshape(B*T, -1)

        # 在 outputs = self.forward(data) 之前


            # ✅ 这里：forward 之前 reset SNN states


        # 2) forward：对 retina_snn 应该返回 (B*T, out_dim)
        outputs = self.forward(data)

        # ⚠️ 3) 不要在这里再 outputs = self.model_lpf(outputs)
        #     （LPF 应该已经在 forward 里做过了）


        # 4) loss
        loss_dict, output_dict = self.compute_loss(outputs, labels)
        output_dict["loss_dict"] = loss_dict
        output_dict["loss"] = loss_dict["total_loss"]

        self.log("train_loss", loss_dict["total_loss"], prog_bar=True)


        return output_dict

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        data, labels, _, _ = batch
        data = data.to(self.device)
        labels = labels.to(self.device)

        if self.training_params["arch_name"] == "retina_snn":
            # ✅ reset SNN states（和 train 保持一致）
            def reset_if_possible(m):
                if hasattr(m, "reset_states"):
                    m.reset_states()
                elif hasattr(m, "reset_state"):
                    m.reset_state()

            for m in self.model.spiking_model.modules():
                reset_if_possible(m)

            B, T = data.shape[:2]

            # ✅ 和 training_step 完全一致的 labels 处理
            # (B,4,4,5) -> (B,T,4,4,5) -> (B*T,80)
            labels = labels.unsqueeze(1).expand(B, T, 4, 4, 5)
            labels = labels.reshape(B * T, -1)

        outputs = self.forward(data)  # retina_snn: (B*T, 80)

        loss_dict, output_dict = self.compute_loss(outputs, labels)
        output_dict["loss_dict"] = loss_dict
        output_dict["loss"] = loss_dict["total_loss"]

        self.log("val_loss", loss_dict["total_loss"], prog_bar=True)
        return output_dict




    def configure_optimizers(self):
        # Optimizer setup
        param_list = [{"params": self.model.parameters(), "lr": self.lr_model}]
        if self.training_params["arch_name"] =="retina_snn":
            param_list += [
                {"params": self.model_lpf.tau_mem, "lr": self.training_params["lr_model_lpf_tau"]},
                {"params": self.model_lpf.tau_syn, "lr": self.training_params["lr_model_lpf_tau"]},
                {"params": self.model_lpf.scale_factor, "lr": self.training_params["lr_model_lpf"]},
            ]

        if self.training_params["optimizer"] == "Adam": 
            self.optimizer = torch.optim.Adam(param_list, lr=self.lr_model)
        elif self.training_params["optimizer"] == "SGD":
            self.optimizer = torch.optim.SGD(param_list, lr=self.lr_model, momentum=0.9)
        else:
            raise NotImplementedError

        # Scheduler setup
        if self.training_params["scheduler"] == "StepLR":
            self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.8)
        elif self.training_params["scheduler"] == "ReduceLROnPlateau":
            self.scheduler = ReduceLROnPlateau(self.optimizer, "min", patience=5)
        else:
            raise NotImplementedError

        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

    def compute_loss(self, outputs, labels):
        S = self.training_params["SxS_Grid"]   # 4
        Bx = self.training_params["num_boxes"] # 1
        W = self.training_params["bbox_w"]     # 5

        # outputs: (N, 80) -> (N, 4, 4, 1, 5)
        outputs = outputs.view(-1, S, S, W)
        labels  = labels.view(-1, S, S, W)

        loss_dict = {}
        output_dict = {}

        loss_dict.update(self.error(outputs, labels))
        output_dict["memory"] = self.error.memory 

        # Speck Loss
        if self.training_params["arch_name"] =="retina_snn":
            loss_dict.update(self.speck_loss())

        # Total loss
        loss_dict["total_loss"] = sum(loss_dict.values())

        # compute_loss 末尾，在 return 前加：
        # for k, v in loss_dict.items():
        #     if torch.is_tensor(v):
        #         print("loss part", k, "device", v.device)
        self.log("val_dist_norm", loss_dict.get("distance_loss", 0), prog_bar=True)

        return loss_dict, output_dict
