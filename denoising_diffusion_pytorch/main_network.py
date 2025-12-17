import torch
import torch.nn as nn
from .agression_network import ResBlock
import time

class mainNetwork(nn.Module):
    def __init__(self,diffusion_extractor, pose_network):
        super(mainNetwork,self).__init__()
        self.diffusion_extractor = diffusion_extractor
        pretrained_dict=self.load_pretrained(self.diffusion_extractor)
        self.diffusion_extractor.load_state_dict(pretrained_dict,strict=False)
        for param in self.diffusion_extractor.parameters():
            param.requires_grad = False
        
        self.resblock = ResBlock(
                3,
                1200,
                2
            )
        self.pose_decoder = pose_network    
            
    
    def forward(self, x):
        inputs = self.diffusion_extractor.get_input(x)
        x_rec, features, loss, loss_dict = self.diffusion_extractor.training_step(inputs[:4])
   
        
        
        intermediate_features=self.pose_decoder.extract_DINOv2_feature(x_rec)
#        intermediate_features=self.resblock(x_rec)
#        print('intermediate_features',intermediate_features.shape)
        
        quaternion = self.pose_decoder(intermediate_features)
        img_GT_alloR=inputs[4]
        img_GT_alloR = img_GT_alloR.view(img_GT_alloR.shape[0], -1, *img_GT_alloR.shape[1:]) # ==> BxVqx3x3
        que_GT_alloR_flat = img_GT_alloR.view(img_GT_alloR.shape[0] ,-1)  #==> (BxVq)x3x3

        loss_quaternion = self.pose_decoder.quaternion_loss(quaternion, que_GT_alloR_flat)
        
        prefix = 'train'
        loss_dict.update({f'{prefix}/loss_quaternion': loss_quaternion})
        loss = loss+loss_quaternion
        loss_dict.update({f'{prefix}/loss': loss})
        
        return loss, loss_dict 

     
    def sample(self, x):
        start_time=time.time()
        x_rec_image,x_rec, features=self.diffusion_extractor.sample(cond=x)
        end_time1=time.time()
        intermediate_features=self.pose_decoder.extract_DINOv2_feature(x_rec)
#        intermediate_features=self.resblock(x_rec)
        end_time2=time.time()
        quaternion = self.pose_decoder(intermediate_features)
        end_time3=time.time()
#        print('time1',end_time1-start_time)
#        print('time2',end_time2-end_time1)
#        print('time3',end_time3-end_time2)
#        print('time_all',end_time3-start_time)
        
        return x_rec_image, quaternion

    def load_pretrained(self, model):
        pretrained_dict_path="/root/autodl-tmp/models/step2/diffusion_part/cat/model-10.pt"
        pretrained_dict = torch.load(pretrained_dict_path)['model'] 
        model_dict = model.state_dict()
        
        print('pretrained_dict_path',pretrained_dict_path)

        pretrained_dict_to_load = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict_to_load)
        
        return model_dict