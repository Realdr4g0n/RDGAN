import torch
import torch.nn.functional as F
import itertools
import os
import numpy as np
import torchvision
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from options.train_options import TrainOptions
import transforms as transforms
import matplotlib.pyplot as plt
import time
class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=1.0 ,help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            
            parser.add_argument('--lambda_expr', type=float, default = 5.0, help='use expr VGG19 ')


        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A','percept_A','expr_A', 'D_B', 'G_B', 'cycle_B', 'idt_B','percept_B','expr_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netVGG = networks.VGG('VGG19')
            #checkpoint = torch.load(os.path.join('fer_train.t7'))
            self.netVGG.load_state_dict(torch.load('./fer_train.t7'))
            self.netVGG.cuda()
            
        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            #self.criterionExpr = torch.nn.L1Loss() # 19.07.03 
            #self.criterionExpr = networks.GANLoss('fer').to(self.device) #Mse loss
            self.criterionExpr = torch.nn.BCELoss()
            self.criterionPerc = networks.PercLoss(self.gpu_ids)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_expr = self.opt.lambda_expr
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
################################################################################# 19.08.23 perceptual loss 추가
        self.loss_percept_A = self.criterionPerc(self.fake_B,self.real_A)
        self.loss_percept_B = self.criterionPerc(self.fake_A,self.real_B)
################################################################################# 19.07.05 Expression loss 수정 / 
        '''
        self.real_A_tmp = self.real_A.data[0].resize_(3,1,48,48)
        self.real_A_tmp = Variable(self.real_A_tmp,volatile=True)
        self.real_B_tmp = self.real_B.data[0].resize_(3,1,48,48)
        self.real_B_tmp = Variable(self.real_B_tmp,volatile=True)
        '''     
        #print('전 :' ,self.fake_B.data[0].shape)
        #torchvision.utils.save_image(self.fake_B.data[0],'./fuck.jpg')
        ###self.fake_B_tmp = self.fake_B.data[0].view(3,1,48,48) # print 찍어보기 
        #torchvision.utils.save_image(self.fake_B_tmp,'./fuck2.jpg')
        #print('후 :' ,self.fake_B_tmp.shape)
        #print('fake_B_tmp : ' ,self.fake_B_tmp)
        #self.fake_B_tmp = Variable(self.fake_B_tmp,volatile=True) #찍은 값에 맞게 라벨 변형 
        #self.fake_A_tmp = self.fake_A.data[0].resize_(3,1,48,48)
        #self.fake_A_tmp = Variable(self.fake_A_tmp,volatile=True)
        
        #self.loss_expr_A = 0
        #self.loss_expr_B = 0
        self.fake_Bavg = self.netVGG(self.fake_B).view(1, -1).mean(0)
        self.fake_Bscore = F.softmax(self.fake_Bavg)
        
        #self.real_Bavg = self.netVGG(self.real_B).view(1, -1).mean(0)
        #self.real_Bscore = F.softmax(self.real_Bavg)
        
        self.fake_Aavg = self.netVGG(self.fake_A).view(1, -1).mean(0)
        self.fake_Ascore = F.softmax(self.fake_Aavg)
        
        #self.real_Aavg = self.netVGG(self.real_A).view(1, -1).mean(0)
        #self.real_Ascore = F.softmax(self.real_Aavg)
        
        #self.loss_expr_A = self.criterionExpr(self.fake_Bscore,self.real_Bscore) * lambda_expr#19.07.24 Expression loss 수정 
        #self.loss_expr_B = self.criterionExpr(self.fake_Ascore,self.real_Ascore) * lambda_expr#19.07.24 Expression loss 수정 
        self.loss_expr_A = self.criterionExpr(self.fake_Bscore ,torch.cuda.FloatTensor([0,1,0,0,0,0,0])) * lambda_expr#19.07.24 Expression loss 수정
        self.loss_expr_B = self.criterionExpr(self.fake_Ascore ,torch.cuda.FloatTensor([0,0,0,0,1,0,0])) * lambda_expr#19.07.24 Expression loss 수정
        
        
        #self.loss_expr_A = self.criterionExpr(self.netVGG(self.fake_B),self.netVGG(self.real_B)) * lambda_expr #19.07.08 Expression loss 수정 -> L1loss
        #self.loss_expr_B = self.criterionExpr(self.netVGG(self.fake_A),self.netVGG(self.real_A)) * lambda_expr #19.07.08 Expression loss 수정 -> L1loss
        #self.loss_expr_A = self.criterionExpr(self.netVGG(self.fake_B_tmp),self.netVGG(self.real_B_tmp)) #19.07.09 Expression loss 수정
        #self.loss_expr_B = self.criterionExpr(self.netVGG(self.fake_A_tmp),self.netVGG(self.real_A_tmp)) #19.07.09 Expression loss 수정
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_expr_A + self.loss_expr_B + self.loss_percept_A + self.loss_percept_B
################################################################################################################      
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
