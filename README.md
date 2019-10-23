# RDGAN(Rebalancing Data Generative Adversarial Network)

- RDGAN is solution of **class imbalance** in facial expression recognition datasets
- It will be pusblished in KSC2019 (2019.12.20)
- This project implementated PyTorch and referenced **Jun-Yan Zhu et al (CycleGAN)** (https://github.com/junyanz)

# Proposed model

aa

- FER2013 and RAF_single have class imbalance problem 

bb

- RDGAN reorganized CycleGAN to eliminate class imbalance
- It can be used with FER2013 and RAF_single

cc

- I've added **Expression Discriminator** to Cyclegan to make it appropriate for the class

dd

- **Expression Discriminator** is Pretrained by FER2013 or RAF_single training-set 
- It can be focus on generate correct class

