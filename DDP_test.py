import torch, torchvision
import torch.nn as nn
import torch.distributed as dist
import torchvision.transforms as transforms
import torch.optim as optim


#input (1,28,28)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv2 = nn.ModuleList()
        self.conv2.append(nn.Sequential(nn.Conv2d(1, 16, 3, stride=2, padding=1),
                                        nn.BatchNorm2d(16),
                                        nn.LeakyReLU(negative_slope=0.2)
        ))
        
        self.conv2.append(nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=1),
                                        nn.BatchNorm2d(32),
                                        nn.LeakyReLU(negative_slope=0.2)
                        ))
        self.conv2.append(nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1),
                                        nn.BatchNorm2d(64),
                                        nn.LeakyReLU(negative_slope=0.2)
        ))
        self.conv2.append(nn.Sequential(nn.Conv2d(64, 1, 3, stride=2),
                                        nn.BatchNorm2d(1),
                                        nn.LeakyReLU(negative_slope=0.2)
        ))
    def forward(self, x):
        for conv_layer in self.conv2:
            x = conv_layer(x)
            
        x = x.view(-1,1)
        return x
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.deconv2 = nn.ModuleList()
        self.deconv2.append(nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,padding=1),
                            nn.BatchNorm2d(32),
                            nn.LeakyReLU()
        ))
        self.deconv2.append(nn.Sequential(nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2,padding=1),
                            nn.BatchNorm2d(16),
                            nn.LeakyReLU()
        ))
        self.deconv2.append(nn.Sequential(nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2,padding=1),
                            nn.BatchNorm2d(1),
                            nn.LeakyReLU()
        ))
    def forward(self, x):
        for layer in self.deconv2:
            x = layer(x)
            
        return x


    
local_rank = 0
dist.init_process_group(backend='nccl', init_method='env://')

disciminator_model = Discriminator()
generator_model = Generator()

torch.cuda.set_device(local_rank)
disciminator_model.cuda(local_rank)
generator_model.cuda(local_rank)

pg1 = dist.new_group(range(dist.get_world_size()))
pg2 = dist.new_group(range(dist.get_world_size()))
disciminator_model = torch.nn.parallel.DistributedDataParallel(disciminator_model, device_ids=[local_rank],
                                                                output_device=local_rank, process_group=pg1)
generator_model = torch.nn.parallel.DistributedDataParallel(generator_model, device_ids=[local_rank],
                                                                output_device=local_rank, process_group=pg2)

# disciminator_model = disciminator_model.train()
# generator_model = generator_model.train()

g_optimizer = optim.Adam(params=generator_model.parameters(), lr=1e-4)
d_optimizer = optim.Adam(params=disciminator_model.parameters(), lr =1e-4)
bcelog_loss = nn.BCEWithLogitsLoss().cuda(local_rank)

train_dataset = torchvision.datasets.MNIST(root='../../data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)                                                              
                                           
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
batch_size = 8
train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=False,
                                           num_workers=4,
                                           pin_memory=True,
                                           sampler=train_sampler) 

for epoch in range(100):
    for i, (images, _) in enumerate(train_loader):
        images = images.cuda(local_rank, non_blocking=True)
        real_tensor = torch.full((batch_size,1), 1, dtype=torch.float32).cuda(local_rank)
        fake_tensor = torch.zeros((batch_size,1), dtype=torch.float32).cuda(local_rank)
        noise_tensor = torch.rand((batch_size, 64, 4, 4))
        gen_image = generator_model(noise_tensor)
        
        d_fake = disciminator_model(gen_image)
        d_real = disciminator_model(images)
        
        d_fake_loss = bcelog_loss(d_fake, fake_tensor)
        d_real_loss = bcelog_loss(d_real, real_tensor)
        
        d_total_loss = d_fake_loss + d_real_loss
        
        g_optimizer.zero_grad()
        d_optimizer.zero_grad()
        
        d_total_loss.backward()
        g_optimizer.step()
        d_optimizer.step()
        print("current epoch: ", epoch)
