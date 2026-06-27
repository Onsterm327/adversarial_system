import torch
import torch.nn as nn
import torch.nn.functional as F
class DownsampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DownsampleLayer, self).__init__()
        self.Conv_BN_ReLU_2=nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.downsample=nn.Sequential(
            nn.Conv2d(in_channels=out_ch,out_channels=out_ch,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self,x):
        """
        :param x:
        :return: out输出到深层，out_2输入到下一层，
        """
        out=self.Conv_BN_ReLU_2(x)
        out_2=self.downsample(out)
        return out,out_2

class UpSampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        # 512-1024-512
        # 1024-512-256
        # 512-256-128
        # 256-128-64
        super(UpSampleLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch*2, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch*2, out_channels=out_ch*2, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU()
        )
        self.upsample=nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_ch*2,out_channels=out_ch,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self,x,out):
        '''
        :param x: 输入卷积层
        :param out:与上采样层进行cat
        :return:
        '''
        x_out=self.Conv_BN_ReLU_2(x)
        x_out=self.upsample(x_out)
        cat_out=torch.cat((x_out,out),dim=1)
        return cat_out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        out_channels=[2**(i+6) for i in range(5)] #[64, 128, 256, 512, 1024]
        #下采样
        self.d1=DownsampleLayer(3,out_channels[0])#3-64
        self.d2=DownsampleLayer(out_channels[0],out_channels[1])#64-128
        self.d3=DownsampleLayer(out_channels[1],out_channels[2])#128-256
        self.d4=DownsampleLayer(out_channels[2],out_channels[3])#256-512
        #上采样
        self.u1=UpSampleLayer(out_channels[3],out_channels[3])#512-1024-512
        self.u2=UpSampleLayer(out_channels[4],out_channels[2])#1024-512-256
        self.u3=UpSampleLayer(out_channels[3],out_channels[1])#512-256-128
        self.u4=UpSampleLayer(out_channels[2],out_channels[0])#256-128-64
        #输出
        self.o=nn.Sequential(
            nn.Conv2d(out_channels[1],out_channels[0],kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0],3,3,1,1),
            nn.Sigmoid(),
            # BCELoss
        )
    def forward(self,x):
        out_1,out1=self.d1(x)
        out_2,out2=self.d2(out1)
        out_3,out3=self.d3(out2)
        out_4,out4=self.d4(out3)
        out5=self.u1(out4,out_4)
        out6=self.u2(out5,out_3)
        out7=self.u3(out6,out_2)
        out8=self.u4(out7,out_1)
        out=self.o(out8)
        return out

class Generator_add_noise(nn.Module):
    def __init__(self):
        super(Generator_add_noise, self).__init__()
        out_channels=[2**(i+6) for i in range(5)] #[64, 128, 256, 512, 1024]
        #下采样
        self.d1=DownsampleLayer(6,out_channels[0])#3-64
        self.d2=DownsampleLayer(out_channels[0],out_channels[1])#64-128
        self.d3=DownsampleLayer(out_channels[1],out_channels[2])#128-256
        self.d4=DownsampleLayer(out_channels[2],out_channels[3])#256-512
        #上采样
        self.u1=UpSampleLayer(out_channels[3],out_channels[3])#512-1024-512
        self.u2=UpSampleLayer(out_channels[4],out_channels[2])#1024-512-256
        self.u3=UpSampleLayer(out_channels[3],out_channels[1])#512-256-128
        self.u4=UpSampleLayer(out_channels[2],out_channels[0])#256-128-64
        #输出
        self.o=nn.Sequential(
            nn.Conv2d(out_channels[1],out_channels[0],kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0],3,3,1,1),
            nn.Sigmoid(),
            # BCELoss
        )
    def forward(self,x):
        out_1,out1=self.d1(x)
        out_2,out2=self.d2(out1)
        out_3,out3=self.d3(out2)
        out_4,out4=self.d4(out3)
        out5=self.u1(out4,out_4)
        out6=self.u2(out5,out_3)
        out7=self.u3(out6,out_2)
        out8=self.u4(out7,out_1)
        out=self.o(out8)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Discriminator_2(nn.Module):
    def __init__(self, num_blocks, num_classes=1):
        super(Discriminator_2, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=2)
        self.linear = nn.Linear(64*BasicBlock.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        # out = F.avg_pool2d(out, 4)
        out = nn.AdaptiveAvgPool2d((1, 1))(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = torch.sigmoid(out)
        return out
    
class BasicBlock_3(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_3, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out)
        return out
    
class Discriminator_3(nn.Module):
    def __init__(self, num_blocks, num_classes=1):
        super(Discriminator_3, self).__init__()
        self.in_planes = 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(32, num_blocks[0], stride=2)
        self.linear = nn.Linear(32, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        layers = []
        layers.append(BasicBlock_3(self.in_planes, planes, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = nn.AdaptiveAvgPool2d((1, 1))(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = torch.sigmoid(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, num_blocks, num_classes=2):
        super(Discriminator, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*BasicBlock.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        out = nn.AdaptiveAvgPool2d((1, 1))(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class conv_block(nn.Module):
	def __init__(self,ch_in,ch_out):
		super(conv_block, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=ch_out,out_channels=ch_out,kernel_size=3,stride=1,padding=1),
			nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True),
		)
		
	def forward(self,x):
		out = self.conv(x)
		return out
	
class up_block(nn.Module):
	def __init__(self,ch_in,ch_out):
		super(up_block, self).__init__()
		self.conv = nn.Sequential(
			nn.Upsample(scale_factor=2),
			nn.Conv2d(in_channels=ch_in,out_channels=ch_out,kernel_size=3,stride=1,padding=1),
			nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True),
		)
		
	def forward(self,x):
		out = self.conv(x)
		return out
	
class U_Net(nn.Module):
	def __init__(self,img_ch=3,output_ch=1):
		super(U_Net, self).__init__()
		self.ndf=64
		self.Maxpool = nn.MaxPool2d(2,2)
		self.conv1 = conv_block(ch_in=img_ch,ch_out=self.ndf)
		self.conv2 = conv_block(ch_in=self.ndf,ch_out=self.ndf * 2)
		self.conv3 = conv_block(ch_in=self.ndf*2,ch_out=self.ndf*2*2)
		self.conv4 = conv_block(ch_in=self.ndf*2*2,ch_out=self.ndf*2*2*2)
		self.conv5 = conv_block(ch_in=self.ndf*2*2*2,ch_out=self.ndf*2*2*2*2)
		
		self.up4 = up_block(ch_in=self.ndf*2*2*2*2,ch_out=self.ndf*2*2*2)
		self.up_conv4 = conv_block(ch_in=self.ndf*2*2*2*2,ch_out=self.ndf*2*2*2)
		
		self.up3 = up_block(ch_in=self.ndf*2*2*2,ch_out=self.ndf*2*2)
		self.up_conv3 = conv_block(ch_in=self.ndf*2*2*2,ch_out=self.ndf*2*2)
		
		self.up2 = up_block(ch_in=self.ndf*2*2,ch_out=self.ndf*2)
		self.up_conv2 = conv_block(ch_in=self.ndf*2*2,ch_out=self.ndf*2)
		
		self.up1 = up_block(ch_in=self.ndf*2,ch_out=self.ndf)
		self.up_conv1 = conv_block(ch_in=self.ndf * 2, ch_out=self.ndf)
		
		self.conv1_1 = conv_block(ch_in=self.ndf,ch_out=output_ch)
		
	def forward(self,x):
		# x [none,3, 256, 256]
		x1 = self.conv1(x) # [none,3,256,256]
		
		x1_ = self.Maxpool(x1) # [none,64,128,128]
		x2 = self.conv2(x1_) # [none,128,128,128]
		
		x2_ = self.Maxpool(x2) # [none,128,64,64]
		x3 = self.conv3(x2_) # [none,256,64,64]
		
		x3_ = self.Maxpool(x3) # [none,256,32,32]
		x4 = self.conv4(x3_) # [none,512,32,32]
		
		x4_ = self.Maxpool(x4)  # [none,512,16,16]
		x5 = self.conv5(x4_)  # [none,1024,16,16]
		
		u4_ = self.up4(x5) # [none,1024,32,32]
		u4 = self.up_conv4(torch.cat([x4,u4_],dim=1)) # [none,512,32,32]
		
		u3_ = self.up3(u4) # [none,512,64,64]
		u3 = self.up_conv3(torch.cat([x3,u3_],dim=1)) # [none,256,64,64]
		
		u2_ = self.up2(u3) # [none,256,128,128]
		u2 = self.up_conv2(torch.cat([x2,u2_],dim=1)) # [none,128,128,128]
		
		u1_ = self.up1(u2) # [none,128,256,256]
		u1 = self.up_conv1(torch.cat([x1,u1_],dim=1)) # [none,64,256,256]
		
		out = self.conv1_1(u1)  # [none,1,256,256]
		out = torch.sigmoid(out)
		return out

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
			nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=ch_out,out_channels=ch_out,kernel_size=3,stride=1,padding=1),
			nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True),
		)
    def forward(self,x):
        out = self.conv(x)
        return out
	
class up_block(nn.Module):
	def __init__(self,ch_in,ch_out):
		super(up_block, self).__init__()
		self.conv = nn.Sequential(
			nn.Upsample(scale_factor=2),
			nn.Conv2d(in_channels=ch_in,out_channels=ch_out,kernel_size=3,stride=1,padding=1),
			nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True),
		)
		
	def forward(self,x):
		out = self.conv(x)
		return out
	
class U_Net(nn.Module):
	def __init__(self,img_ch=3,output_ch=3):
		super(U_Net, self).__init__()
		self.ndf=64
		self.Maxpool = nn.MaxPool2d(2,2)
		self.conv1 = conv_block(ch_in=img_ch,ch_out=self.ndf)
		self.conv2 = conv_block(ch_in=self.ndf,ch_out=self.ndf * 2)
		self.conv3 = conv_block(ch_in=self.ndf*2,ch_out=self.ndf*2*2)
		self.conv4 = conv_block(ch_in=self.ndf*2*2,ch_out=self.ndf*2*2*2)
		self.conv5 = conv_block(ch_in=self.ndf*2*2*2,ch_out=self.ndf*2*2*2*2)
		
		self.up4 = up_block(ch_in=self.ndf*2*2*2*2,ch_out=self.ndf*2*2*2)
		self.up_conv4 = conv_block(ch_in=self.ndf*2*2*2*2,ch_out=self.ndf*2*2*2)
		
		self.up3 = up_block(ch_in=self.ndf*2*2*2,ch_out=self.ndf*2*2)
		self.up_conv3 = conv_block(ch_in=self.ndf*2*2*2,ch_out=self.ndf*2*2)
		
		self.up2 = up_block(ch_in=self.ndf*2*2,ch_out=self.ndf*2)
		self.up_conv2 = conv_block(ch_in=self.ndf*2*2,ch_out=self.ndf*2)
		
		self.up1 = up_block(ch_in=self.ndf*2,ch_out=self.ndf)
		self.up_conv1 = conv_block(ch_in=self.ndf * 2, ch_out=self.ndf)
		
		self.conv1_1 = conv_block(ch_in=self.ndf,ch_out=output_ch)
		
	def forward(self,x):
		# x [none,3, 256, 256]
		x1 = self.conv1(x) # [none,3,256,256]
		
		x1_ = self.Maxpool(x1) # [none,64,128,128]
		x2 = self.conv2(x1_) # [none,128,128,128]
		
		x2_ = self.Maxpool(x2) # [none,128,64,64]
		x3 = self.conv3(x2_) # [none,256,64,64]
		
		x3_ = self.Maxpool(x3) # [none,256,32,32]
		x4 = self.conv4(x3_) # [none,512,32,32]
		
		x4_ = self.Maxpool(x4)  # [none,512,16,16]
		x5 = self.conv5(x4_)  # [none,1024,16,16]
		
		u4_ = self.up4(x5) # [none,1024,32,32]
		u4 = self.up_conv4(torch.cat([x4,u4_],dim=1)) # [none,512,32,32]
		
		u3_ = self.up3(u4) # [none,512,64,64]
		u3 = self.up_conv3(torch.cat([x3,u3_],dim=1)) # [none,256,64,64]
		
		u2_ = self.up2(u3) # [none,256,128,128]
		u2 = self.up_conv2(torch.cat([x2,u2_],dim=1)) # [none,128,128,128]
		
		u1_ = self.up1(u2) # [none,128,256,256]
		u1 = self.up_conv1(torch.cat([x1,u1_],dim=1)) # [none,64,256,256]
		
		out = self.conv1_1(u1)  # [none,1,256,256]
		out = torch.sigmoid(out)
		return out
# model = Generator()
# model.cuda()
# summary(model,(3,64,64))