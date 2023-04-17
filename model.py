import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torchvision.models import vgg16, VGG16_Weights
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from torchvision.models import resnet152, ResNet152_Weights
import torch
voc_cls = {'urban':0,
           'rangeland': 2,
           'forest':3,
           'unknown':6,
           'barreb land':5,
           'Agriculture land':1,
           'water':4}
cls_voc = {0:'urban',
           2:'rangeland',
           3:'forest',
           6:'unknown',
           5:'barreb land',
           1:'Agriculture land',
           4:'water'}
class Resnet152(nn.Module):
  def __init__(self):
    super().__init__()
    num_classes = 50
    self.model_ft = models.resnet152(weights=ResNet152_Weights.DEFAULT)
    num_ftrs = self.model_ft.fc.in_features
    self.model_ft.fc = nn.Linear(num_ftrs, num_classes)
    #print(self.model_ft)
  def forward(self, x):
    x = self.model_ft(x)
    return x


class CNN_base(nn.Module):
  def __init__(self):
    super().__init__()
    self.network = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding = 0),
      nn.BatchNorm2d(32),
      nn.MaxPool2d(2,2),
      nn.ReLU(),
      nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding = 0),
      nn.BatchNorm2d(64),
      nn.Dropout2d(0.5),
      nn.MaxPool2d(2,2),
      nn.ReLU(),
      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding = 0),
      nn.BatchNorm2d(128),
      nn.Dropout2d(0.5),
      nn.MaxPool2d(2,2),
      nn.ReLU(),

      nn.Flatten(),
      nn.Linear(2048,512),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(512,50)
    )
  def forward(self, x):
    return self.network(x)
class VGG16_FCN32(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 7
        self.vggnet = models.vgg16(weights='IMAGENET1K_V1')
        print(self.vggnet.features)
        self.c1 = nn.Conv2d(512, 4096, 7,stride=1,padding=3)#(512,4096,2)
        #self.c1 = nn.Conv2d(512, 4096,2)#(512,4096,2)
        self.r1 = nn.ReLU(inplace=True)
        self.d1 = nn.Dropout(0.5)
        self.c2 = nn.Conv2d(4096,  4096, 1)
        self.r2 = nn.ReLU(inplace=True)
        self.d2 = nn.Dropout(0.5)
        self.score = nn.Conv2d(4096,self.num_classes, 1)
        self.upsample = nn.ConvTranspose2d(self.num_classes,self.num_classes,32,32)#(64,32)
        #self.upsample = nn.ConvTranspose2d(self.num_classes,self.num_classes,64,32)#(64,32)
    def forward(self,x):
        v = self.vggnet.features(x)
        v = self.c1(v)
        v = self.r1(v)
        v = self.d1(v)
        v = self.c2(v)
        v = self.r2(v)
        v = self.d2(v)
        v = self.score(v)
        out = self.upsample(v)
        #print(out.shape) #(B,C,512,512)
        return out#F.interpolate(out, size=512,mode='bilinear', align_corners=True)

class VGG16_FCN16(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 7
        self.vggnet = models.vgg16(weights='IMAGENET1K_V1')
        feats = list(self.vggnet.features.children())
        self.feats = nn.Sequential(*feats[0:16])
        self.feat4 = nn.Sequential(*feats[17:23])
        self.feat5 = nn.Sequential(*feats[24:30])
        self.fcn = nn.Sequential(
            nn.Conv2d(512, 4096, 7,stride=1,padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(4096,  4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.score = nn.Conv2d(4096,self.num_classes, 1)
        self.score_feat4 = nn.Conv2d(512,self.num_classes, 1)
        self.upsample = nn.ConvTranspose2d(self.num_classes,self.num_classes,4,4)#(64,32)
        self.UPsample = nn.ConvTranspose2d(self.num_classes,self.num_classes,8,8)#(64,32)

    def forward(self,x):
        v = self.vggnet.features(x)
        fconn = self.fcn(v)
        score_fconn = self.score(fconn)
        feats = self.feats(x)
        feat4 = self.feat4(feats)
        score_feat4 = self.score_feat4(feat4)
        score_fconn  = self.UPsample(score_fconn)
        #print(score_feat4.size())
        score_fconn += score_feat4
        out = self.upsample(score_fconn)
        return out#F.interpolate(out, size=512,mode='bilinear', align_corners=True)
class SegFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 7
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.pretrained = "nvidia/segformer-b4-finetuned-ade-512-512"
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained(self.pretrained)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            self.pretrained,
            num_labels=self.num_classes,
            id2label=cls_voc,
            label2id=voc_cls,
            ignore_mismatched_sizes=True,
        )
        self.upsample = nn.ConvTranspose2d(self.num_classes,self.num_classes,4,4)#(512,512)
        #print(self.model.config)
    def forward(self,images,masks):
        inputs = self.feature_extractor(images=list(images), segmentation_maps=list(masks),return_tensors="pt")
        inputs.to(self.device)
        #masks = masks.to(self.device)#(N,512,512)(N,d1,d2)
        outputs = self.model(inputs.pixel_values,inputs.labels)#(N,7,512,512)(N,C,d1,d2)
        logit = F.interpolate(outputs.logits, size=512,mode='bilinear', align_corners=True)
        return logit

