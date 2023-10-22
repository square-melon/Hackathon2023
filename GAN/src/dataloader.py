import os
import torch.utils.data
import PIL.Image as Image
import re
from torchvision import transforms

class ImageLoader(torch.utils.data.Dataset):
    def __init__(self, pic_path, img_res):
        self.pic_path = pic_path
        self.img_res = img_res
        assert os.path.isdir(pic_path)

        self.all_file = []
        for file in os.listdir(self.pic_path):
            try:
                im = Image.open(os.path.join(self.pic_path, file)).convert('RGB')
                self.all_file.append(file)
            except:
                continue

        self.file_num = len(self.all_file)

    def __len__(self):
        return self.file_num

    def __getitem__(self, index):
        file_name = self.all_file[index]
        with Image.open(os.path.join(self.pic_path, file_name)).convert('RGB') as im:
            transformer = transforms.Compose([
                transforms.CenterCrop((min(im.size), min(im.size))),
                transforms.Resize(self.img_res),
                transforms.ToTensor()
            ])
            img = transformer(im)
            return img