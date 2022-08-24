import onnxruntime as ort
from data_set.dataloader import LFW
import torchvision.transforms as transforms
import numpy as np
import os

if __name__ == '__main__':
    front = ort.InferenceSession("MobileFaceNet_Front.onnx")

    transform = transforms.Compose([
            transforms.ToTensor(),  
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    root = 'data_set/lfw'
    file_list = 'data_set/lfw/lfw_pair.txt'
    dataset = LFW(root, file_list, transform=transform)

    out_root = 'data_set/lfw_latent'

    for i, faces in enumerate(dataset):
        left = front.run(None, {"input": np.expand_dims(faces[0].detach().numpy(), 0)})
        right = front.run(None, {"input": np.expand_dims(faces[2].detach().numpy(), 0)})
        
        np.save(os.path.join(out_root, f"{2*i:05d}.npy"), left[0])
        np.save(os.path.join(out_root, f"{2*i+1:05d}.npy"), right[0])