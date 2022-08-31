import onnxruntime as ort
from data_set.dataloader import LFW
import torchvision.transforms as transforms
import numpy as np
import os

if __name__ == '__main__':
    front = ort.InferenceSession("MobileFaceNet_FrontShallow.onnx")
    back = ort.InferenceSession("MobileFaceNet_BackShallow.onnx")

    transform = transforms.Compose([
            transforms.ToTensor(),  
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    root = 'data_set/lfw'
    file_list = 'data_set/lfw/lfw_pair.txt'
    dataset = LFW(root, file_list, transform=transform)

    out_root = 'data_set/lfw_latent_shallow'

    for j in range(4):
        for i in range(300):
            index = j*300 + i

            out_data = front.run(None, {"input": np.expand_dims(dataset[index][0].detach().numpy(), 0)})[0]
            out_verify = front.run(None, {"input": np.expand_dims(dataset[index][0].detach().numpy(), 0)})[0]
            out_verify = back.run(None, {"input": out_verify})[0]
            np.save(os.path.join(out_root, f"data/{4*i + j:05d}.npy"), out_data)
            np.save(os.path.join(out_root, f"verification/{4*i + j:05d}.npy"), out_verify)