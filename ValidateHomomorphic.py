import os
import torch
import torchvision.transforms as transforms
import numpy as np
from face_model import MobileFaceNet
from data_set.dataloader import LFW
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import onnxruntime

if __name__ == '__main__':
    test_data_path = "C:/test_data"
    TEST_IDS = [1, 3, 5, 7, 9, 601, 603, 609]

    root = 'data_set/lfw'
    file_list = 'data_set/lfw/lfw_pair.txt'
    transform = transforms.Compose([
            transforms.ToTensor(),  
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    dataset = LFW(root, file_list, transform=transform)

    detect_model = MobileFaceNet(latent_size=320).to(device)  # embeding size is 512 (feature vector)
    detect_model.load_state_dict(torch.load('Weights/Retrained', map_location=lambda storage, loc: storage)["net_state_dict"])
    print('MobileFaceNet face detection model generated')

    frontend = onnxruntime.InferenceSession("MobileFaceNet_Front.onnx")
    backend = onnxruntime.InferenceSession("MobileFaceNet_ReducedBack.onnx")

    detect_model.eval()

    print(f"              Clear |     TenSEAL     |     Concrete")

    for id in TEST_IDS:
        index = (id - 1) // 2

        left_image = dataset[index][0].unsqueeze(0).detach().numpy()
        right_image = dataset[index][2].unsqueeze(0).detach().numpy()

        tenseal = np.load(os.path.join(test_data_path, f"{id:05d}_ten.npy")).squeeze()
        concrete = np.load(os.path.join(test_data_path, f"{id:05d}_con.npy")).squeeze()
        #left = detect_model(dataset[index][0].unsqueeze(0)).detach().numpy().squeeze()
        #right = detect_model(dataset[index][2].unsqueeze(0)).detach().numpy().squeeze()

        left = frontend.run(None, {"input": left_image})[0]
        right = frontend.run(None, {"input": right_image})[0]

        left = backend.run(None, {"input": left})[0].squeeze()
        right = backend.run(None, {"input": right})[0].squeeze()

        tenseal_norm = np.sqrt(np.sum(np.power(tenseal, 2)))
        concrete_norm = np.sqrt(np.sum(np.power(concrete, 2)))
        left_norm = np.sqrt(np.sum(np.power(left, 2)))
        right_norm = np.sqrt(np.sum(np.power(right, 2)))


        clear_score = np.dot(left / left_norm, right / right_norm)

        tenseal_score = np.dot(tenseal / tenseal_norm, right / right_norm)
        tenseal_similarity = np.dot(left / left_norm, tenseal / tenseal_norm)
        concrete_score = np.dot(concrete / concrete_norm, right / right_norm)
        concrete_similarity = np.dot(left / left_norm, concrete / concrete_norm)

        print(f"{id:05d}-{id+1:05d}: {clear_score: .3f} | {tenseal_score: .3f} ({tenseal_similarity: .3f}) | {concrete_score: .3f} ({concrete_similarity: .3f})")