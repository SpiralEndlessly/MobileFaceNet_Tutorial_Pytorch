import os
import torch
import torchvision.transforms as transforms
import numpy as np
from face_model import MobileFaceNet
from data_set.dataloader import LFW
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    test_data_path = "C:/test_data"
    TEST_IDS = [1, 3, 5, 7, 9, 601, 603, 605, 607, 609]

    root = 'data_set/lfw'
    file_list = 'data_set/lfw/lfw_pair.txt'
    transform = transforms.Compose([
            transforms.ToTensor(),  
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    dataset = LFW(root, file_list, transform=transform)

    detect_model = MobileFaceNet(latent_size=320).to(device)  # embeding size is 512 (feature vector)
    detect_model.load_state_dict(torch.load('Weights/Retrained', map_location=lambda storage, loc: storage)["net_state_dict"])
    print('MobileFaceNet face detection model generated')

    detect_model.eval()

    for id in TEST_IDS:
        index = (id - 1) // 2

        homomorphic = np.load(os.path.join(test_data_path, f"{id:05d}_out.npy")).squeeze()
        left = detect_model(dataset[index][0].unsqueeze(0)).detach().numpy().squeeze()
        right = detect_model(dataset[index][2].unsqueeze(0)).detach().numpy().squeeze()

        homomorphic_norm = np.sqrt(np.sum(np.power(homomorphic, 2)))
        left_norm = np.sqrt(np.sum(np.power(left, 2)))
        right_norm = np.sqrt(np.sum(np.power(right, 2)))

        homomorphic_score = np.dot(homomorphic / homomorphic_norm, right / right_norm)
        clear_score = np.dot(left / left_norm, right / right_norm)
        self_score = np.dot(left / left_norm, homomorphic / homomorphic_norm)

        print(f"{id:05d}: Clear {clear_score: .3f} \t Enc {homomorphic_score: .3f} \t Self {self_score: .3f}")