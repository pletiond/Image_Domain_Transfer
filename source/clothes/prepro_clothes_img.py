import cv2
from tqdm import tqdm
import numpy as np
import os
from glob import glob

def get_dir_lists(dataset_path='./img/'):
    temp = []

    for dirname, dirnames, filenames in os.walk(dataset_path):
        temp.append(dirnames)
        break

    temp = temp[0]

    lists = sorted([temp[i] for i in range(len(temp)) if 'T-Shirt' in temp[i] or 'Tee' in temp[i] \
                    or 'Trouser' in temp[i] or 'pants' in temp[i] \
                    or 'Pullover' in temp[i] or 'Hoodie' in temp[i] or 'Jacket' in temp[i] \
                    or 'Dress' in temp[i] \
                    or ('Coat' in temp[i] and 'Coated' not in temp[i])])

    return lists


def get_cropped_images(lists, dataset_path='./img/', length=289224):
    f = open(dataset_path + 'list_bbox.txt')
    f.readline()
    f.readline()

    lst = []

    for i in tqdm(range(length - 3)):
        l = f.readline()
        l = l.split(' ')
        l = [l[n] for n in range(len(l)) if l[n] is not '']
        l[0] = l[0].replace('img/', '')
        l[0] = l[0].split('/')  # [Directory Name, Image Name]
        l[4] = l[4].replace('\n', '')

        if l[0][0] in lists:
            image = cv2.imread(dataset_path + l[0][0] + '/' + l[0][1])
            image = np.array(image[int(l[2]):int(l[4]), int(l[1]):int(l[3])])
            image = cv2.resize(image, (64, 64))
            lst.append([l[0], image])

    return lst


def save_list_images(list, save_path='./crop-images/'):
    filename = ''
    stats =[0, 0, 0, 0, 0]
    for i in tqdm(range(len(list))):
        if 'T-Shirt' in list[i][0][0] or 'Top' in list[i][0][0] or 'Tee' in list[i][0][0]:
            filename = save_path + 'Shirt_' + str(i) + '.jpg'
            stats[0] += 1
        elif 'Trouser' in list[i][0][0] or 'pants' in list[i][0][0]:
            filename = save_path + 'Trouser_' + str(i) + '.jpg'
            stats[1] += 1
        elif 'Pullover' in list[i][0][0] or 'Hoodie' in list[i][0][0]:
            filename = save_path + 'Pullover_' + str(i) + '.jpg'
            stats[2] += 1
        elif 'Dress' in list[i][0][0]:
            filename = save_path + 'Dress_' + str(i) + '.jpg'
            stats[3] += 1
        elif 'Coat' in list[i][0][0] or 'Jacket' in list[i][0][0]:
            filename = save_path + 'Coat_' + str(i) + '.jpg'
            stats[4] += 1

        cv2.imwrite(filename, list[i][1])

    print(stats)

def get_shoes_files(dir = './Shoes/', save_path = './crop-images/'):
    result = [y for x in os.walk(dir) for y in glob(os.path.join(x[0], '*.jpg'))]
    print(len(result))
    i = 0

    for item in result:

        image = cv2.imread(item)
        image = cv2.resize(image, (64, 64))
        filename = save_path + 'Shoes_' + str(i) + '.jpg'
        cv2.imwrite(filename, image)

        i += 1




lists = get_dir_lists()
cropped = get_cropped_images(lists, dataset_path='./img/')
save_list_images(cropped, save_path='./crop-images/')
get_shoes_files()