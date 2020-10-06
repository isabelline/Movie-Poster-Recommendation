from os import listdir
from os.path import isfile, join
import numpy as np
from PIL import Image
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import io
from google.cloud import vision
from google.cloud.vision import types
import numpy as np
from random import shuffle
import pandas as pd


def get_index(label,label_ids):
	"""
    label_ids 내 label의 index(label이 있는 순서 위치)를 탐지
	:param label: vision api의 19985개의 라벨 중 하나
	:param label_ids: vision api의 19985개의 라벨 전체 리스트
	:return: label 의 label_ids 내 인덱스
	"""
	for idx, label_id in enumerate(label_ids):
		if label == label_id:
			return idx



def get_feature_from_anno_object(onlyfiles, label_ids):
	"""
    캡쳐한 이미지 리스트에 대해 vision api의 객체 탐지 작성 진행
	:param onlyfiles: 이미지 파일 이름 리스트
	:param label_ids: vision api의 19985개 라벨의 리스트
	:return: 객체탐지로 만들어진 피쳐 벡터와 와 등장한 라벨의 Counter 딕셔너리
	"""
	print("Making features from " + str(len(onlyfiles)) + " files....")

    # 빈 피쳐 벡터 생성
	feature = np.zeros([len(onlyfiles), len(label_ids)], dtype='float32')

	label_count = dict()

	for file_idx, file in enumerate(onlyfiles):
        # vision api 클라이언트 연결 (사전에 api key 있어야 연결 가능)
		client = vision.ImageAnnotatorClient()
		file_name = file

		with io.open(file_name, 'rb') as image_file:
			content = image_file.read()

		image = types.Image(content=content)

		label_detection_feature = {
			'type': vision.enums.Feature.Type.LABEL_DETECTION, 'max_results': 100}

		request_features = [label_detection_feature]

        # 객체 탐지 API 호출
		response = client.object_localization(image=image)

        # 결과 라벨 리스트
		annotations = response.localized_object_annotations

		label_score = []
		cnt = 0

        # vision api의 라벨링 결과 리스트 annotations 파싱하여 피쳐 생성
		for anno in annotations:
			label = anno.mid
			score = anno.score
			if label[1] == 'm':
				label_score.append([label, score])
				label_index = get_index(label)
				feature[file_idx][label_index] = score
				cnt += 1
                # 여태껏 등장한 라벨 개수 카운팅
				if not label in label_count:
					label_count[label] = 0
				label_count[label] += 1

		print("Total " + str(cnt) + " labels annotated from " + file)

	print("Annotated total {} unique labels".format(len(label_count)))

	return feature, label_count



def get_feature_from_anno(onlyfiles, label_ids):
	"""
    캡쳐한 이미지 파일 리스트에 대해 Vision api 이미지 라벨링 진행
	:param onlyfiles: 캡펴한 이미지의 파일 이름 리스트
	:param label_ids: Vision api의 19985개의 라벨 리스트
	:return: 이미지 라벨링으로 만들어진 피쳐와 등장 라벨의 개수를 센 Counter 딕셔너리
	"""
	print("Making features from " + str(len(onlyfiles)) + " files....")
	feature = np.zeros([len(onlyfiles), len(label_ids)], dtype='float32')

	label_count = dict()

	for file_idx, file in enumerate(onlyfiles):
        # vision api 클라이언트 연결
		client = vision.ImageAnnotatorClient()
		file_name =  file

        # 이미지 읽어오기
		with io.open(file_name, 'rb') as image_file:
			content = image_file.read()

		image = types.Image(content=content)

		label_detection_feature = {
			'type': vision.enums.Feature.Type.LABEL_DETECTION, 'max_results': 100}
		request_features = [label_detection_feature]

        # 이미지 라벨링 vision api 호출
		response = client.annotate_image(
			{'image': image, 'features': request_features})
		annotation_text = response.label_annotations

		label_score = []
		cnt = 0

        # vision api의 라벨링 결과
		for anno in annotation_text:
            # 라벨 이름과 점수 추출
			label = anno.mid
			score = anno.score

            # 라벨 ID에 'm'이 있어야 정상 라벨
			if label[1] == 'm':
				label_score.append([label, score])
				label_index = get_index(label)
				feature[file_idx][label_index] = score
				cnt += 1
                # 등장한 라벨의 개수를 카운팅
				if not label in label_count:
					label_count[label] = 0
				label_count[label] += 1

		print("Total " + str(cnt) + " labels annotated from " + file)

	print("Annotated total {} unique labels".format(len(label_count)))

	return feature, label_count


def get_all_images():
	"""
    2019년 영화 캡쳐 이미지를 각각 다른 폴더에 보관했는데, 이미지를 모두 읽어오기
	:return: 2019년 영화 중 캡쳐 이미지의 파일 이름 리스트
	"""
    # 2019년 영화 캡쳐 이미지가 있는 파일
	root = "drive/My Drive/trailor/"
	folders = []
	all_files = []
    # 영화 마다 1 ~ 60의 폴더에 캡쳐 이미지 각각 보관
	for i in range(1,61):
	  folders.append(root+ str(i)+"/")
	for folder in folders:
		mypath = folder
        # 폴더 내의 이미지 파일 모두
		onlyfiles = [join(mypath, f) for f in listdir(mypath) if
							   isfile(join(mypath, f)) and f[-3:] == "jpg"]
		all_files.extend(onlyfiles)

	print("Total file num  " +str(len(all_files)))
	return all_files


def shuffle_array(list1, list2):
	"""
    2개의 리스트의 쌍을 유지하면서 순서 섞는 함수
	:param list1: 리스트
	:param list2: 리스트
	:return: 두 리스트의 쌍은 유지하면서 순서가 섞인 2개의 리스트
	"""
	print("Shuffling....")
	list1_shuf = []
	list2_shuf = []
	index_shuf = list(range(len(list1)))
	shuffle(index_shuf)
	for i in index_shuf:
		list1_shuf.append(list1[i])
		list2_shuf.append(list2[i])
	return list1_shuf, list2_shuf

def make_feature():
    """
    각 이미지를 19985개의 벡터로 나타내어 이미지 라벨링과 객체 검출로 만들어진 벡터를 합한 뒤 모든 이미지 벡터를 합친 최종 피쳐 만들기
    :rtype: 학습에 사용할 최종 피쳐, (이미지 개수 x 19985) 크기의 행렬
    """
    # 라벨 정보가 있는 파일
    df = pd.read_csv("drive/My Drive/class-descriptions.csv", encoding='utf-8', header=None)
    label_ids = list(df[0])
    print("Total label number is {}".format(str(len(label_ids))))

    onlyfiles = get_all_images()

    # 이미지 라벨링 진행
    feature, label_count = get_feature_from_anno(onlyfiles, label_ids)

    # 이미지 객체 검출 진행
    feature_two, label_count_two = get_feature_from_anno_object(onlyfiles, label_ids)

    # 위 2개의 피쳐를 합한 것을 최종 피쳐로 사용
    feature_new = feature + feature_two

    return feature_new

