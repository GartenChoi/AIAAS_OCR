from local_base import *
from itertools import product
import os
import shutil
import re
import time
import datetime
import pickle
from math import ceil
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.metrics
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
import torchvision.transforms as transforms
import scipy.sparse as sp
from scipy.sparse import csr_matrix, lil_matrix
from gensim.models.keyedvectors import KeyedVectors
from konlpy.tag import Kkma
import docx
from olefile import OleFileIO
from PIL import Image
from pytesseract import *
from tika import parser
import easyocr

DATA_DIR=f'{BASE_DIR}/data'
SAMPLE_DATA_DIR=f'{DATA_DIR}/sample'
TEST_DATA_DIR=f'{DATA_DIR}/test'
SAMPLE_PKL_DATA_DIR=f'{DATA_DIR}/sample_pkl'
TEST_PKL_DATA_DIR=f'{DATA_DIR}/test_pkl'

MODEL_DIR = f'{BASE_DIR}/model'
MODEL_PATH=f'{MODEL_DIR}/classification.pth.tar'

CATEGORIES={
    '지식': {'근저당권 설정계약서', '독점출판허락계약서', '전용실시권 설정계약서', '캐릭터 라이선스 계약서', '공연계약서', '상표 라이선스 계약서', '통역 용역계약서', '전속계약서(가수, 배우)', '번역 용역계약서', '공동기술 개발계약서', '기술이전계약서', '출판권 및 배타적 발행권 설정계약서', '배타적발행권 설정계약서', '관광버스 지입계약서', '저작권 사용계약서', '저작권 양도계약서', '디지털 만화 공급계약서', '출판권 설정계약서', '단순출판허락계약서', '콘텐츠 제휴계약서'},
    '용역,위탁,대행': {'업무제휴계약서', '운송계약서', '컨설팅계약서', '위탁운영계약서', '위탁관리계약서', '급식 위탁공급계약서', '여행계약서', '개발용역계약서', '제작물 공급계약서', '온라인 위탁판매계약서', '위탁교육계약서', '위탁가공계약서', '행사대행계약서', 'PC방 위탁운영계약서', '위촉계약서', '온라인 광고계약서', '조사 용역계약서', '청소용역계약서', '프랜차이즈계약서', '광고대행계약서', '용역계약서', 'OEM 공급계약서', '마케팅 제휴계약서', '수입대행계약서', '금형제작계약서', '경비용역계약서', '경영자문계약서', '위임계약서(민사)', '인테리어계약서', '업무대행계약서', '콘텐츠 공급계약서', '도급계약서', 'PC방 유지보수계약서', '업무위탁계약서', '공동경영계약서', '대리점 계약서', '위탁판매계약서', '유지보수계약서', '홈페이지 개발계약서'},
    '물건': {'공장 매매계약서', '부동산 임대차계약서', '영업양수도계약서', '공장 임대차계약서', '곡류 공급계약서', '상가 임대차계약서', '상가 매매계약서', '물품 구매계약서', '부동산 매매계약서', '아파트 임대차계약서', '자동차 임대계약서', '독점 공급계약서', '기계 매매계약서', '자동차 매매계약서', '자동판매기 임대차계약서', '골재 공급계약서', '매장 임대차계약서', '교환계약서', '농작물 공급계약서', '기계 임대차계약서', '주택 임대차계약서', 'PC 렌탈계약서', '임대차계약서'},
    '부동산': {'공유물 분할합의서', '가등기담보 설정계약서'},
    '금전': {'채무인수계약서', '질권 설정계약서', '양도담보권 설정계약서', '자금지원계약서', '채권양도계약서', '금전소비대차계약서', '연대보증계약서'},
    '건설,공사': {'전기공사계약서', '건설공사 하도급계약서', '설계 용역계약서', '건설공사계약서', '감리용역계약서'},
    '근로,고용': {'근로계약서(아르바이트용)', '근로계약서', '근로계약서(연봉근로자용)', '임원 위임계약서', '비밀유지서약서(입사자용)', '비밀유지서약서(현재 근무자용)', '근로계약서(일용직 근로자용)'},
    '회사운영': {'신주인수계약서', '기업인수 자문계약서', '주주총회 의사록', '투자계약서', '주식매수선택권 부여계약서(Stock Option)', '주주간계약서(SHA)', '주식양수도계약서(SPA)'},
    '기타': {'차용금일부변제', '임대차보증금', '차용금전부변제', '매매대금'}
}
CLASSES=[('물건', '공장 임대차계약서'), ('물건', '농작물 공급계약서'), ('금전', '채권양도계약서'), ('회사운영', '주주총회 의사록'), ('용역,위탁,대행', '대리점 계약서'), ('물건', '주택 임대차계약서'), ('건설,공사', '전기공사계약서'), ('용역,위탁,대행', '위탁관리계약서'), ('근로,고용', '근로계약서(연봉근로자용)'), ('지식', '출판권 및 배타적 발행권 설정계약서'), ('용역,위탁,대행', '도급계약서'), ('용역,위탁,대행', '용역계약서'), ('지식', '번역 용역계약서'), ('용역,위탁,대행', '업무대행계약서'), ('기타', '임대차보증금'), ('용역,위탁,대행', '위탁교육계약서'), ('근로,고용', '비밀유지서약서(현재 근무자용)'), ('물건', '기계 매매계약서'), ('용역,위탁,대행', '위임계약서(민사)'), ('물건', '자동판매기 임대차계약서'), ('용역,위탁,대행', '경영자문계약서'), ('용역,위탁,대행', '제작물 공급계약서'), ('물건', '자동차 임대계약서'), ('근로,고용', '임원 위임계약서'), ('물건', '자동차 매매계약서'), ('금전', '금전소비대차계약서'), ('용역,위탁,대행', '업무제휴계약서'), ('지식', '근저당권 설정계약서'), ('물건', '임대차계약서'), ('용역,위탁,대행', '프랜차이즈계약서'), ('물건', '골재 공급계약서'), ('지식', '저작권 양도계약서'), ('금전', '양도담보권 설정계약서'), ('용역,위탁,대행', 'PC방 유지보수계약서'), ('용역,위탁,대행', '마케팅 제휴계약서'), ('물건', '매장 임대차계약서'), ('부동산', '가등기담보 설정계약서'), ('지식', '단순출판허락계약서'), ('용역,위탁,대행', '위탁판매계약서'), ('용역,위탁,대행', 'PC방 위탁운영계약서'), ('지식', '기술이전계약서'), ('용역,위탁,대행', '급식 위탁공급계약서'), ('회사운영', '주주간계약서(SHA)'), ('금전', '질권 설정계약서'), ('금전', '연대보증계약서'), ('용역,위탁,대행', '업무위탁계약서'), ('용역,위탁,대행', '콘텐츠 공급계약서'), ('용역,위탁,대행', '위탁가공계약서'), ('지식', '디지털 만화 공급계약서'), ('물건', '상가 매매계약서'), ('용역,위탁,대행', '홈페이지 개발계약서'), ('근로,고용', '비밀유지서약서(입사자용)'), ('회사운영', '신주인수계약서'), ('용역,위탁,대행', '광고대행계약서'), ('지식', '전용실시권 설정계약서'), ('용역,위탁,대행', '인테리어계약서'), ('용역,위탁,대행', '수입대행계약서'), ('회사운영', '투자계약서'), ('건설,공사', '건설공사 하도급계약서'), ('근로,고용', '근로계약서(일용직 근로자용)'), ('회사운영', '주식양수도계약서(SPA)'), ('지식', '상표 라이선스 계약서'), ('물건', '영업양수도계약서'), ('건설,공사', '설계 용역계약서'), ('기타', '차용금전부변제'), ('지식', '독점출판허락계약서'), ('근로,고용', '근로계약서(아르바이트용)'), ('용역,위탁,대행', '운송계약서'), ('물건', '물품 구매계약서'), ('지식', '통역 용역계약서'), ('금전', '자금지원계약서'), ('용역,위탁,대행', '행사대행계약서'), ('용역,위탁,대행', '금형제작계약서'), ('물건', '공장 매매계약서'), ('부동산', '공유물 분할합의서'), ('건설,공사', '건설공사계약서'), ('지식', '콘텐츠 제휴계약서'), ('용역,위탁,대행', '여행계약서'), ('용역,위탁,대행', '공동경영계약서'), ('용역,위탁,대행', '컨설팅계약서'), ('물건', '부동산 매매계약서'), ('용역,위탁,대행', '청소용역계약서'), ('용역,위탁,대행', '위촉계약서'), ('지식', '저작권 사용계약서'), ('용역,위탁,대행', 'OEM 공급계약서'), ('기타', '차용금일부변제'), ('물건', '아파트 임대차계약서'), ('용역,위탁,대행', '경비용역계약서'), ('지식', '출판권 설정계약서'), ('지식', '전속계약서(가수, 배우)'), ('용역,위탁,대행', '유지보수계약서'), ('물건', '기계 임대차계약서'), ('지식', '공연계약서'), ('지식', '공동기술 개발계약서'), ('용역,위탁,대행', '개발용역계약서'), ('용역,위탁,대행', '온라인 광고계약서'), ('금전', '채무인수계약서'), ('회사운영', '주식매수선택권 부여계약서(Stock Option)'), ('지식', '배타적발행권 설정계약서'), ('회사운영', '기업인수 자문계약서'), ('지식', '캐릭터 라이선스 계약서'), ('용역,위탁,대행', '조사 용역계약서'), ('물건', '교환계약서'), ('용역,위탁,대행', '온라인 위탁판매계약서'), ('물건', '부동산 임대차계약서'), ('물건', '상가 임대차계약서'), ('근로,고용', '근로계약서'), ('용역,위탁,대행', '위탁운영계약서'), ('기타', '매매대금'), ('지식', '관광버스 지입계약서'), ('물건', 'PC 렌탈계약서'), ('물건', '독점 공급계약서'), ('물건', '곡류 공급계약서'), ('건설,공사', '감리용역계약서')]


CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ','ㅣ']
JONGSUNG_LIST = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ',
                 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
PARSE_DICT = {
    'ㄲ': 'ㄱㄱ', 'ㄸ': 'ㄷㄷ', 'ㅃ': 'ㅂㅂ', 'ㅆ': 'ㅅㅅ', 'ㅉ': 'ㅈㅈ',
    'ㅘ': 'ㅗㅏ', 'ㅙ': 'ㅗㅐ', 'ㅚ': 'ㅗㅣ', 'ㅝ': 'ㅜㅓ', 'ㅞ': 'ㅜㅔ', 'ㅟ': 'ㅜㅣ',
    'ㄳ': 'ㄱㅅ', 'ㄵ': 'ㄴㅈ', 'ㄶ': 'ㄴㅎ', 'ㄺ': 'ㄹㄱ', 'ㄻ': 'ㄹㅁ', 'ㄼ': 'ㄹㅂ', 'ㄽ': 'ㄹㅅ', 'ㄾ': 'ㄹㅌ', 'ㄿ': 'ㄹㅍ', 'ㅀ': 'ㄹㅎ',
    'ㅄ': 'ㅂㅅ'
}
TAG_IMPORTANT={'NN','NNG','NNM','NNP','NNB','VV','VA','VX','VXV','VCN','XR','UN',}

BATCH_SIZE=10
EPOCH_SIZE=20000
LR=0.001
HIDDEN_NUM=64
HIDDEN_CARDINALITY=20
PENULTIMATE = 64
MESSAGE_PASSING_LAYERS = 2
WINDOW_SIZE = 2
DIRECTED = True
USE_MASTER_NODE = True
NORMALIZE = True
DROPOUT = 0.5
PATIENCE = 20
DIMENSION=300
CVFOLD_NUM=5


KKMA = Kkma()
EASYOCR=easyocr.Reader(['ko','en'],gpu=CUDA)

def filename_to_lines(filepath):
    filename=filepath.split('/')[-1]
    extension=filename.split('.')[-1]
    if '.' not in filename or extension in ['txt']:
        return open(filepath, 'r', encoding='utf-8').readlines()
    if extension in ['hwp']:
        return OleFileIO(filepath).openstream('PrvText').read().decode('utf-16').split('\n')
    if extension in ['doc','docx']:
        return [p.text for p in docx.Document(filepath).paragraphs]
    if extension in ['pdf']:
        return parser.from_file(filepath)['content'].split('\n')
    if extension in ['jpg','png','jpeg','bmp','gif','tiff','jfif']:
        easyocr_terms=EASYOCR.readtext(filepath, detail=0)
        tesseract_terms=image_to_string(Image.open(filepath), lang='kor+eng').split('\n')
        return easyocr_terms+tesseract_terms
        # return EASYOCR.readtext(filepath, detail=0)
    else:
        raise ValueError('알려지지 않은 확장자')
