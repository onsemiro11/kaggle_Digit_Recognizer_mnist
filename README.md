# kaggle_Digit_Recognizer_mnist

Learn computer vision fundamentals with the famous MNIST data

 
## Goal

: 이 대회의 목표는 손으로 쓴 한 자리 숫자의 이미지를 가져와 그 숫자가 무엇인지 확인하는 것입니다.
테스트 세트의 모든 항목에 대해 올바른 레이블을 예측합니다.

## Metric

: 이 경쟁은 예측의 분류 정확도(정확한 이미지의 비율)로 평가됩니다.
 
마지막 제출파일 : sample.submission.csv
평가용 픽셀 데이터-> 구축한 모델에 구동 -> 나온 예측값 -> label열에 넣어 제출
Label을 가장 잘 예측한 모델이 승자인 대회.

## 1 . Import Libraries & Load Data
- train.csv파일 불러와 라벨 열만 추출하여 라벨 객체로 할당
- 제거한 데이터를 features객체로 할당(학습할 데이터)
- 실제 라벨값은 labels객체로 따로 할당.(학습 데이터의 정답)

## 2 . Data Check

>>> 한쪽 label에 치우쳐져 있지 않고 어느정도 고르게 분포되어 있음을 확인하여, 추후 dataset을 나눌때 무작위로 나눠도 무관하겠구나 판단함.
>>>print 결과 features 데이터셋과 data_test 셋에 결측치 값은 존재하지 않음을 확인함. (출력값 각각 0)

## 3. Exploration 

--임의의 몇개 image 데이터를 확인해보자.--
 
get_image_matrix 함수 생성

이유 : 784개 픽셀 데이터를 그리기 위해서는 (28,28) 크기로 재형성해줘야 하기 때문

-> 검은 배경에 중간에 숫자가 배치되어 있음을 확인.

## 4. Data Augmentation 

기존데이터는 0-255까지 명도를 수치로 표현한 행렬데이터다.
data augmentation에서 회전한 데이터의 value들을 확인해보면, 실수형의 소숫값으로 형성되어있기 때문에 기존데이터도 1도만 변화해주는 과정을 진행하여 동일한 형태의 value를 구성하도록 하였다.
--- rotate 함수로 회전시킨후, image_aug에 변환된 값들을 넣어주고 dataframe형태로 변환시켜 data_test로 할당한다. ---

 
<기존 학습데이터에도 동일하게 진행>
-추가되는 부분은 학습용 데이터이기 때문에 앞에 label열을 insert함수로 추가한다.


<증강 학습데이터에도 동일하게 진행>
-기존데이터는 1도를 회전시켰고 증강데이터에는 45도를 회전시킨다. rotate에서 1을 45로만 변경하여 동일하게 진행한다.

<1도 회전시킨 기존데이터와 45도 회전시킨 증강데이터를 합친다.>
-pandas의 concat함수를 통해 데이터 결합 진행.

##   5. Principal Component Analysis  (PCA 주성분 분석)

: 학습용 데이터와 평가용 데이터에  PCA를 진행하여 784개였던 피처수를 60개로 줄여 과대적합을 예방해보자.

##  6. Model (Cross Validation)

: make_pipeline을 활용해서 최대한 test셋을 건들지 않고 분석을 진행하였다.

<SVM (Support Vector Machine)>

<RandomForestClassifier>
 
<KNeighborsClassifier>
 
<GaussianNB> : 나이브 베이즈 분류는 확실히 일반화 성능이 낮다.
 
<GradientBoostingClassifier>
 
<MLPClassifier>
 
--> 나이브 베이즈에서 가우디안과 그래디언트 부스터, MLP를 써보았지만 SVM보단 낮은 성능을 보임.
--> 많은 모델중에서 SVM이 성능이 가장 좋았음!
(이유)
피처가 많아도 복잡한 결정 경계를 만들어낼 수 있고 다양한 데이터 세트에서 잘 적용하기 때문에 이번 데이터에서 가장 좋은 성능을 볼 수 있었던 것으로 생각됨.
 
##  7. Hyperparameter Tuning

위 다양한 알고리즘중 서포터 백터머신(SVM)이 가장 높은 수치를 보여줬다.
SVM에 그리드 서치를 진행하여 가장 적합한 파라미터를 찾아보자.
 
<gridsearch 진행>
: 가장 높은 평가치를 보여준 SVM을 활용하여 gridsearch를 진행한다.
-> 가장 적합한 파라미터를 확인하고 정도율과 재현율, f1 score값을 확인하자.
 
-결과가 보기 좋게 출력되기 위해 따로 함수를 먼저 생성함.
 
## 8. Prediction

- sample submission에 test.csv 데이터를 예측한 값을 label에 넣어보자.
 
가장 높은 성능을 보여준 파라미터를 입력하여 모델 학습 진행. -> 평가 지표(정확도)도 확인.

##  8 - Apply goodnote image 

직접 굿노트에 쓴 숫자 이미지를 모델에 넣어 예측을 잘하는지 확인해보았다.
 
goodnote 이미지 하나하나를 불러올 때는 PCA를 진행할 수 없기 때문에, pca를 진행하지 않고 모델을 구축하여 예측해야한다.
 
이 결과를 보면 알 수 있듯이, PCA를 하였을때, 더 일반화에 강하다는 것을 확인할 수 있다. pca를 진행한 모델의 testing accuracy는 98%였기 때문이다.

-goodnote의 손글씨 이미지를 가지고 오기위해 opencv를 활용하였고,
현재 모델을 학습시킨 이미지들이 다 검은 화면에 하얀 글씨로 숫자가 그려져 있기 때문에,
cv2.COLOR_BGR2GRAY를 통해 흑백으로 이미지를 변환시켰다.
 
-같은 값의 형태를 만들기 위해서 rotate 1도를 진행해주었다.
 
-모델이 학습한 데이터 크기가 (28,28)이었기 때문에 goodnote 이미지도 cv2.resize를 활용해서 변형해주었다.
 
 
## 결론

다른 참가자의 코드를 참고하고

높은 정확도르 얻기 위해 augmentation을 활용하여 성능 수치를 향상시켰다.

다양한 모델을 적용시켜 각 모델들으 특성으 파악하는데 도움으 주었다.
