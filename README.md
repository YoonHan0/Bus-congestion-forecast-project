# Bus-congestion-forecast-project🚌
Intel AI Global Impact Festival에 참여하여 진행한 버스 혼잡도 예측 프로젝트입니다. 일상생활에서 사용하는 대중교통인 버스의 혼잡도를 예측하는 것을 목표로 프로젝트를 진행하였습니다.

</br>

## 💻프로그램 흐름도
<img src="https://user-images.githubusercontent.com/87405950/190971034-5c26d545-55ac-443f-ba59-a24ab35b0e4f.PNG">

</br>

## 🎈주제를 선정한 이유
- 코로나가 장기화 되면서 다양한 나라의 국민들의 안전이 위협을 받고 있습니다. 이런 상황에서 사람이 많이 모이는 대중교통에 문제가 있다고 생각을 하였습니다. 
- 그리고 대중교통문제 중 최우선으로 개선을 바라는 사항이 배차간격을 조절하는 것입니다. 혼잡도를 예측할 수 있다면 버스의 배차 간격을 늘리는 것도 쉽게 해결될 것입니다.

그래서 저희는 버스 내의 승객 수를 계산하고 혼잡도를 예측하는 것을 주제로 선정하였습니다.

</br>

## 🔧Skills / Editer
**Skill** : pandas, numpy, sklearn, matplotlib, seaborn </br>
**Editer** : Jupyter, Vscode, PyCharm </br>
**Language** : Python

</br>

## 🚌프로젝트 설명

- **데이터셋** : 대전시 버스 이용 데이터, 대전시 날씨데이터, 대전시 코로나19 현황 데이터 (2020.12~2022.06)
- **모델** : KFold

전체 데이터 중 10일 만을 Test data로 그 이외는 Train data로 정하여 지도학습 방식으로 진행하였습니다.
**교차검증을 위해서 KFold 모델을 사용**하였고, 프로그램 내에서 **정확도와 재현율 그리고 특성데이터별로 혼잡도에 영향을 얼마나 주는지를 확인하기 위한 히트맵을 출력**하였습니다.

버스 혼잡도에 영향을 주는 데이터(요일, 시간, 휴일여부, 날씨, 코로나현황 등)를 특성데이터로 하고 잔여 승객수를 타겟 데이터로 하여 혼잡도를 예측하였습니다. 

**혼잡도를 보통 / 혼잡 / 매우 혼잡 총 3단계로 나누었습니다.** 이 혼잡도를 이용해 배차간격을 조절하는 프로젝트입니다. 

</br>

## 어려웠던점 / 개선사항
- **어려웠던점😥**</br>
공공데이터를 수집하는 것, 필요한 데이터들을 찾아보는 것에 시간이 오래걸렸습니다. 대부분 버스데이터들은 크기가 너무 크거나 불필요한 데이터들이 많아 필요한 형태로 수정하고 데이터들을 결합하는 것에 어려움이 있었습니다. 

- **개선사항✨**
1. `main.py`에 있는 코드를 모듈화해서 분리하는 식으로 리펙토링을 하면 좋을 것 같습니다.
2. 혼잡도 예측까지는 구현을 완료하였습니다. 추후에 버스 배차간격을 조절하는 서비스로까지 발전시킬 수 있으면 좋겠습니다.

</br>

## 데이터불러온곳
- 🚌 [버스공공데이터](https://www.bigdata-transportation.kr/)
- 🌈 [날씨데이터](https://data.kma.go.kr/cmmn/main.do)
- 😷 [코로나19현황](https://www.data.go.kr/data/15099476/fileData.do)
