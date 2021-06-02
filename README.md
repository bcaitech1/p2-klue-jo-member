

<img width="194" alt="image-20210602142343433" src="https://user-images.githubusercontent.com/68108633/120448574-a985ef00-c3c6-11eb-99a4-5dd01401b980.png">





# 문장내 개체관 관계 추출

P-stage 2 KLUE가 끝이 났다.

이번 stage에서는 리더보드 순위를 올리는데에만 집중하기 보다는 다양한 task를 써보고 원리를 이해하고 결과를 토론계시판에 꼭 조금이라도 공유하는 방식으로 하기로 마음먹었었습니다.

<!-- more -->



## Overview

- **문제정의** : 문장의 단어(Entity)에 대한 속성과 관계를 예측하라. 

- Input data : 9000개의 train data, 1000개의 test data

  <img width="729" alt="image-20210423130448198" src="https://user-images.githubusercontent.com/68108633/120505156-aeb36000-c3ff-11eb-84b7-ec960592a3ef.png">

  우리가 빼내야 할 column : sentence, entities, place of entities

  

- **Output**

  총 42개의 class를 예측 

  ```python
  {'관계_없음': 0, '인물:배우자': 1, '인물:직업/직함': 2, '단체:모회사': 3, '인물:소속단체': 4, '인물:동료': 5, '단체:별칭': 6, '인물:출신성분/국적': 7, '인물:부모님': 8, '단체:본사_국가': 9, '단체:구성원': 10, '인물:기타_친족': 11, '단체:창립자': 12, '단체:주주': 13, '인물:사망_일시': 14, '단체:상위_단체': 15, '단체:본사_주(도)': 16, '단체:제작': 17, '인물:사망_원인': 18, '인물:출생_도시': 19, '단체:본사_도시': 20, '인물:자녀': 21, '인물:제작': 22, '단체:하위_단체': 23, '인물:별칭': 24, '인물:형제/자매/남매': 25, '인물:출생_국가': 26, '인물:출생_일시': 27, '단체:구성원_수': 28, '단체:자회사': 29, '인물:거주_주(도)': 30, '단체:해산일': 31, '인물:거주_도시': 32, '단체:창립일': 33, '인물:종교': 34, '인물:거주_국가': 35, '인물:용의자': 36, '인물:사망_도시': 37, '단체:정치/종교성향': 38, '인물:학교': 39, '인물:사망_국가': 40, '인물:나이': 41} 
  ```

- **평가 Metric**

  <img width="707" alt="image-20210426120429513" src="https://user-images.githubusercontent.com/68108633/120505236-c12d9980-c3ff-11eb-941f-841273723011.png">

<br/>

## EDA

이번 자연어 task의 경우에는 데이터도 적고 image task에 비해 복잡한 eda가 필요한것 같지는 않았습니다

<img width="1061" alt="image-20210603020606969" src="https://user-images.githubusercontent.com/68108633/120523985-82a0da80-c411-11eb-9bea-e5da9496ae93.png">

따라서 가장 중요한 label들의 갯수만을 확인하고 빠르게 다음 단계로 넘어갔습니다. (토론글에도 하나 작성하긴 했지만, 거기서 작성했던 label들간의 유사성으로 class imbalnace를 해결하기 보다는 focal loss를 사용하는게 직관적으로 더 쉬워 보여서 focal loss를 사용하기로 하였습니다.

1. label들의 분포

주피터 노트북을 통해 빠르게 data를 불러와서 label들의 분포를 확인하여 보았습니다. label들간의 불균형이 매우 매우 심했습니다. 심지어 인물 : 사망국가의 label을 가지는 data는 1개 뿐 이였습니다. 이 1개로 과연 test data의 해당 label을 잘 맞출수 있을까? 아닐것 같습니다.

이러한 imbalnace문제에 효과적으로 대처하는 방법은 이전 p-stage에서도 배웠었습니다.  바아아아로 focal loss or weighted cross entropy입니다.

focal loss에 대해서 간략하게 알아보겠습니다.

```python
class FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )
```

Focal loss는 페이스북의 Lin et al이 제안한 loss function입니다.

간단하게 말하면 분류 에러에 근거하여 맞춘 확률이 높은 Class는 조금의 loss를, 맞춘 확률이 낮은 Class는 Loss를 훨씬 높게 부여해주는 가중치를 주어서 class imbalance에 더욱 효율적으로 대처하는 loss funtion이라고 생각하면 되겠습니다.

loss 함수를 호출하게 되면 input tensor들에 대한 확률로 표현된 tensor를 얻은뒤 F.nll_loss를 호출합니다.

`F.nll_loss(((1-prob) ** self.gamma) * log_prob,target_tensor,weight=self.weight,reduction=self.reduction)`

Weight 값을 우리가 가지고 있는 label 분포에 대한 1-d tensor로 넣어줍니다.

Ex) 우리가 가지고 있는 label들은 42개니까 42 length를 가지는 1-d tensor 값은 label의 분포에 맞게

이렇게 loss 함수를 불러다가 짜주면 됩니다.

Focal loss 활용시 다양한 gamma 값들로 실험을 진행해 보았습니다.



- Date 증강

data의 수를 늘려주기 위한 방법으로 Back translation을 적용해 보았습니다. Back translation이란 말 그대로 기존 문장(source)을 다른 언어의 문장(target)로 번역한 후 다시 source언어로 번역하는 방법입니다. 이를 적용하는 방법은 아주 간단합니다. 

바로 Google translation을 이용하는 것입니다.

python에는 google translation api를 import 해서 쓸수있게 되어있습니다.

자세한 코드는 code를 참조하시면 됩니다.

Back translation으로 생성해낸 data를 포함시킨 추가적인 daataset으로 baseline을 학습시킨 결과 2%정도의 LB score 향상을 확인하였습니다.







## Model

모델을 굉장히 여러개 불러다가처음엔 돌려보았습니다. 사용해본 model 목록

1. bert-base-multilingual-cased
2. xlm-roberta-large
3. koelectra-base-v3-discriminator
4. kobert

결국 결론만 말해보자면 RoBERTa-large를 사용했습니다.
자연어 처리 task관련해서 한국어에 특화된 koelectra가 잘나올것이라 예측을 했습니다만, 많은 data를 활용해서 학습된 multilingual model이 더욱 좋은 지표를 보여주었습니다. BERT model은 huggingface 내부에 large model이 공개되어있지 않기 때문에 더욱 큰 model과 많은 data로 학습하였으면 조금더 발전된 architecture를 가지고 있는 roberta-large model을 최종적으로 사용했습니다.

Hugging face의 공식 robert의 docs와 논문을 읽어보면 아래와 같은 문구가 있습니다.

> This paper shows that pretraining multilingual language models at scale leads to **significant performance gains** for a wide range of cross-lingual transfer tasks.
>
> 

RoBERTa는 bert와 유사하지만 BERT에 다양한 방법을 적용시켜 성능을 향상한 model입니다. Model의 구조는 bert와 흡사하지만, 단지 hyperparameter를 최적화하고 NSP를 없애고 최대한 max_length에 맞춰서 문장을 넣어주고, masking을 더 다양한 방법으로 해주었다고 합니다.



## train

1. loss
2. optimizer
3. Train-set, validation-set 나누기

이번 KLUE에서 Huggingface의 라이브러리는 질리도록 다루었습니다.

Hugging face를 활용하여 model name만을 바꾸어주고 Autotokenizer와 config 파일에 model name을 인자로 전달해주면 간단하게 pretrained된 model을 불러올 수 있습니다. 이전 P-stage에서는 training 과정을 naive 한 pytorch로 구현했던 과정과는 사뭇 다른 code 진행이였습니다.

또한 trainer라는 아주 편리한 tool까지 있어 이안에 data와 model, train argument들을 전달해주면, model과 log 저장부터 training 과정의 시각화까지 간단하게 진행할 수 있었습니다.

roberta large-autotokenizer로 불러온 tokenizer은 대부분의 vocab들을 포함하고 있어, unknown으로 나오는 token들이 적은것을 확인하였습니다. 특히 entity가 unk로 나오게되면 큰 문제임으로 이를 체크하였는데, 모두 잘 tokenize된것을 확인하였습니다.

Huggingface 홈페이지에 들어가서 trainer를 자세히 살펴보았습니다.

처음에 정했던 focal loss를 사용하기 위해서는 trainer class를 상속받아서 나만의 trainer class를 구현해주어야 했습니다. 홈페이지를 보면 관련 예제가 있어 쉽게 바꿔줄수 있었습니다

```python
class FocalLossTrainer(Trainer) :
    def compute_loss(self, model, inputs, return_outputs=False) :
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = FocalLoss(weight=weight)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else los
```

여기서 `loss_fn`만 위에서 정의한 FocalLoss()를 불러다가 사용하면 됩니다. 



이처럼 Trainer class를 상속받아, 추가적인 내부 함수만을 수정하여 준다면 다양한 실험들을 손쉽게 진행해 줄 수 있습니다.

Optimizer관련해서는 trainer 안에서 사용하는 AdamW를 그대로 사용하였습니다



Default 로 설정된 lr_scheduler은 step에 따라 linear 하게 lr 이 감소하는 StepLR을 사용하고 있습니다. 저번에 사용했던 **CosineAnnealingWarmRestarts** 으로 바꾸어서 사용해보면 어떨까? 생각을 해보았고. 이는 성능향상을 불러왔습니다.

세세한 parameter은 seed를 고정시킨 이후 validation score를 기준으로 설정해주었습니다.
validation 과 train은 2:8 로 나누었고 제출전에는 data모두를 사용해서 train 한 model로 inference 하였다.



**자 이제 가장크게 고민을 해준 부분입니다**



## Input 형식에 따른 성능

이번 base line code에서 가장 의문점이 들었던 것은 tokenizer에 넣어주는 data 의 형식이였습니다.



**base line input 형식**

ent01 : 이순신

ent02 : 무신

sentence : 이순신은 조선중기의 무신이다.

\# RoBERTa tokenizer 기준 special token

```
{'bos_token': '<s>',
 'eos_token': '</s>',
 'unk_token': '<unk>',
 'sep_token': '</s>',
 'pad_token': '<pad>',
 'cls_token': '<s>',
 'mask_token': '<mask>'}
 
 <s> 이순신 </s> 무신 </s><s> 문장 </s>
```


이러한 input형식의 이유에 대해 오피스아워에서 질문을 해보았는데, 별다른 이유 없이 간단하게 정해준 형식이라고 하셨습니다.

따라서 초코송이님께서 올려주신 ner을 entitiy사이에 넣어주는 논문글을 읽고 pororo library를 이용하여 ner을 얻어낸뒤 이를 entity양옆에 삽입하여 주었습니다.

<img width="790" alt="image-20210429181250424" src="https://user-images.githubusercontent.com/68108633/120505846-5466cf00-c400-11eb-8222-30a1a837f6d9.png">

이와같은 형식입니다. 이러한 실험의 방법에 관한 내용들과 결과를 토론글에 공유하였습니다 

<img width="1050" alt="image-20210603022140850" src="https://user-images.githubusercontent.com/68108633/120524920-8a14b380-c412-11eb-848f-a5adb2286891.png">

처음에는 저 기호들을 special token에 추가해준뒤, model에 넣어주었지만!!

혜린님의 질문과 피드백으로 논문에서는 special 토큰으로 지정하지 않고, 원래 vocab에 있는 기호들을 사용하였다는 사실을 알게되었습니다......
이미 토론글에 올렸는데......... 올리기 잘했다는 생각이 들었습니다. 올리지 않았다면 이 문제를 모르고 잘못된 지식을 가진채로 실험하였을것이며, 잘못된 정보를 캠퍼분들께 전달해 드릴뻔 했습니다.

마스터이나 조교님들 말씀대로 일단 올리며 남들에게 feedback을 받는게 좋은것 같습니다. 남들에게 배울점도 많고, 스스로 좀더 찾아보고 학습하게 되는것 같습니다.

이러한 input 형식은 동일 seed model hyperparamter으로 약 1.5퍼 정도 leaderboard acc의 상승을 이끌어냈습니다. 다만 ner을 붙혀주는 과정에서 조금 모호한 부분들이있었는데 이는 code에 주석으로 설명해 놓았습니다.





하지만 이번 stage로 관심이 없었던 NLP에 대해 다시한번 생각해보게 되었습니다. 생각보다 재미있는 아이디어들이 많았고, 특히 오피스아워나 마스터 클래스에서 멘토님과 마스터님의 열정을 보고 굉장히 큰 영감과 자극을 받았던것 같습니다. 

이후 stage를 모두 vision 관련 task로 선정하였지만, NLP 관련 공부나 project도 계속해서 해나가고 싶습니다 : )





