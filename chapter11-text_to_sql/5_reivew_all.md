# Text-to-SQL 프로젝트

이 프로젝트는 자연어 질문을 SQL 쿼리로 변환하는 Text-to-SQL 시스템을 구축하는 전체 과정을 다룹니다. 영어 데이터셋을 한국어로 번역하여 활용하는 방법부터 모델 학습, 평가, 그리고 실제 응용 사례까지 포함합니다.

## 📋 목차

1. [데이터 전처리: 번역](#1-데이터-전처리-번역)
2. [데이터 생성: OpenAI API 활용](#2-데이터-생성-openai-api-활용)
3. [모델 학습](#3-모델-학습)
4. [번역 데이터와 영어 데이터의 성능 비교](#4-번역-데이터와-영어-데이터의-성능-비교)
5. [실무 응용: 도서관 관리 시스템](#5-실무-응용-도서관-관리-시스템)

## 1. 데이터 전처리: 번역

영어 Text-to-SQL 데이터셋을 한국어로 번역하여 활용하는 과정입니다.

### 주요 과정

- **Clinton/Text-to-sql-v1** 데이터셋 로드 및 분석
- GPT-4o를 사용하여 영어 질문을 한국어로 번역
- SQL 쿼리 관련 특수 구문(테이블명, 필드명, SQL 키워드 등)은 유지

```python
# 번역을 위한 시스템 프롬프트
translation_system_prompt = """
당신은 SQL 관련 지시문을 영어에서 한국어로 번역하는 전문가입니다. 다음 규칙을 반드시 준수하세요:

1. SQL 쿼리 구문(SELECT, FROM, WHERE 등)은 번역하지 않고 원문 그대로 유지합니다.
...중략...
"""
```

### 결과 및 데이터셋

- **코드**: [1_preprocessing_translation.ipynb](1_preprocessing_translation.ipynb)
- **최종 데이터셋**: `daje/kotext-to-sql-v1-hard`

## 2. 데이터 생성: OpenAI API 활용

직접 데이터를 생성하여 데이터셋을 보강하는 과정입니다.

### 주요 과정

- 다양한 도메인, SQL 복잡성, 작업 유형을 포함하는 메타데이터 정의
- 난이도별 SQL 문법 조합 자동 생성
- GPT-4o를 활용한 고품질 Text-to-SQL 데이터 생성
- 생성된 데이터의 품질 평가 및 필터링

```python
# 생성 프롬프트의 일부
system_prompt = "당신은 Text-to-SQL 데이터셋 생성을 위한 전문 AI 도우미입니다."

user_prompt = """## 목표
주어진 조건에 맞는 자연스러운 한국어 질문과 해당하는 SQL 쿼리를 생성하세요.

## 출력 형식
다음 JSON 형식으로 데이터를 생성하세요:
{
  "TEXT": "사용자 질문 (한국어)",
  "MySQL": "SQL 쿼리",
  "Schema": "DB 및 테이블 정의",
  "Difficulty": 1-5 숫자,
  ...
}
"""
```

### 결과 및 데이터셋

- **코드**: [2_preprocessing_generation.ipynb](2_preprocessing_generation.ipynb)
- **최종 데이터셋**: `daje/synthetic-ko-sql-hard`

## 3. 모델 학습

번역 및 생성된 데이터셋을 사용하여 모델을 학습하는 과정입니다.

### 주요 과정

- **Qwen2.5-Coder-7B-Instruct** 모델을 기반으로 한 파인튜닝
- LoRA (Low-Rank Adaptation) 기법을 활용한 효율적인 학습
- 데이터셋 전처리 및 토큰화
- 학습 파라미터 최적화 및 모델 저장

### 한국어 데이터셋 학습

```python
# LoRA 설정
peft_config = LoraConfig(
    lora_alpha=128,                       
    lora_dropout=0.05,                    
    r=256,                                
    bias="none",                          
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "up_proj", "down_proj", "gate_proj"],
    task_type="CAUSAL_LM",                
)

# 학습 파라미터
args = TrainingArguments(
    output_dir="Qwen2.5-260000_ko",
    num_train_epochs=1,                 
    per_device_train_batch_size=1,       
    gradient_accumulation_steps=2,       
    learning_rate=5e-5,                  
    # ... 추가 설정
)

# 모델 학습
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    peft_config=peft_config,
    tokenizer=tokenizer,
    data_collator=collator,
)
```

### 영어 데이터셋 학습

- 동일한 모델 아키텍처와 학습 설정 사용
- 영어 원본 데이터 활용

### 결과 및 모델

- **코드**: 
  - [3_1_train.ipynb](3_1_train.ipynb)
  - [3_2_train_qwen2.5_coder-en_20250413.ipynb](3_2_train_qwen2.5_coder-en_20250413.ipynb)
  - [3_2_train_qwen2.5_coder-ko_20250413.ipynb](3_2_train_qwen2.5_coder-ko_20250413.ipynb)
- **최종 모델**: `daje/Qwen2.5-coder-7B-ko-merged`

## 4. 번역 데이터와 영어 데이터의 성능 비교

번역된 한국어 데이터셋과 원본 영어 데이터셋으로 학습한 모델의 성능을 비교합니다.

### 평가 방법

- **Exact Match**: 생성된 SQL 쿼리와 정답이 정확히 일치하는지 확인
- **Semantic Equivalence**: OpenAI API를 활용하여 의미적 동등성 평가
- **종합 정확도**: Exact Match와 Semantic Equivalence를 결합한 평가

### 결과 요약

| 모델 | Exact Match | Semantic Equivalence | 종합 정확도 |
|------|-------------|----------------------|------------|
| 한국어 학습 모델 | 60.67% | 6.16% | 63.09% |
| 영어 학습 모델 | 76.47% | 27.71% | 82.99% |

```python
# 의미적 동등성 평가
prompt = """아래에는 한 개의 문제와 두 개의 SQL 쿼리가 주어집니다.
이때, "두 쿼리가 문자열로 달라도 실제로 동일한 결과를 반환하는지"를 판단하세요.
...
"""
```

### 분석

- 영어 데이터로 학습한 모델이 더 높은 성능을 보였으나, 한국어 모델도 양호한 성능을 보임
- 번역 과정에서 발생할 수 있는 의미의 미묘한 차이가 성능 차이의 원인으로 추정됨
- 한국어 환경에서는 번역된 데이터로 학습한 모델이 실용적 가치가 높음

## 5. 실무 응용: 도서관 관리 시스템

Text-to-SQL 기술을 활용한 실제 애플리케이션 구현 사례입니다.

### 시스템 구성

- SQLite 기반 도서관 관리 데이터베이스 구축
- LangChain 및 Gradio를 활용한 자연어 인터페이스 개발
- 대화 컨텍스트를 유지하는 지속적 대화 시스템 구현

```python
# 자연어 질문 처리 함수
def process_question(question, use_context=True):
    # 이전 대화 컨텍스트 가져오기
    context_str = conversation_context.get_context_str() if use_context else ""

    # 컨텍스트가 있는 경우 질문에 추가
    if context_str:
        enhanced_question = f"{context_str}\n새로운 질문: {question}"
    else:
        enhanced_question = question

    query = generate_sql(enhanced_question)
    # ... SQL 실행 및 답변 생성
```

### 주요 기능

- 도서, 회원, 대출, 예약, 이벤트 등 다양한 정보 조회
- 자연어 질문을 SQL 쿼리로 변환하고 결과 제공
- 이전 대화 컨텍스트를 고려한 연속적인 질의응답
- 사용자 친화적인 웹 인터페이스

### 결과 및 코드

- **코드**: [6_webinterface_도서관관리시스템.ipynb](6_webinterface_도서관관리시스템.ipynb)
- **실행 방법**: Jupyter Notebook에서 코드 실행 후 생성된 Gradio 링크 접속

## 결론

이 프로젝트는 Text-to-SQL 시스템의 개발 전 과정을 보여줍니다. 데이터 번역과 생성부터 모델 학습, 성능 평가, 그리고 실제 응용 사례까지 다루며, 특히 한국어 환경에서의 Text-to-SQL 구현에 초점을 맞추고 있습니다.

번역 데이터와 영어 데이터의 성능 비교를 통해 언어적 특성이 모델 성능에 미치는 영향을 확인할 수 있었으며, 도서관 관리 시스템 예제를 통해 실제 비즈니스 환경에서의 활용 가능성을 보여주었습니다.

## 참고 자료

- Qwen 모델: https://huggingface.co/Qwen
- Text-to-SQL 평가 방법: https://yale-lily.github.io/spider
- LangChain 문서: https://python.langchain.com/docs/get_started/introduction

---

## 라이센스

이 프로젝트는 MIT 라이센스를 따릅니다. 자세한 내용은 LICENSE 파일을 참조하세요.