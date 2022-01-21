import re

"""## 1) .기호"""

r = re.compile("a.c")
r.search("kkk")  # 아무런 결과도 출력되지 않는다.

r.search("abc")

"""## 2) ?기호"""

r = re.compile("ab?c")
r.search("abbc")  # 아무런 결과도 출력되지 않는다.

r.search("abc")

r.search("ac")

"""## 3) *기호"""

r = re.compile("ab*c")
r.search("a")  # 아무런 결과도 출력되지 않는다.

r.search("ac")

r.search("abc")

r.search("abbbbc")

"""## 4) +기호"""

r = re.compile("ab+c")
r.search("ac")  # 아무런 결과도 출력되지 않는다.

r.search("abc")

r.search("abbbbc")

"""##5) ^기호"""

r = re.compile("^ab")
r.search("bbc")  # 아무런 결과도 출력되지 않는다.
r.search("zab")  # 아무런 결과도 출력되지 않는다.

r.search("abz")

"""## 6) {숫자} 기호"""

r = re.compile("ab{2}c")
r.search("ac")  # 아무런 결과도 출력되지 않는다.
r.search("abc")  # 아무런 결과도 출력되지 않는다.
r.search("abbbbbc")  # 아무런 결과도 출력되지 않는다.

r.search("abbc")

"""## 7) {숫자1, 숫자2} 기호"""

r = re.compile("ab{2,8}c")
r.search("ac")  # 아무런 결과도 출력되지 않는다.
r.search("abc")  # 아무런 결과도 출력되지 않는다.

r.search("abbc")

r.search("abbbbbbbbc")

r.search("abbbbbbbbbc")  # 아무런 결과도 출력되지 않는다.

"""## 8) {숫자,} 기호"""

r = re.compile("a{2,}bc")
r.search("bc")  # 아무런 결과도 출력되지 않는다.
r.search("aa")  # 아무런 결과도 출력되지 않는다.
r.search("aaaaaaaabc")  # 아무런 결과도 출력되지 않는다.

r.search("aabc")

"""## 9) [] 기호"""

r = re.compile("[abc]")  # [abc]는 [a-c]와 같다.
r.search("zzz")  # 아무런 결과도 출력되지 않는다.

r.search("a")

r.search("aaaaaaa")

r.search("baac")

r = re.compile("[a-z]")
r.search("AAA")  # 아무런 결과도 출력되지 않는다.
r.search("111")  # 아무런 결과도 출력되지 않는다.

r.search("aBC")

"""## 10) [^문자] 기호"""

r = re.compile("[^abc]")

# 아래의 세 코드는 아무런 결과도 출력되지 않는다.
r.search("a")
r.search("ab")
r.search("b")

r.search("d")

r.search("1")

"""# 2. 정규 표현식 모듈 함수 예제

## 1) re.match() 와 re.search()의 차이
"""

r = re.compile("ab.")
r.match("kkkabc")  # 아무런 결과도 출력되지 않는다.

r.search("kkkabc")

r.match("abckkk")

"""## 2) re.split()"""

text = "사과 딸기 수박 메론 바나나"
re.split(" ", text)

text = """사과
딸기
수박
메론
바나나"""

re.split("\n", text)

text = "사과+딸기+수박+메론+바나나"

re.split("\+", text)

"""## 3) re.findall()"""

text = """이름 : 김철수
전화번호 : 010 - 1234 - 1234
나이 : 30
성별 : 남"""

re.findall("\d+", text)

re.findall("\d+", "문자열입니다.")

"""## 4) re.sub()"""

text = (
    "Regular expression : A regular expression, regex or regexp[1] (sometimes called a rational expression)[2][3]"
    " is, in theoretical computer science and formal language theory, a sequence of characters that define a "
    "search pattern."
)

preprocessed_text = re.sub("[^a-zA-Z]", " ", text)
print(preprocessed_text)

"""# 3. 정규 표현식 텍스트 전처리 예제"""

text = """100 John    PROF
101 James   STUD
102 Mac   STUD"""

re.split("\s+", text)

re.findall("\d+", text)

re.findall("[A-Z]", text)

re.findall("[A-Z]{4}", text)

re.findall("[A-Z][a-z]+", text)

"""# 4. 정규 표현식을 이용한 토큰화"""

from nltk.tokenize import RegexpTokenizer

text = "Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop"

tokenizer1 = RegexpTokenizer("[\w]+")
tokenizer2 = RegexpTokenizer("[\s]+", gaps=True)

print(tokenizer1.tokenize(text))
print(tokenizer2.tokenize(text))
