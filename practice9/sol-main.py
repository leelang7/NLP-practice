import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import codecs
import time

def np_cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0  # 0벡터 예외 처리

    return dot_product / (norm1 * norm2)


def create_similarity_matrix(corpus):
    """
    문서를 담은 corpus list에서 각 문서 간 유사도 측정
    :param corpus: 문서를 text로 가지는 list
    :return: 문서 간 유사도 값을 가지는 행렬
    """
    corpus_size = len(corpus)
    sim_mat = np.zeros((corpus_size, corpus_size))
    
    corpus_vsm = make_vector_space_model_tf_idf(corpus)
    
    for i in range(corpus_size):
        for j in range(corpus_size):
            if i != j:  # No need to compute similarity with its own (cos(0)=1)
                sim_mat[i, j] = compute_doc_pair_similarity(corpus_vsm[i], corpus_vsm[j])
    
    return sim_mat

def make_vector_space_model_tf_idf(corpus):
    """
    문서를 담은 corpus list에서 tf-idf 가중치를 가지는 vector space model 생성
    :param corpus: 문서를 text로 가지는 list
    :return: tf-idf 가중치를 가지는 vector space model (numpy array)
    """
    vectorizer = TfidfVectorizer()
    corpus_vsm = vectorizer.fit_transform(corpus).toarray()
    return corpus_vsm

def compute_doc_pair_similarity(doc_1, doc_2):
    """
    두 문서의 유사도를 cosine similarity로 계산
    :param doc_1: 첫 번째 문서 tf-idf 가중치를 가지는 벡터
    :param doc_2: 두 번째 문서 tf-idf 가중치를 가지는 벡터
    :return: 두 문서의 유사도 값
    """
    #print(doc_1.shape)
    cos_sim = cosine_similarity(doc_1.reshape(1, -1), doc_2.reshape(1, -1))[0][0]
    s2_time = time.time()
    np_sim_mat = np_cosine_similarity(doc_1[0], doc_2[0])
    e2_time = time.time()
    print('넘파이 코사인 유사도', s2_time - e2_time)

    return cos_sim

def main():
    # 데이터 'news.txt' 파일을 불러옵니다.
    corpus = list()
    with codecs.open("./data/news.txt", "r", "utf-8") as txt_f:
        for line in txt_f:
            corpus.append(line.strip())
    
    # corpus 내 문서들의 유사도값을 계산합니다.
    s_time = time.time()
    sim_mat = create_similarity_matrix(corpus)
    e_time = time.time()

    # 결과 출력
    print("First news title: {}".format(corpus[0]))
    print("Second news title: {}".format(corpus[1]))
    print("Similarity between them: {}".format(sim_mat[0, 1]))
    print()

    print("Second news title: {}".format(corpus[1]))
    print("Third news title: {}".format(corpus[2]))
    print("Similarity between them: {}".format(sim_mat[1, 2]))
    print()

    print("Third news title: {}".format(corpus[2]))
    print("Forth news title: {}".format(corpus[3]))
    print("Similarity between them: {}".format(sim_mat[3, 2]))
    print('사이킷런 코사인 유사도:', e_time - s_time)

if __name__ == "__main__":
    main()