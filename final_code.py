import pandas as pd
import numpy as np

rc = pd.read_excel('cont_result.xlsx')
rr = pd.read_csv('rating_sampling.csv')

ret_senti = rr
ret_senti = ret_senti.drop(['index'], axis=1) #이 줄은 한번만 돌릴것


###################################################################################
####################while문 시작########################################################
#########################################################################################

##모델 합치기 input 모델링###


def app_sent(ret_senti, sentiment):
  min = 1
  for i in range(ret_senti.shape[0]):
    for j in range(ret_senti.shape[1]):
      if ret_senti.iloc[i, j] == np.nan:
        continue
      else:
        mse = (sentiment-ret_senti.iloc[i, j])**2

      if min > mse:
        min = mse
        row = i
        col = j

  return row, ret_senti.columns[col]
# row = userid
# col = cafeid
###################################################################################\
#######################행렬 분해 잠재요인 협업 추천##########################################
############################################################################


import numpy as np
from sklearn.metrics import mean_squared_error

def get_rmse(R, P, Q, non_zeros):
    error = 0
    # 두개의 분해된 행렬 P와 Q.T의 내적 곱으로 예측 R 행렬 생성
    full_pred_matrix = np.dot(P, Q.T)
    
    # 실제 R 행렬에서 널이 아닌 값의 위치 인덱스 추출하여 실제 R 행렬과 예측 행렬의 RMSE 추출
    x_non_zero_ind = [non_zero[0] for non_zero in non_zeros]
    y_non_zero_ind = [non_zero[1] for non_zero in non_zeros]
    R_non_zeros = R[x_non_zero_ind, y_non_zero_ind]
    
    full_pred_matrix_non_zeros = full_pred_matrix[x_non_zero_ind, y_non_zero_ind]
      
    mse = mean_squared_error(R_non_zeros, full_pred_matrix_non_zeros)
    rmse = np.sqrt(mse)
    
    return rmse

def matrix_factorization(R, K, steps=50, learning_rate=0.01, r_lambda = 0.01):
    num_users, num_items = R.shape
    # P와 Q 매트릭스의 크기를 지정하고 정규분포를 가진 랜덤한 값으로 입력합니다. 
    np.random.seed(1)
    P = np.random.normal(scale=1./K, size=(num_users, K))
    Q = np.random.normal(scale=1./K, size=(num_items, K))

    break_count = 0
       
    # R > 0 인 행 위치, 열 위치, 값을 non_zeros 리스트 객체에 저장. 
    non_zeros = [ (i, j, R[i,j]) for i in range(num_users) for j in range(num_items) if R[i,j] > 0 ]
   
    # SGD기법으로 P와 Q 매트릭스를 계속 업데이트. 
    for step in range(steps):
        for i, j, r in non_zeros:
            # 실제 값과 예측 값의 차이인 오류 값 구함
            eij = r - np.dot(P[i, :], Q[j, :].T)
            # Regularization을 반영한 SGD 업데이트 공식 적용
            P[i,:] = P[i,:] + learning_rate*(eij * Q[j, :] - r_lambda*P[i,:])
            Q[j,:] = Q[j,:] + learning_rate*(eij * P[i, :] - r_lambda*Q[j,:])
       
        rmse = get_rmse(R, P, Q, non_zeros)
        if (step % 10) == 0 :
            print("### iteration step : ", step," rmse : ", rmse)
            
    return P, Q

import pandas as pd
import numpy as np

ratings_matrix = ret_senti ##@#$%^@##데이터셋 소환
ratings_matrix = ratings_matrix.fillna(0)

P, Q = matrix_factorization(ratings_matrix.values, K=50, steps=50, learning_rate=0.01, r_lambda = 0.01)
pred_matrix = np.dot(P, Q.T)


ratings_pred_matrix = pd.DataFrame(data=pred_matrix, index= ratings_matrix.index,
                                   columns = ratings_matrix.columns)

ratings_pred_matrix.head(3)



def get_ungone_cafe(ratings_matrix, userId):
    # userId로 입력받은 사용자의 모든 영화정보 추출하여 Series로 반환함. 
    # 반환된 user_rating 은 영화명(title)을 index로 가지는 Series 객체임. 
    user_rating = ratings_matrix.loc[userId,:]
    
    # user_rating이 0보다 크면 기존에 관람한 영화임. 대상 index를 추출하여 list 객체로 만듬
    already_gone = user_rating[ user_rating > 0].index.tolist()
    
    # 모든 영화명을 list 객체로 만듬. 
    cafe_list = ratings_matrix.columns.tolist()
    
    # list comprehension으로 already_seen에 해당하는 movie는 movies_list에서 제외함. 
    ungone_list = [ cafe for cafe in cafe_list if cafe not in already_gone]
    
    return ungone_list

def recomm_cafe_by_userid(pred_df, userId, ungone_list, top_n=116):
    # 예측 평점 DataFrame에서 사용자id index와 ungone_list로 들어온 영화명 컬럼을 추출하여
    # 가장 예측 평점이 높은 순으로 정렬함. 
    recomm_cafe = pred_df.loc[userId, ungone_list].sort_values(ascending=False)[:top_n]
    return recomm_cafe

#=============================================================================
# 웹으로부터 입력
sentiment = 0.1 #input()#
row, col_name = app_sent(ret_senti, sentiment) #데이터셋 소환
# ============================================================================
# 사용자가 관람하지 않는 영화명 추출   
ungone_list = get_ungone_cafe(ratings_matrix, row)

# 아이템 기반의 인접 이웃 협업 필터링으로 영화 추천 ###인풋값
recomm_cafe = recomm_cafe_by_userid(ratings_pred_matrix, row, ungone_list, top_n=116)

# 평점 데이타를 DataFrame으로 생성. ###인풋값
recomm_cafe = pd.DataFrame(data=recomm_cafe.values,index=recomm_cafe.index,columns=['pred_score'])
recomm_cafe

print(row, col_name) ###행렬분해 잠재요인 협업 추천 리스트


####################################################################################
###########################################################################################
#######################컨텐츠 기반_메뉴#########################################
########################################################################################
########################################################################
import pandas as pd
import numpy as np
import warnings; warnings.filterwarnings('ignore')

cafe_df = rc[['검색어', '메뉴', 'sentiment', 'count']]



from sklearn.feature_extraction.text import CountVectorizer

# CountVectorizer를 적용하기 위해 공백문자로 word 단위가 구분되는 문자열로 변환. 
cafe_df['menu_literal'] = cafe_df['메뉴'].apply(lambda x : ('').join(x))
count_vect = CountVectorizer(min_df=0, ngram_range=(1,2))
menu_mat = count_vect.fit_transform(cafe_df['menu_literal'])
cafe_df['menu_mat'] = count_vect.fit_transform(cafe_df['menu_literal'])

print(menu_mat.shape)
print('\n')

from sklearn.metrics.pairwise import cosine_similarity
menu_sim = cosine_similarity(menu_mat, menu_mat)
# print(menu_sim.shape)
# print('\n')

# print(menu_sim[:2])
# print('\n')
menu_sim_sorted_ind = menu_sim.argsort()[:, ::-1]
# print(menu_sim_sorted_ind[:1])
# print('\n')

def find_sim_cafe(df, sorted_ind, title_name, top_n=116):
    # 인자로 입력된 movies_df DataFrame에서 'title' 컬럼이 입력된 title_name 값인 DataFrame추출
    name_cafe = df[df['검색어'] == title_name]
    # title_named을 가진 DataFrame의 index 객체를 ndarray로 반환하고 
    # sorted_ind 인자로 입력된 genre_sim_sorted_ind 객체에서 유사도 순으로 top_n 개의 index 추출
    title_index = name_cafe.index.values
    similar_indexes = sorted_ind[title_index, :(top_n)]
    # 추출된 top_n index들 출력. top_n index는 2차원 데이터 임. 
    #dataframe에서 index로 사용하기 위해서 1차원 array로 변경
    similar_indexes = similar_indexes.reshape(-1)
    return df.iloc[similar_indexes].sort_values(ascending=False)[:top_n]

# # def find_sim_cafe(df, sorted_ind, title_name, top_n=116):
# #     title_cafe = df[df['검색어'] == title_name]
# #     title_index = title_cafe.index.values
# #     # top_n의 2배에 해당하는 쟝르 유사성이 높은 index 추출 
# #     similar_indexes = sorted_ind[title_index, :(top_n)]
# #     similar_indexes = similar_indexes.reshape(-1)
# #     # 기준 영화 index는 제외
# #     similar_indexes = similar_indexes[similar_indexes != title_index]
# #     # top_n의 2배에 해당하는 후보군에서 weighted_vote 높은 순으로 top_n 만큼 추출 
# #     return df.iloc[similar_indexes].sort_values('weighted_vote', ascending=False)[:top_n]

# # similar_cafe = find_sim_cafe(cafe_df, menu_sim_sorted_ind, row,10)
# # similar_cafe[['검색어', 'sentiment']]

################################################################################################
#########################################투표 보정식#############################################
###########################################################################################

C = cafe_df['sentiment'].mean()
m = cafe_df['count'].quantile(0.6)
print('C:',round(C,3), 'm:',round(m,3))
print('\n')



percentile = 0.6
m = cafe_df['count'].quantile(percentile)
C = cafe_df['sentiment'].mean()

def weighted_vote_average(record):
    v = record['count']
    R = record['sentiment']
    
    return ( (v/(v+m)) * R ) + ( (m/(m+v)) * C )   

cafe_df['weighted_vote'] = cafe_df.apply(weighted_vote_average, axis=1)



cafe_df[['검색어','sentiment','weighted_vote','count']].sort_values('weighted_vote', ascending=False)[:116]


def find_sim_cafe(df, sorted_ind, title_name, top_n=116):
    title_cafe = df[df['검색어'] == title_name]
    title_index = title_cafe.index.values
    # top_n의 2배에 해당하는 쟝르 유사성이 높은 index 추출 
    similar_indexes = sorted_ind[title_index, :(top_n)]
    similar_indexes = similar_indexes.reshape(-1)
    # 기준 영화 index는 제외
    similar_indexes = similar_indexes[similar_indexes != title_index]
    # top_n의 2배에 해당하는 후보군에서 weighted_vote 높은 순으로 top_n 만큼 추출 
    return df.iloc[similar_indexes].sort_values('weighted_vote', ascending=False)[:top_n]

print('\n')
similar_cafe = find_sim_cafe(cafe_df, menu_sim_sorted_ind, col_name,116)
print(similar_cafe[['검색어', 'sentiment', 'weighted_vote']]) ##컨텐츠 기반 추천 리스트
# print(menu_mat)

#################################################################################################
###############################마무리##############################################################
###############################################################################################
df = similar_cafe[['검색어', 'weighted_vote']]

df2 = recomm_cafe.reset_index()
df2.columns = ['검색어', 'pred_score']
df2


###while문 돌아가면서 m비율에서 n비율로 가중치가 증가하도록#######################

m=7 #컨텐츠기반
n=3 #행렬분해잠재요인
df3 = pd.merge(df, df2, how='left', on='검색어')
df3['final'] = df3['weighted_vote']*m + df3['pred_score']*n
a = df3.sort_values('final', ascending=False)[:10] 

print(a['검색어']) #최종 모델 리스트
