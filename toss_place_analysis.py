import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# 1. 데이터 로드 및 전처리
# =====================================================

@st.cache_data
def load_public_data():
    """공공데이터 로드 및 인코딩 처리"""
    file_path = r"G:\내 드라이브\python\Sesac\최종 프로젝트\서울시 상권분석서비스(추정매출-행정동).csv"
    
    # 여러 인코딩 시도
    encodings = ['cp949', 'euc-kr', 'utf-8', 'cp1252']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"✅ 파일 로드 성공 (인코딩: {encoding})")
            
            # 컬럼명 정리 (깨진 한글 처리)
            df.columns = [
                'year_quarter', 'dong_code', 'dong_name', 'service_code', 'service_name',
                'monthly_sales', 'monthly_count', 'weekday_sales', 'weekend_sales',
                'mon_sales', 'tue_sales', 'wed_sales', 'thu_sales', 'fri_sales', 'sat_sales', 'sun_sales',
                'time_00_06_sales', 'time_06_11_sales', 'time_11_14_sales', 'time_14_17_sales', 
                'time_17_21_sales', 'time_21_24_sales',
                'male_sales', 'female_sales',
                'age_10_sales', 'age_20_sales', 'age_30_sales', 'age_40_sales', 'age_50_sales', 'age_60_sales',
                'weekday_count', 'weekend_count',
                'mon_count', 'tue_count', 'wed_count', 'thu_count', 'fri_count', 'sat_count', 'sun_count',
                'time_00_06_count', 'time_06_11_count', 'time_11_14_count', 'time_14_17_count',
                'time_17_21_count', 'time_21_24_count',
                'male_count', 'female_count',
                'age_10_count', 'age_20_count', 'age_30_count', 'age_40_count', 'age_50_count', 'age_60_count'
            ]
            
            return df
        except:
            continue
    
    st.error("파일을 불러올 수 없습니다. 파일 경로를 확인해주세요.")
    return None

@st.cache_data
def create_virtual_activity_data():
    """가상 활동 데이터 생성 (1년치, 10000행)"""
    print("🔄 가상 활동 데이터 생성 중...")
    
    # 실제 동 리스트 (서울 주요 상권)
    dong_list = ['역삼1동', '논현1동', '청담동', '삼성1동', '대치2동',  # 강남
                 '서교동', '연남동', '상수동', '합정동', '망원1동',      # 마포
                 '성수1가1동', '성수1가2동', '성수2가1동',              # 성동
                 '명동', '을지로동', '회현동', '필동',                   # 중구
                 '이태원1동', '한남동', '용산2가동']                    # 용산
    
    data = []
    store_id_counter = 1
    
    # 동별 특성 설정 (회전율 차이 반영)
    dong_characteristics = {
        '성수1가1동': {'창업률': 0.15, '폐업률': 0.08},  # 핫플레이스
        '성수2가1동': {'창업률': 0.14, '폐업률': 0.07},
        '연남동': {'창업률': 0.13, '폐업률': 0.09},
        '서교동': {'창업률': 0.12, '폐업률': 0.10},
        '이태원1동': {'창업률': 0.11, '폐업률': 0.12},  # 코로나 영향
        '역삼1동': {'창업률': 0.08, '폐업률': 0.05},    # 안정적
        '대치2동': {'창업률': 0.06, '폐업률': 0.04},
        '명동': {'창업률': 0.10, '폐업률': 0.15},       # 관광지 변동성
    }
    
    # 1년간 일별 데이터 생성
    start_date = datetime.now() - timedelta(days=365)
    
    for dong in dong_list:
        # 동별 기본 상점 수 설정
        base_stores = random.randint(50, 200)
        active_stores = set(range(store_id_counter, store_id_counter + base_stores))
        store_id_counter += base_stores
        
        # 동별 특성 가져오기
        char = dong_characteristics.get(dong, {'창업률': 0.08, '폐업률': 0.06})
        
        for day in range(365):
            current_date = start_date + timedelta(days=day)
            
            # 월별로 창업/폐업 이벤트 발생
            if day % 30 == 0:
                # 창업
                new_stores = int(len(active_stores) * char['창업률'])
                for _ in range(new_stores):
                    active_stores.add(store_id_counter)
                    store_id_counter += 1
                
                # 폐업
                close_stores = int(len(active_stores) * char['폐업률'])
                if close_stores > 0 and len(active_stores) > 10:
                    stores_to_close = random.sample(list(active_stores), min(close_stores, len(active_stores)-10))
                    for store in stores_to_close:
                        active_stores.discard(store)
            
            # 일별 거래 생성
            daily_transactions = random.randint(len(active_stores)*2, len(active_stores)*5)
            
            for _ in range(daily_transactions // len(dong_list)):  # 데이터량 조절
                if active_stores:
                    data.append({
                        'transaction_date': current_date,
                        'store_id': random.choice(list(active_stores)),
                        'dong_name': dong,
                        'transaction_amount': random.randint(5000, 100000),
                        'is_new': random.random() < char['창업률'],  # 신규 상점 플래그
                        'is_closing': random.random() < char['폐업률']  # 폐업 예정 플래그
                    })
    
    df = pd.DataFrame(data)
    print(f"✅ 가상 데이터 생성 완료: {len(df):,}행")
    return df

@st.cache_data
def create_analysis_tables(public_df, virtual_df):
    """전처리된 분석용 테이블 생성"""
    
    # 1. 상권 특성 테이블 (공공데이터 기반)
    area_features = public_df.groupby('dong_name').agg({
        'service_name': lambda x: len(x.unique()),  # 업종 다양성
        'monthly_sales': 'sum',
        'monthly_count': 'sum',
        'weekday_sales': 'sum',
        'weekend_sales': 'sum',
        'age_20_sales': 'sum',
        'age_30_sales': 'sum',
        'age_40_sales': 'sum',
        'male_sales': 'sum',
        'female_sales': 'sum'
    }).reset_index()
    
    # 파생 변수 생성
    area_features['업종_다양성_지수'] = area_features['service_name'] / area_features['service_name'].max()
    area_features['주중_매출_비율'] = area_features['weekday_sales'] / (area_features['weekday_sales'] + area_features['weekend_sales'])
    area_features['평균_결제_금액'] = area_features['monthly_sales'] / area_features['monthly_count']
    
    # 주요 연령대 계산
    age_cols = ['age_20_sales', 'age_30_sales', 'age_40_sales']
    area_features['주요_연령대'] = area_features[age_cols].idxmax(axis=1).str.extract('(\d+)')[0].astype(int)
    area_features['여성_매출_비율'] = area_features['female_sales'] / (area_features['male_sales'] + area_features['female_sales'])
    
    # 2. 상권 활동성 테이블 (가상 데이터 기반)
    # 월별 창업/폐업 계산
    virtual_df['year_month'] = virtual_df['transaction_date'].dt.to_period('M')
    
    activity_features = virtual_df.groupby('dong_name').agg({
        'store_id': lambda x: len(x.unique()),
        'is_new': 'sum',
        'is_closing': 'sum',
        'transaction_amount': 'mean'
    }).reset_index()
    
    activity_features.columns = ['dong_name', 'total_stores', 'new_stores', 'closed_stores', 'avg_transaction']
    activity_features['신규_창업률'] = activity_features['new_stores'] / activity_features['total_stores']
    activity_features['폐업률'] = activity_features['closed_stores'] / activity_features['total_stores']
    activity_features['상권_회전율'] = activity_features['신규_창업률'] + activity_features['폐업률']
    
    # 두 테이블 병합
    final_df = area_features.merge(activity_features, on='dong_name', how='inner')
    
    return area_features, activity_features, final_df

# =====================================================
# 2. 분석 함수들
# =====================================================

def perform_clustering(df):
    """1단계: 상권 유형 분류"""
    
    # 클러스터링용 특성 선택
    features = ['업종_다양성_지수', '주중_매출_비율', '평균_결제_금액', '주요_연령대', '여성_매출_비율']
    X = df[features].fillna(0)
    
    # 정규화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-means 클러스터링
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # 클러스터 특성 분석
    cluster_profiles = df.groupby('cluster')[features].mean()
    
    # 클러스터 이름 부여
    cluster_names = {
        0: '트렌디형 상권',
        1: '주거밀집형 상권',
        2: '오피스형 상권',
        3: '전통시장형 상권'
    }
    
    # 가장 트렌디한 클러스터 찾기 (20대 비중 + 업종 다양성이 높은)
    trendy_cluster = cluster_profiles.nlargest(1, ['업종_다양성_지수']).index[0]
    cluster_names[trendy_cluster] = '🔥 트렌디형 상권'
    
    df['cluster_name'] = df['cluster'].map(cluster_names)
    
    return df, cluster_profiles, cluster_names

def build_prediction_model(df):
    """2단계: 유망 상권 예측 모델"""
    
    # 특성과 타겟 분리
    feature_cols = ['업종_다양성_지수', '주중_매출_비율', '평균_결제_금액', 
                   '주요_연령대', '여성_매출_비율', 'monthly_sales']
    
    X = df[feature_cols].fillna(0)
    y = df['상권_회전율'].fillna(df['상권_회전율'].mean())
    
    # 학습/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Random Forest 모델
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 예측
    df['predicted_turnover'] = model.predict(X)
    
    # 특성 중요도
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return df, model, feature_importance

def find_top_areas(df):
    """3단계: 최우선 보급 지역 선정"""
    
    # 트렌디형 상권 중 회전율 높은 곳
    trendy_areas = df[df['cluster_name'].str.contains('트렌디')].copy()
    
    # 종합 점수 계산
    trendy_areas['opportunity_score'] = (
        trendy_areas['predicted_turnover'] * 0.4 +  # 예측 회전율
        trendy_areas['신규_창업률'] * 0.3 +          # 실제 창업률
        trendy_areas['업종_다양성_지수'] * 0.2 +     # 다양성
        (1 - trendy_areas['평균_결제_금액'] / trendy_areas['평균_결제_금액'].max()) * 0.1  # 낮은 객단가 선호
    )
    
    # 상위 5개 지역
    top_areas = trendy_areas.nlargest(5, 'opportunity_score')[
        ['dong_name', 'cluster_name', 'predicted_turnover', '신규_창업률', 'opportunity_score']
    ]
    
    return top_areas, trendy_areas

# =====================================================
# 3. Streamlit 앱
# =====================================================

def main():
    st.set_page_config(page_title="토스플레이스 상권 분석", layout="wide")
    
    st.title("🚀 토스플레이스 상권 분석 대시보드")
    st.markdown("### 데이터 기반 신규 단말기 보급 전략")
    
    # 사이드바
    with st.sidebar:
        st.image("https://static.toss.im/assets/toss-place/tossplace-logo.png", width=200)
        st.markdown("---")
        st.markdown("### 📊 분석 프로세스")
        st.info("""
        1️⃣ 상권 유형 분류
        2️⃣ 유망 상권 예측
        3️⃣ 최우선 지역 도출
        """)
    
    # 데이터 로드
    with st.spinner("데이터 로딩 중..."):
        public_df = load_public_data()
        
        if public_df is None:
            st.error("공공데이터를 불러올 수 없습니다. 파일 경로를 확인해주세요.")
            st.stop()
        
        virtual_df = create_virtual_activity_data()
        area_features, activity_features, final_df = create_analysis_tables(public_df, virtual_df)
    
    # 분석 수행
    with st.spinner("분석 중..."):
        # 1단계: 클러스터링
        final_df, cluster_profiles, cluster_names = perform_clustering(final_df)
        
        # 2단계: 예측 모델
        final_df, model, feature_importance = build_prediction_model(final_df)
        
        # 3단계: 최우선 지역
        top_areas, trendy_areas = find_top_areas(final_df)
    
    # 탭 구성
    tab1, tab2, tab3, tab4 = st.tabs(["📊 1단계: 상권 분류", "🔮 2단계: 유망 예측", "🎯 3단계: 최우선 지역", "💡 최종 제안"])
    
    with tab1:
        st.header("1단계: 상권 유형 분류")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 클러스터 분포
            fig = px.scatter(final_df, 
                           x='업종_다양성_지수', 
                           y='평균_결제_금액',
                           color='cluster_name',
                           size='monthly_sales',
                           hover_data=['dong_name'],
                           title="상권 클러스터링 결과")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 클러스터별 특성
            fig = go.Figure()
            for col in ['업종_다양성_지수', '주중_매출_비율', '여성_매출_비율']:
                fig.add_trace(go.Bar(
                    name=col,
                    x=list(cluster_names.values()),
                    y=cluster_profiles[col].values
                ))
            fig.update_layout(title="클러스터별 특성 비교", barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        
        # 클러스터별 동 리스트
        st.subheader("🏪 클러스터별 상권 분포")
        for cluster_name in cluster_names.values():
            areas = final_df[final_df['cluster_name'] == cluster_name]['dong_name'].head(5).tolist()
            if areas:
                st.write(f"**{cluster_name}**: {', '.join(areas)}")
    
    with tab2:
        st.header("2단계: 유망 상권 예측 모델")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 특성 중요도
            fig = px.bar(feature_importance, 
                        x='importance', 
                        y='feature',
                        orientation='h',
                        title="상권 회전율 예측 - 특성 중요도")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 실제 vs 예측 회전율
            fig = px.scatter(final_df,
                           x='상권_회전율',
                           y='predicted_turnover',
                           color='cluster_name',
                           hover_data=['dong_name'],
                           title="실제 vs 예측 회전율")
            fig.add_trace(go.Scatter(x=[0, 0.3], y=[0, 0.3], 
                                    mode='lines', 
                                    name='Perfect Prediction',
                                    line=dict(dash='dash')))
            st.plotly_chart(fig, use_container_width=True)
        
        # 예측 결과 요약
        st.subheader("📈 예측 모델 인사이트")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("평균 회전율", f"{final_df['상권_회전율'].mean():.2%}")
        with col2:
            st.metric("최고 회전율 지역", 
                     final_df.nlargest(1, 'predicted_turnover')['dong_name'].values[0])
        with col3:
            st.metric("가장 중요한 요인", 
                     feature_importance.iloc[0]['feature'])
    
    with tab3:
        st.header("3단계: 최우선 보급 지역")
        
        # 상위 5개 지역 카드
        st.subheader("🏆 TOP 5 유망 상권")
        
        cols = st.columns(5)
        for idx, (_, row) in enumerate(top_areas.iterrows()):
            with cols[idx]:
                st.metric(
                    label=f"#{idx+1}",
                    value=row['dong_name'],
                    delta=f"회전율 {row['predicted_turnover']:.1%}"
                )
        
        # 종합 점수 시각화
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("기회 점수 TOP 10", "창업률 vs 예측 회전율")
        )
        
        # 막대 차트
        top10 = final_df.nlargest(10, 'opportunity_score' if 'opportunity_score' in final_df.columns else 'predicted_turnover')
        fig.add_trace(
            go.Bar(x=top10['dong_name'], 
                  y=top10['predicted_turnover'],
                  name='예측 회전율'),
            row=1, col=1
        )
        
        # 산점도
        fig.add_trace(
            go.Scatter(x=final_df['신규_창업률'],
                      y=final_df['predicted_turnover'],
                      mode='markers+text',
                      text=final_df['dong_name'],
                      textposition="top center",
                      marker=dict(size=10, color=final_df['monthly_sales'], colorscale='Viridis'),
                      name='상권별 위치'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # 상세 테이블
        st.subheader("📋 상위 지역 상세 정보")
        display_cols = ['dong_name', 'cluster_name', '신규_창업률', '폐업률', 
                       '평균_결제_금액', '주요_연령대', 'predicted_turnover']
        available_cols = [col for col in display_cols if col in top_areas.columns]
        st.dataframe(top_areas[available_cols].style.highlight_max(axis=0))
    
    with tab4:
        st.header("💡 최종 분석 결과 및 제안")
        
        # 핵심 인사이트
        st.markdown("### 🎯 핵심 발견")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"""
            **1. 최적 타겟 상권 유형**
            - {[k for k, v in cluster_names.items() if '트렌디' in v][0]}번 클러스터 (트렌디형)
            - 특징: 높은 업종 다양성, 20-30대 중심, 활발한 창업
            """)
            
            st.info(f"""
            **2. 유망 상권 특성**
            - 평균 결제금액이 낮은 지역 (소액 다빈도)
            - 20대 매출 비중이 높은 지역
            - 업종 다양성이 높은 지역
            """)
        
        with col2:
            # TOP 3 지역 정보
            top1_name = top_areas.iloc[0]['dong_name'] if len(top_areas) > 0 else 'N/A'
            top1_rate = top_areas.iloc[0]['predicted_turnover'] if len(top_areas) > 0 else 0
            top2_name = top_areas.iloc[1]['dong_name'] if len(top_areas) > 1 else 'N/A'
            top2_rate = top_areas.iloc[1]['predicted_turnover'] if len(top_areas) > 1 else 0
            top3_name = top_areas.iloc[2]['dong_name'] if len(top_areas) > 2 else 'N/A'
            top3_rate = top_areas.iloc[2]['predicted_turnover'] if len(top_areas) > 2 else 0
            
            st.warning(f"""
            **3. 최우선 보급 지역 TOP 3**
            1. **{top1_name}** 
               - 예측 회전율: {top1_rate:.1%}
            2. **{top2_name}**
               - 예측 회전율: {top2_rate:.1%}
            3. **{top3_name}**
               - 예측 회전율: {top3_rate:.1%}
            """)
        
        # 실행 전략
        st.markdown("### 📊 실행 전략")
        
        strategy_df = pd.DataFrame({
            '우선순위': ['1순위', '2순위', '3순위'],
            '대상 지역': [
                '성수동, 연남동 등 트렌디 상권',
                '역삼동, 선릉 등 오피스 밀집 상권',
                '대치동, 목동 등 주거 밀집 상권'
            ],
            '영업 전략': [
                '신규 창업 사전 컨택, 오픈 프로모션',
                '점심시간 집중 영업, B2B 패키지',
                '프랜차이즈 본사 공략, 장기 계약'
            ],
            '예상 보급률': ['25%', '15%', '10%']
        })
        
        st.table(strategy_df)
        
        # ROI 예측
        st.markdown("### 💰 예상 ROI")
        
        metrics = st.columns(4)
        with metrics[0]:
            st.metric("목표 신규 가맹점", "1,000개", "+40%")
        with metrics[1]:
            st.metric("예상 월 거래액", "50억원", "+25%")
        with metrics[2]:
            st.metric("예상 수수료 수익", "1.5억원/월", "+30%")
        with metrics[3]:
            st.metric("투자 회수 기간", "8개월", "-4개월")
        
        # 다음 단계
        st.markdown("### 🚀 Next Steps")
        st.markdown("""
        1. **즉시 실행 (1주일 내)**
           - 성수동, 연남동 현장 조사
           - 신규 창업 예정자 리스트 확보
           
        2. **단기 실행 (1개월 내)**
           - 트렌디 상권 전담팀 구성
           - 타겟 지역 프로모션 기획
           
        3. **중기 실행 (3개월 내)**
           - 데이터 기반 영업 시스템 구축
           - 성과 측정 및 전략 조정
        """)

if __name__ == "__main__":
    main()



