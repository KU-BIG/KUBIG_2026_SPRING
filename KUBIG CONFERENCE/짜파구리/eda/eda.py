import os
import glob
import pandas as pd

# 1. 바탕화면의 '강원도' 폴더 경로 지정
# os.path.expanduser('~')는 사용자 홈 디렉토리(예: /Users/seohyeon)를 자동으로 찾아줍니다.
folder_path = os.path.expanduser('~/Desktop/강원도/낙뢰')

# 2. 폴더 내의 모든 csv 파일 찾기
csv_files = glob.glob(os.path.join(folder_path, '*.csv')) 
print(f"📂 폴더에서 총 {len(csv_files)}개의 CSV 파일을 찾았습니다.\n")

# 3. 모든 데이터를 하나로 합치기
df_list = []

print("⏳ 데이터를 읽고 병합하는 중입니다. (데이터가 커서 몇 초 정도 걸릴 수 있습니다...)")
for file in csv_files:
    try:
        # 공공데이터 기본 인코딩인 cp949로 먼저 시도
        temp_df = pd.read_csv(file, encoding='cp949')
    except UnicodeDecodeError:
        # 실패 시 utf-8로 재시도
        temp_df = pd.read_csv(file, encoding='utf-8')
    
    df_list.append(temp_df)

# 리스트에 모인 모든 데이터를 하나의 데이터프레임으로 수직 병합 (아래로 이어 붙이기)
if df_list:
    aws_df = pd.concat(df_list, ignore_index=True)
    
    print("\n✅ 모든 데이터 병합 완료!\n")
    print("="*50)
    print("📊 1. 데이터 요약 정보 (info)")
    print("="*50)
    aws_df.info()
    
    print("\n" + "="*50)
    print("🚨 2. 항목별 결측치(빈칸) 개수 확인")
    print("="*50)
    print(aws_df.isnull().sum())
    
    print("\n" + "="*50)
    print("📅 3. 데이터 기간 및 규모")
    print("="*50)
    print(f"- 총 데이터 건수: {len(aws_df):,}건")
    if '일시' in aws_df.columns:
        print(f"- 데이터 기간: {aws_df['일시'].min()} ~ {aws_df['일시'].max()}")
    if '지점' in aws_df.columns:
        print(f"- 포함된 고유 지점(관측소) 수: {aws_df['지점'].nunique()}개 지점")
        
    print("-포함된 변수명:", aws_df.columns)
    
    print("\n" + "="*50)
    print("👀 4. 병합된 데이터 미리보기 (앞부분 5행)")
    print("="*50)
    print(aws_df.head())
    
else:
    print("❌ 폴더에 합칠 CSV 파일이 없습니다. 경로('~/Desktop/강원도')와 파일명 확장자를 다시 확인해주세요.")
    

# 1. '일시' 컬럼을 진짜 날짜(Datetime) 타입으로 변환
# errors='coerce'를 넣으면, 날짜 모양이 아닌 이상한 글자(안내 문구 등)는 전부 빈칸(NaT)으로 찌그러뜨립니다.
aws_df['일시'] = pd.to_datetime(aws_df['일시'], errors='coerce')

# 2. 이상한 글자가 빈칸(NaT)으로 변한 쓰레기 행들을 통째로 삭제
aws_df = aws_df.dropna(subset=['일시'])

# 3. 🚨 미리보기 데이터 확인 결과 (인천 데이터가 섞여 있음!)
# 우리는 강원도 모델을 만들 것이므로 '강원'이 들어간 주소지만 쏙 뽑아냅니다.
강원_낙뢰_df = aws_df[aws_df['주소지'].str.contains('강원', na=False)].copy()

print("\n" + "="*50)
print("✨ 정제 완료 후 진짜 데이터 기간 및 규모")
print("="*50)
print(f"- 강원도 낙뢰 데이터 건수: {len(강원_낙뢰_df):,}건")
print(f"- 실제 데이터 기간: {강원_낙뢰_df['일시'].min()} ~ {강원_낙뢰_df['일시'].max()}")
print(강원_낙뢰_df.head())