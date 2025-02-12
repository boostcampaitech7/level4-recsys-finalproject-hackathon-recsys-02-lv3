

<p align="center">
  <img src="https://github.com/user-attachments/assets/efde952e-0dbb-4923-809c-3ff57ee18bde" alt="로고 이미지" width="300">
</p>
<br>
<p align="center">
    <img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=FastAPI&logoColor=white" alt="FastAPI badge">
    <img src="https://img.shields.io/badge/React-61DAFB?style=flat-square&logo=react&logoColor=black" alt="React badge">
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=PyTorch&logoColor=white" alt="PyTorch badge">
    <img src="https://img.shields.io/badge/Faiss-0467DF?style=flat-square&logo=meta&logoColor=white" alt="faiss badge">
    <br>
    <img src="https://img.shields.io/badge/MongoDB-47A248?style=flat-square&logo=mongodb&logoColor=white" alt="MongoDB badge">
    <img src="https://img.shields.io/badge/postgresql-4169e1?style=flat-square&logo=postgresql&logoColor=white" alt="PostgreSQL badge">
    <img src="https://img.shields.io/badge/Redis-DC382D?style=flat-square&logo=redis&logoColor=white" alt="Redis badge">
    <img src="https://img.shields.io/badge/Apache%20Airflow-017CEE?style=flat-square&logo=Apache%20Airflow&logoColor=white" alt="Airflow badge">
    <img src="https://img.shields.io/badge/docker-257bd6?style=flat-square&logo=docker&logoColor=white" alt="Docker badge">
  </p>
<br><br>

## 프로젝트 설명
**[🎶TuneyourShop으로 플레이리스트 추천받기](https://tuneyourshop.vercel.app/)**
<br><br>
<strong>Tune your Shop</strong>은 가게 무드에 최적화된 음악을 추천하는 솔루션입니다.  
태그를 통해 손쉽게 매장에 어울리는 플레이리스트를 생성해드려요. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/f3cb5e48-7bf0-4847-91e1-05bddaaeb846" width="200">
  <img src="https://github.com/user-attachments/assets/d86f16f4-ba9c-4f51-9629-a288d925b19d" width="200">
  <img src="https://github.com/user-attachments/assets/f18c0e04-34b7-47e5-ac74-dc7e8da7056d" width="200">
  <img src="https://github.com/user-attachments/assets/4a9273fa-4bc3-46ae-af85-f2a3531a7ea1" width="200">
</p>

<br><br><br>

## 시연 영상
https://github.com/user-attachments/assets/30628ec3-17a4-47df-849c-2aa28fc15f1e

<br><br><br>

## 💻 팀 구성 및 역할
| 박재욱 | 서재은 | 임태우 | 조유솔 | 허진경 |
|:---:|:---:|:---:|:---:|:---:|
|[<img src="https://github.com/user-attachments/assets/0c4ff6eb-95b0-4ee4-883c-b10c1a42be14" width=130>](https://github.com/park-jaeuk)|[<img src="https://github.com/user-attachments/assets/79f5b6b2-bc16-45e7-b3de-ca6bf9a55923" width=130>](https://github.com/JaeEunSeo)|[<img src="https://github.com/user-attachments/assets/f6572f19-901b-4aea-b1c4-16a62a111e8d" width=130>](https://github.com/Cyberger)|[<img src="https://avatars.githubusercontent.com/u/112920170?v=4" width=130>](https://github.com/YusolCho)|[<img src="https://github.com/user-attachments/assets/7ab5112f-ca4b-4e54-a005-406756262384" width=130>](https://github.com/jinnk0)|
|Data Engineering|Modeling, Frontend|MLOps|Modeling|Backend|

<br><br><br>

## 서비스 아키텍쳐
<img width="100%" src="https://github.com/user-attachments/assets/bb830eaf-74ee-47d7-b708-1b5d90c3945d"/>

<br><br><br>

## 모델 아키텍쳐
<table>
  <tr align='center'>
    <td>
    <strong>LightGCN - Candidate Generation Model</strong>
    <img width="50%" src="https://github.com/user-attachments/assets/518e0fa0-b9f6-4f2f-9851-77b1f9cd26a3"/>
    </td>
    <td>
    <strong>BiEncoder - Reranking Model</strong>
    <img width="70%" src="https://github.com/user-attachments/assets/21a20b71-55fd-47b0-9c05-95e97cfd9dd4"/>
      </td>
</tr>
</table>

<br><br><br>

## 데이터셋
이 프로젝트에서는 **Spotify Playlists 데이터셋**을 활용하여 맞춤형 음악 추천 시스템을 구축합니다.  
해당 데이터셋은 **Spotify에서 제공하는 플레이리스트 기반 곡 정보**를 포함하며, 유저와 트랙 간의 상호작용 데이터를 제공합니다.

- **데이터셋 원본:** [Kaggle - Spotify Playlists](https://www.kaggle.com/datasets/andrewmvd/spotify-playlists?select=spotify_dataset.csv)
- **데이터 크기:** 1.2GB (CSV 파일)
- **데이터 유형:** 테이블 데이터 (유저-아이템 상호작용)

<br>

### **1. 사용자-곡 상호작용 데이터**
**Candidate Generation Model을 학습**하는 데 활용한 데이터입니다.

### **핵심 필드 (파일 경로: `spotify_dataset.csv`)**
| Column Name | Description |
|-------------|------------|
| `user_id`   | 유저 식별자 (임의 생성) |
| `track_name` | 곡 제목 |
| `artist`    | 아티스트 이름 |

<br>

### **2. 곡 메타데이터 수집**
**Reranking Model**을 학습하는데 메타데이터가 필요하지만, 기본 데이터셋에는 **트랙명, 아티스트, 플레이리스트명 정보만 포함**되어 있습니다.  
이를 보완하기 위해 **Last.fm 웹사이트**를 통해 추가적인 곡 메타데이터를 수집했습니다.

### **추가 메타데이터 핵심 필드**
| Column Name | Description |
|-------------|------------|
| `Listeners` | 곡 청취자 수 |
| `Length` | 곡 길이 |
| `Genre` | 장르 태그 |
| `Introduction` | 곡 소개 |

<br><br>

## Appendix
- [발표 자료](https://drive.google.com/file/d/1NEFwS3l9uHlbgT2XElacOubI_-_FsrEs/view)
