import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import streamlit as st
import os
from sklearn.preprocessing import MinMaxScaler
from fuzzywuzzy import process
import altair as alt
from streamlit_lottie import st_lottie
import requests
import matplotlib.pyplot as plt

# Fungsi untuk memuat animasi Lottie
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Tema Visual dengan CSS
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
        font-family: 'Arial', sans-serif;
    }
    h1, h2, h3 {
        color: #00695c;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header dengan Desain Menarik
st.markdown(
    """
    <div style="background-color:#e0f7fa; padding:20px; border-radius:10px;">
        <h1 style="color:#00695c; text-align:center;">Sistem Rekomendasi Film</h1>
        <p style="text-align:center; color:#004d40;">Temukan film favorit Anda dengan teknologi AI canggih</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Langkah 2: Memuat dataset
file_path = "tmdb_5000_movies.csv"

if os.path.exists(file_path):
    try:
        movies = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        st.error("File dataset kosong. Harap unggah file yang valid.")
        st.stop()
    except Exception as e:
        st.error(f"Gagal membaca dataset: {str(e)}")
        st.stop()
else:
    uploaded_file = st.file_uploader("Unggah file dataset CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            movies = pd.read_csv(uploaded_file)
        except pd.errors.EmptyDataError:
            st.error("File yang diunggah kosong. Harap unggah file yang valid.")
            st.stop()
        except Exception as e:
            st.error(f"Gagal membaca dataset yang diunggah: {str(e)}")
            st.stop()
    else:
        st.error("Dataset tidak ditemukan. Harap unggah file dataset atau pastikan path file benar.")
        st.stop()

# Langkah 3: Memilih kolom yang relevan
movies = movies[['title', 'overview', 'genres', 'vote_average', 'vote_count']]
movies = movies.dropna()  # Menghapus data kosong

# Langkah 4: Pra-pemrosesan Genres
def extract_genres(genres_str):
    try:
        genres_list = ast.literal_eval(genres_str)
        return ' '.join([genre['name'] for genre in genres_list])
    except Exception as e:
        return ''

movies['genres'] = movies['genres'].apply(extract_genres)

# Langkah 5: Kombinasi Overview dan Genres
movies['content'] = movies['overview'] + ' ' + movies['genres']

# Langkah 6: Representasi Teks
count_vectorizer = CountVectorizer(stop_words='english')
count_matrix = count_vectorizer.fit_transform(movies['content'])

# Langkah 7: Menghitung Kemiripan Kosinus
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Langkah 8: Normalisasi Skor dan Popularitas
scaler = MinMaxScaler()
movies['normalized_vote_average'] = scaler.fit_transform(movies[['vote_average']])
movies['normalized_vote_count'] = scaler.fit_transform(movies[['vote_count']])

# Langkah 9: Menggabungkan Skor
def calculate_final_score(vote_weight, count_weight):
    movies['final_score'] = (
        vote_weight * movies['normalized_vote_average'] +
        count_weight * movies['normalized_vote_count']
    )

# Pengaturan Bobot
st.subheader("Pengaturan Bobot")
st.markdown(
    """
    Pastikan total bobot (Skor Review + Popularitas) adalah 1.0 untuk menjaga keseimbangan.
    Contoh: Jika Bobot Skor Review adalah 0.7, maka Bobot Popularitas harus 0.3.
    """
)

vote_weight = st.slider("Bobot Skor Review (Vote Average)", 0.0, 1.0, 0.7)
count_weight = 1.0 - vote_weight

if st.button("Terapkan Bobot"):
    calculate_final_score(vote_weight, count_weight)
    st.success(f"Bobot telah diterapkan: \n- Bobot Skor Review: {vote_weight}\n- Bobot Popularitas: {count_weight}")

# Fungsi Rekomendasi
def recommend(title, cosine_sim=cosine_sim, movies=movies, genre_filter=None):
    try:
        idx = movies[movies['title'] == title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:21]
        movie_indices = [i[0] for i in sim_scores]

        recommended_movies = movies.iloc[movie_indices]
        if genre_filter and genre_filter.strip():
            genre_filter = genre_filter.lower()
            recommended_movies = recommended_movies[recommended_movies['genres'].str.contains(genre_filter, case=False, na=False)]

        return recommended_movies.sort_values(by='final_score', ascending=False).head(10)
    except IndexError:
        st.error("Film tidak ditemukan dalam dataset. Pastikan judul yang dimasukkan benar.")
        return pd.DataFrame()

# Sidebar
with st.sidebar:
    st.markdown("Navigasi", unsafe_allow_html=True)
    tab_selection = st.radio("Pilih Halaman", ["Daftar Film", "Rekomendasi", "Tentang", "Informasi Dataset"])
    all_genres = sorted(set(" ".join(movies['genres']).split()))
    selected_genre = st.selectbox("Pilih genre untuk rekomendasi:", ["Semua"] + all_genres)
    genre_filter = selected_genre if selected_genre != "Semua" else None

if tab_selection == "Daftar Film":
    st.subheader("Daftar Film yang Tersedia:")
    filtered_movies = (
        movies[movies['genres'].str.contains(genre_filter, case=False, na=False)]
        if genre_filter else movies
    )
    st.write(filtered_movies['title'].tolist())
