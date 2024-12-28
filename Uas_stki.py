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
file_path = "C:/Users/rehan/Documents/Uas STKI/tmdb_5000_movies.csv"
movies = pd.read_csv(file_path)

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
    if 'normalized_vote_average' in movies.columns and 'normalized_vote_count' in movies.columns:
        movies['final_score'] = (
            vote_weight * movies['normalized_vote_average'] +
            count_weight * movies['normalized_vote_count']
        )
    else:
        st.error("Kolom 'normalized_vote_average' atau 'normalized_vote_count' tidak ditemukan. Periksa proses normalisasi.")

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
st.slider("Bobot Popularitas (Vote Count)", 0.0, 1.0, count_weight)

if 'vote_weight' not in st.session_state:
    st.session_state.vote_weight = vote_weight
if 'count_weight' not in st.session_state:
    st.session_state.count_weight = count_weight

if st.button("Terapkan Bobot"):
    st.session_state.vote_weight = vote_weight
    st.session_state.count_weight = count_weight
    calculate_final_score(st.session_state.vote_weight, st.session_state.count_weight)
    st.success(f"Bobot telah diterapkan: \n- Bobot Skor Review: {st.session_state.vote_weight}\n- Bobot Popularitas: {st.session_state.count_weight}")

# Fungsi Rekomendasi
def recommend(title, cosine_sim=cosine_sim, movies=movies, genre_filter=None):
    try:
        if 'final_score' not in movies.columns:
            calculate_final_score(st.session_state.vote_weight, st.session_state.count_weight)
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

# Fungsi Menampilkan Film Terbaik
def get_top_movies(movies, genre_filter=None, n=30):
    if 'final_score' not in movies.columns:
        calculate_final_score(st.session_state.vote_weight, st.session_state.count_weight)
    top_movies = movies
    if genre_filter and genre_filter.strip():
        genre_filter = genre_filter.lower()
        top_movies = top_movies[top_movies['genres'].str.contains(genre_filter, case=False, na=False)]
    return top_movies.sort_values(by='final_score', ascending=False).head(n)

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

if tab_selection == "Rekomendasi":
    st.subheader(f"Top 30 Film Terbaik (Genre: {selected_genre}):")
    st.dataframe(get_top_movies(movies, genre_filter=genre_filter, n=30))

if tab_selection == "Tentang":
    st.write("Aplikasi ini adalah platform cerdas yang memanfaatkan teknologi Natural Language Processing (NLP) dengan model Transformer untuk memberikan rekomendasi film yang akurat dan relevan. Dengan memahami preferensi pengguna melalui analisis data teks, seperti ulasan, deskripsi film, dan komentar pengguna, aplikasi ini mampu menghasilkan rekomendasi yang benar-benar personal dan dinamis.")

if tab_selection == "Informasi Dataset":
    st.subheader("Statistik Dataset")
    st.write(f"Jumlah Film: {len(movies)}")
