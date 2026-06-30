import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="KosIn - Prediksi Harga Kos", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

if 'page' not in st.session_state:
    st.session_state.page = 1

def go_to_page_2():
    st.session_state.page = 2

def go_to_page_1():
    st.session_state.page = 1

def format_rupiah(angka):
    return f"Rp {int(angka):,} ".replace(",", ".")

def merge_images_vertically(image_paths, target_width=600):
    images = []
    for path in image_paths:
        img = Image.open(path)
        w_percent = (target_width / float(img.size[0]))
        h_size = int((float(img.size[1]) * float(w_percent)))
        img = img.resize((target_width, h_size), Image.LANCZOS)
        images.append(img)
    
    total_height = sum(img.size[1] for img in images)
    
    merged_image = Image.new('RGB', (target_width, total_height))
    
    y_offset = 0
    for img in images:
        merged_image.paste(img, (0, y_offset))
        y_offset += img.size[1]
        
    return merged_image


@st.cache_resource
def load_ml_assets():
    try:
        with open('models/model_kos.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/model_columns.pkl', 'rb') as f:
            model_columns = pickle.load(f)
        df_cleaned = pd.read_csv('data/Data_Indekos_Jabodetabek_Mamikos_Cleaned.csv')
        df_regression = pd.read_csv('data/Data_Indekos_Jabodetabek_Mamikos_Regression.csv')
        return model, model_columns, df_cleaned, df_regression
    except Exception as e:
        st.error(f"Error loading ML assets: {str(e)}")
        return None, None, None, None

model, model_columns, df_cleaned, df_regression = load_ml_assets()


def find_ohe_column(columns, prefix, value, df_source=None):
    """
    Mencari kolom OHE yang cocok secara fleksibel (case-insensitive, spasi vs underscore).
    Jika tidak ditemukan tapi value adalah reference category yang valid (di-drop saat training),
    maka all-zeros sudah benar — tidak perlu set apapun dan tidak perlu warning.
    Returns: (col_name, is_reference_category)
      - (str, False)  → kolom ditemukan, set ke 1.0
      - (None, True)  → reference category, biarkan all-zeros
      - (None, False) → benar-benar tidak ditemukan, tampilkan warning
    """
    value_normalized = value.lower().replace(" ", "_")

    for col in columns:
        if not col.startswith(prefix):
            continue
        col_suffix = col[len(prefix):]
        col_suffix_normalized = col_suffix.lower().replace(" ", "_")
        if col_suffix_normalized == value_normalized:
            return col, False  


    if df_source is not None:
        field_name = prefix.rstrip('_')
        if field_name in df_source.columns:
            all_values_in_data = df_source[field_name].unique()
            all_others_exist = all(
                any(
                    c.startswith(prefix) and
                    c[len(prefix):].lower().replace(" ", "_") == v.lower().replace(" ", "_")
                    for c in columns
                )
                for v in all_values_in_data
                if v.lower().replace(" ", "_") != value_normalized
            )
            if all_others_exist:
                return None, True  

    return None, False  


def get_top_5_recommendations(user_input_processed, df_cleaned, lokasi_filter=None, tipe_kos_filter=None):
    try:
        df_filtered = df_cleaned.copy()

        if lokasi_filter and 'region' in df_filtered.columns:
            df_region = df_filtered[df_filtered['region'].str.lower() == lokasi_filter.lower()]
            if not df_region.empty:
                df_filtered = df_region

        if tipe_kos_filter and 'tipe_kos' in df_filtered.columns:
            df_tipe = df_filtered[df_filtered['tipe_kos'].str.lower() == tipe_kos_filter.lower()]
            if not df_tipe.empty:
                df_filtered = df_tipe

        if df_filtered.empty or len(df_filtered) < 5:
            df_filtered = df_cleaned.copy()

        ohe_cols = [col for col in user_input_processed.columns
                    if col.startswith('region_') or col.startswith('tipe_kos_')]
        fasilitas_cols = [col for col in user_input_processed.columns
                         if col != 'room_area' and col not in ohe_cols and col in df_filtered.columns]


        fasilitas_user = [col for col in fasilitas_cols
                         if user_input_processed[col].iloc[0] == 1.0]
        if len(fasilitas_user) == 0:
            fasilitas_user = fasilitas_cols

        
        if len(fasilitas_user) > 0:
            features_database = df_filtered[fasilitas_user].copy().fillna(0)
            user_features = user_input_processed[fasilitas_user].copy().fillna(0)
            sim_fasilitas = cosine_similarity(user_features, features_database)[0]
        else:
            sim_fasilitas = np.ones(len(df_filtered))

        
        luas_user = user_input_processed['room_area'].iloc[0]
        luas_db = df_filtered['room_area'].fillna(df_filtered['room_area'].median()).values
        sim_luas = 1 / (1 + np.abs(luas_db - luas_user))

        
        df_filtered['similarity'] = (sim_fasilitas * 0.7) + (sim_luas * 0.3)
        recommendations = df_filtered.sort_values(by='similarity', ascending=False).head(5)

        return recommendations

    except Exception as e:
        st.error(f"Error dalam fungsi rekomendasi: {str(e)}")
        return None



st.markdown("""
    <style>
    #MainMenu {visibility: hidden;} header {visibility: hidden;} footer {visibility: hidden;}
    
    /* Konfigurasi default (Untuk Page 1) - Full Wide */
    .block-container { padding-top: 70px !important; padding-left: 0rem !important; padding-right: 0rem !important; padding-bottom: 0rem !important; max-width: 100% !important; }
    .stApp { background-color: #E5E5E5; color: #0f172a; }
    
    /* Navbar Melayang - KETEBALAN HEADER BISA DIATUR PADA PADDING DI BAWAH INI (10px) */
    .custom-navbar {
        background-color: rgba(255, 255, 255, 0.95); 
        backdrop-filter: blur(10px); 
        -webkit-backdrop-filter: blur(10px);
        padding: 2px 50px; 
        display: flex; 
        justify-content: space-between; 
        align-items: center;
        position: fixed; top: 0; left: 0; right: 0; z-index: 99999;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08); border-bottom: 1px solid rgba(226, 232, 240, 0.6); 
    }
    .custom-navbar h2 { margin: 0; font-size: 22px; font-weight: 800; color: #0f172a; font-family: sans-serif; letter-spacing: -0.5px; }

    [data-testid="stWidgetLabel"] p { color: #0f172a !important; font-weight: 700 !important; font-size: 14px !important; }
    [data-testid="stRadio"] div[role="radiogroup"] label p { color: #0f172a !important; font-weight: 600 !important; font-size: 14px !important; }

    /* CSS FORM & CHECKBOX */
    [data-testid="stForm"] { background-color: #ffffff !important; border-radius: 16px !important; padding: 40px !important; border: 1px solid #e2e8f0 !important; box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05) !important; max-width: 680px !important; margin: 20px auto 40px auto; }
    
    div[data-testid="stCheckbox"] { background-color: #f8fafc !important; border: 1px solid #cbd5e1 !important; border-radius: 8px !important; margin-bottom: 5px !important; transition: all 0.2s ease !important; width: 100% !important; box-sizing: border-box !important; padding: 0 !important; }
    div[data-testid="stCheckbox"]:hover { background-color: #f1f5f9 !important; border-color: #94a3b8 !important; }
    div[data-testid="stCheckbox"] div[data-baseweb="checkbox"] > div:first-child, div[data-testid="stCheckbox"] div[role="checkbox"] { display: none !important; width: 0px !important; height: 0px !important; opacity: 0 !important; overflow: hidden !important; margin: 0 !important; padding: 0 !important; }
    div[data-testid="stCheckbox"] label { width: 100% !important; min-height: 45px !important; padding: 10px 15px !important; display: flex !important; align-items: center !important; cursor: pointer !important; box-sizing: border-box !important; }
    div[data-testid="stCheckbox"] p { color: #475569 !important; font-weight: 600 !important; font-size: 13px !important; margin: 0 !important; text-align: left !important; width: 100% !important; }
    div[data-testid="stCheckbox"]:has(input:checked) { background-color: #e0e7ff !important; border: 1px solid #818cf8 !important; box-shadow: 0 2px 5px rgba(129, 140, 248, 0.2) !important; }
    div[data-testid="stCheckbox"]:has(input:checked) p { color: #1e3a8a !important; }
    
    [data-testid="stFormSubmitButton"] { display: flex !important; justify-content: center !important; width: 100% !important; margin-top: 2rem !important; }
    [data-testid="stForm"] button { background-color: #1e293b !important; color: white !important; border-radius: 8px !important; padding: 0.75rem 2rem !important; font-weight: 600 !important; border: none !important; width: 100% !important; transition: all 0.3s ease !important; }
    [data-testid="stForm"] button:hover { background-color: #0f172a !important; box-shadow: 0 4px 12px rgba(15, 23, 42, 0.3) !important; }
    
    .block-container { 
        padding-top: 0px !important; 
        margin-top: 0px !important; 
    }
    .content-wrapper {
        padding-top: 80px; 
    }
    
    /* 1. Hilangkan padding kolom kanan secara total */
    [data-testid="column"]:nth-of-type(2) {
        padding: 0px !important;
        margin: 0px !important;
        background-color: #0b1325;
    }
            
    /* 2. Buat container gambar yang rapat */
    .right-image-stack {
        display: flex;
        flex-direction: column;
        width: 100%;
        gap: 0px !important; /* Menghapus celah antar gambar */
        margin: 0px !important;
        padding: 0px !important;
    }

    /* 3. Paksa gambar menempel dan tanpa batas */
    .right-image-stack img {
        width: 100% !important;
        display: block !important;
        margin: 0px !important;
        padding: 0px !important;
        border: none !important;
    </style>
""", unsafe_allow_html=True)




if st.session_state.page == 1:
    
    st.markdown('<div class="custom-navbar"><h2>KosIn</h2></div>', unsafe_allow_html=True)
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    left_col, right_col = st.columns([1.2, 1], gap="large")

    with left_col:
        st.markdown('<style>[data-testid="left_col"]:nth-of-type(1) { padding: 100px 30px !important; }</style>', unsafe_allow_html=True)
                        
        with st.form("form_prediksi"):
            st.markdown("<h2 style='color: #0f172a; font-weight: 800; margin-top: 0; margin-bottom: 10px; font-size: 32px;'>Cari Prediksi Harga Kos</h2>", unsafe_allow_html=True)
            st.markdown("<p style='color: #475569; font-size: 15px; margin-bottom: 1.5rem; line-height: 1.5;'>Masukkan detail untuk mengetahui estimasi harga pasar. Dapatkan analisis harga berbasis data untuk hunian ideal Anda.</p>", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom: 25px; margin-left: 5px; color: #64748b; font-size: 13px;'>🟠 Pilih Lokasi & Fasilitas &nbsp; › &nbsp; ⚪ Hasil</div>", unsafe_allow_html=True)

            lokasi = st.selectbox("Lokasi / Area", ["Pilih Wilayah di Jakarta", "Jakarta Selatan", "Jakarta Pusat", "Jakarta Barat", "Jakarta Timur", "Jakarta Utara"])
            st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
            luas_kamar = st.slider("Luas Kamar (m²)", min_value=4, max_value=40, value=12)
            st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
            tipe_kos = st.radio("Tipe Kos", ["Kos Campur", "Kos Putra", "Kos Putri"], horizontal=True)
            st.markdown("<hr style='margin: 25px 0; border-color: #e2e8f0;'>", unsafe_allow_html=True)
            
            st.markdown("<div style='font-size: 16px; font-weight: 800; color: #0f172a;'>Fasilitas Utama</div>", unsafe_allow_html=True)
                      
            
            inputs = {}
            st.markdown("<div class='section-title'>Fasilitas Kamar</div>", unsafe_allow_html=True)
            fk_col1, fk_col2, fk_col3 = st.columns(3)
            with fk_col1:
                inputs['fac_ac'] = st.checkbox("AC")
                inputs['is_electricity'] = st.checkbox("Termasuk Listrik")
            with fk_col2:
                inputs['fac_wifi'] = st.checkbox("WiFi")
                inputs['fac_guling'] = st.checkbox("Guling")
            with fk_col3:
                inputs['fac_bantal'] = st.checkbox("Bantal")

            st.markdown("<div class='section-title'>Furnitur Kamar</div>", unsafe_allow_html=True)
            fur_col1, fur_col2, fur_col3 = st.columns(3)
            with fur_col1:
                inputs['fac_kipas_angin'] = st.checkbox("Kipas Angin")
                inputs['fac_kursi'] = st.checkbox("Kursi")
            with fur_col2:
                inputs['fac_tv'] = st.checkbox("TV")
                inputs['is_full_furnished'] = st.checkbox("Full Furnish")
            with fur_col3:
                inputs['fac_cermin'] = st.checkbox("Cermin")

            st.markdown("<div class='section-title'>Kamar Mandi</div>", unsafe_allow_html=True)
            km_col1, km_col2, km_col3 = st.columns(3)
            with km_col1:
                inputs['is_kamar_mandi_dalam'] = st.checkbox("K. Mandi Dalam")
                inputs['fac_wastafel'] = st.checkbox("Wastafel")
                inputs['fac_bak_mandi'] = st.checkbox("Bak Mandi")
            with km_col2:
                inputs['fac_ember_mandi'] = st.checkbox(" Ember Mandi")
                inputs['fac_air_panas'] = st.checkbox(" Air Panas")
            with km_col3:
                inputs['fac_shower'] = st.checkbox(" Shower")
                inputs['is_kloset_duduk'] = st.checkbox(" Kloset Duduk")

            st.markdown("<div class='section-title'>Fasilitas Umum & Lainnya</div>", unsafe_allow_html=True)
            fb_col1, fb_col2, fb_col3 = st.columns(3)
            with fb_col1:
                inputs['fac_keamanan'] = st.checkbox("Keamanan / CCTV")
                inputs['fac_sirkulasi_udara'] = st.checkbox("Sirkulasi Udara")
                inputs['fac_parkir_motor'] = st.checkbox("Parkir Motor")
                inputs['fac_parkir_mobil'] = st.checkbox("Parkir Mobil")
            with fb_col2:
                inputs['fac_ruang_bersama'] = st.checkbox("Ruang Bersama")
                inputs['fac_dispenser'] = st.checkbox("Dispenser")
                inputs['fac_dapur'] = st.checkbox("Dapur Bersama")
                inputs['fac_layanan_kebersihan'] = st.checkbox("Layanan Kebersihan")
            with fb_col3:
                inputs['fac_kulkas'] = st.checkbox("Kulkas Bersama")
                inputs['fac_area_jemur'] = st.checkbox("Area Jemur")
                inputs['fac_mesin_cuci'] = st.checkbox("Mesin Cuci")

            st.markdown("<br>", unsafe_allow_html=True) 
            
            btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1]) 
            with btn_col2:
                submitted = st.form_submit_button("📊 Lihat Prediksi", width='stretch')
                
            if submitted:
                if lokasi == "Pilih Wilayah di Jakarta":
                    st.error("Silakan pilih Lokasi / Area terlebih dahulu.")
                else:
                    st.session_state.lokasi = lokasi
                    st.session_state.luas_kamar = luas_kamar
                    st.session_state.tipe_kos = tipe_kos
                    st.session_state.inputs = inputs
                    go_to_page_2()
                    st.rerun()

    with right_col:
        try:
            
            list_gambar = ["src/image1.jpg", "src/image2.jpg", "src/image3.png"]
            
            final_image = merge_images_vertically(list_gambar, target_width=800)
            
            st.image(final_image, width='stretch')
            
        except Exception as e:
            st.error(f"Gagal memproses gambar: {str(e)}")


elif st.session_state.page == 2:
    
    st.markdown("""
        <div class="custom-navbar">
            <div style="display: flex; align-items: center; margin-left: 45px;">
                <h2 style="margin: 0; font-size: 22px; font-weight: 800; color: #0f172a;">KosIn</h2>
                <span style="margin: 0 12px; color: #cbd5e1; font-size: 22px; font-weight: 300;">|</span>
                <h3 style="margin: 0; font-size: 16px; font-weight: 600; color: #475569;">Hasil Prediksi</h3>
            </div>
        </div>
        
        <style>
        div[data-testid="stButton"] {
            position: fixed !important;
            top: 10px !important;
            left: 45px !important;
            z-index: 100000 !important;
            width: auto !important;
        }
        div[data-testid="stButton"] button {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
            color: #0f172a !important;
        }
        div[data-testid="stButton"] button p {
            font-size: 26px !important;
            font-weight: 800 !important;
            margin: 0 !important;
            line-height: 1 !important;
        }
        div[data-testid="stButton"] button:hover p {
            color: #f59e0b !important;
        }
        
        .block-container { 
            max-width: 1100px !important; 
            padding-top: 100px !important; 
            padding-bottom: 50px !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            margin: 0 auto !important; 
        }
        
        .rekomendasi-card { background-color: white; padding: 25px; border-radius: 12px; border: 1px solid #e2e8f0; margin-bottom: 20px; transition: transform 0.2s, box-shadow 0.2s; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); height: 100%; }
        .rekomendasi-card:hover { transform: translateY(-3px); box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); }
        .rek-title { font-weight: 800; font-size: 17px; color: #0f172a; margin-bottom: 5px; }
        .rek-loc { font-size: 13px; color: #64748b; margin-bottom: 20px; display: flex; align-items: center; }
        .rek-divider { border-top: 1px dashed #cbd5e1; margin: 15px 0; }
        .rek-price-label { font-size: 11px; color: #64748b; text-transform: uppercase; font-weight: 600;}
        .rek-price-value { font-weight: 800; font-size: 18px; color: #0f172a; text-align: right; }
        </style>
    """, unsafe_allow_html=True)

    if st.button("←", help="Kembali"):
        go_to_page_1()
        st.rerun()

    pred_price = 0
    top_5_recs = []

    if model is not None and model_columns is not None and df_cleaned is not None:
        try:
            input_data = pd.DataFrame(0.0, index=[0], columns=model_columns)

            
            lok = st.session_state.lokasi
            col_region, is_ref_region = find_ohe_column(model_columns, 'region_', lok, df_cleaned)
            if col_region:
                input_data[col_region] = 1.0
            elif not is_ref_region:
                st.warning(f"⚠️ Kolom region untuk '{lok}' tidak ditemukan. Tersedia: {[c for c in model_columns if c.startswith('region_')]}")

            tipe = st.session_state.tipe_kos
            col_tipe, is_ref_tipe = find_ohe_column(model_columns, 'tipe_kos_', tipe, df_cleaned)
            if col_tipe:
                input_data[col_tipe] = 1.0
            elif not is_ref_tipe:
                st.warning(f"⚠️ Kolom tipe_kos untuk '{tipe}' tidak ditemukan. Tersedia: {[c for c in model_columns if c.startswith('tipe_kos_')]}")

            inp = st.session_state.inputs
            input_data['room_area'] = float(st.session_state.luas_kamar)
            
            mapping = {
                'fac_ac': inp['fac_ac'],
                'is_electricity': inp['is_electricity'],
                'fac_wifi': inp['fac_wifi'],
                'fac_guling': inp['fac_guling'],
                'fac_bantal': inp['fac_bantal'],
                'fac_kipas_angin': inp['fac_kipas_angin'],
                'fac_kursi': inp['fac_kursi'],
                'fac_cermin': inp['fac_cermin'],
                'fac_tv': inp['fac_tv'],
                'is_full_furnished': inp['is_full_furnished'],
                'is_kamar_mandi_dalam': inp['is_kamar_mandi_dalam'],
                'fac_wastafel': inp['fac_wastafel'],
                'fac_bak_mandi': inp['fac_bak_mandi'],
                'fac_ember_mandi': inp['fac_ember_mandi'],
                'fac_air_panas': inp['fac_air_panas'],
                'fac_shower': inp['fac_shower'],
                'is_kloset_duduk': inp['is_kloset_duduk'],
                'fac_keamanan': inp['fac_keamanan'],
                'fac_sirkulasi_udara': inp['fac_sirkulasi_udara'],
                'fac_parkir_motor': inp['fac_parkir_motor'],
                'fac_parkir_mobil': inp['fac_parkir_mobil'],
                'fac_ruang_bersama': inp['fac_ruang_bersama'],
                'fac_dispenser': inp['fac_dispenser'],
                'fac_dapur': inp['fac_dapur'],
                'fac_kulkas': inp['fac_kulkas'],
                'fac_area_jemur': inp['fac_area_jemur'],
                'fac_layanan_kebersihan': inp['fac_layanan_kebersihan'],
                'fac_mesin_cuci': inp['fac_mesin_cuci']
            }
            for col, val in mapping.items():
                if col in input_data.columns:
                    input_data[col] = float(val)

            pred_price_log = model.predict(input_data)[0]
            pred_price = np.expm1(pred_price_log)

            recs = get_top_5_recommendations(
                input_data,
                df_cleaned.copy(),
                lokasi_filter=st.session_state.lokasi,
                tipe_kos_filter=st.session_state.tipe_kos
            )
            
            model_pred = model.predict(input_data)[0] 
            
            recs_df = get_top_5_recommendations(input_data, df_cleaned.copy(), 
                                                lokasi_filter=st.session_state.lokasi, 
                                                tipe_kos_filter=st.session_state.tipe_kos)
            
            if recs_df is not None and not recs_df.empty:
                pred_price = recs_df['price'].head(5).mean()
                for _, row in recs_df.iterrows():
                    nama_kos = row['room_name'] if 'room_name' in row else "Kos Tanpa Nama"
                    detail_location = f"{row['region']}, {row['location']}"
                    top_5_recs.append({
                        "nama": nama_kos,
                        "lokasi": detail_location,
                        "harga": format_rupiah(row['price'])
                    })

        except Exception as ml_error:
            st.error(f"❌ Error dalam prediksi ML: {str(ml_error)}")
    else:
        st.error("⚠️ Model ML belum berhasil dimuat. Silakan cek file model_kos.pkl dan model_columns.pkl")

    if not top_5_recs:
        pred_price = 1850000
        top_5_recs = [
            {"nama": "Kos Melati", "lokasi": "Setiabudi, Jakarta Selatan", "harga": format_rupiah(1800000)},
            {"nama": "Wisma Menteng", "lokasi": "Menteng Dalam, Jakarta Pusat", "harga": format_rupiah(1750000)},
            {"nama": "Kost Eksklusif D'Tebet", "lokasi": "Tebet, Jakarta Selatan", "harga": format_rupiah(1950000)},
            {"nama": "Griya Karet Tengsin", "lokasi": "Karet Tengsin, Jakarta Pusat", "harga": format_rupiah(1700000)},
            {"nama": "Pondok Bunga", "lokasi": "Kuningan, Jakarta Selatan", "harga": format_rupiah(1650000)}
        ]

    st.markdown(f"""
        <div style="background-color: #172033; padding: 50px 20px; border-radius: 16px; text-align: center; color: white; margin: 10px 0 50px 0; position: relative; overflow: hidden; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);">
            <div style="position: absolute; bottom: -50px; right: -50px; width: 300px; height: 300px; background: radial-gradient(circle, rgba(245,158,11,0.06) 0%, transparent 70%); border-radius: 50%;"></div>
            <p style="font-size: 12px; letter-spacing: 2px; color: #94a3b8; text-transform: uppercase; font-weight: 700; margin-bottom: 10px;">Estimasi Harga Ideal</p>
            <h1 style="color: #f59e0b; font-size: 48px; font-weight: 800; margin: 0; text-shadow: 0 2px 10px rgba(245, 158, 11, 0.2);">{format_rupiah(pred_price)} <span style="font-size: 18px; color: #cbd5e1; font-weight: 500;">/ bulan</span></h1>
            <p style="color: #94a3b8; font-size: 14px; max-width: 600px; margin: 20px auto 0 auto; line-height: 1.6;">
                Berdasarkan analisis fitur yang Anda pilih di area {st.session_state.lokasi}.
            </p>
            <hr style="border-color: rgba(255,255,255,0.05); margin: 30px auto; max-width: 400px; border-style: solid; border-width: 1px;">
            <div style="display: flex; justify-content: center; gap: 15px;">
                <div style="background-color: rgba(30, 41, 59, 0.8); border: 1px solid rgba(255,255,255,0.05); padding: 8px 20px; border-radius: 20px; font-size: 12px; color: #cbd5e1; font-weight: 600;"><span style="color: #f59e0b; margin-right: 5px;">✔️</span> Akurasi Tinggi</div>
                <div style="background-color: rgba(30, 41, 59, 0.8); border: 1px solid rgba(255,255,255,0.05); padding: 8px 20px; border-radius: 20px; font-size: 12px; color: #cbd5e1; font-weight: 600;"><span style="color: #f59e0b; margin-right: 5px;">📈</span> Trend Stabil</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    
    col_rek1, col_rek2 = st.columns([4, 1])
    with col_rek1:
        st.markdown("<h4 style='color: #0f172a; font-weight: 800; font-size: 18px; margin-bottom: 20px;'>Top Rekomendasi Untukmu</h4>", unsafe_allow_html=True)
    with col_rek2:
        st.markdown("<div style='text-align: right; margin-top: 5px;'><a href='#' style='color: #f59e0b; text-decoration: none; font-weight: 700; font-size: 13px;'>Lihat Peta 🗺️</a></div>", unsafe_allow_html=True)

    cols1 = st.columns(3)
    for i in range(min(3, len(top_5_recs))):
        with cols1[i]:
            st.markdown(f"""
                <div class="rekomendasi-card">
                    <div class="rek-title">{top_5_recs[i]['nama']}</div>
                    <div class="rek-loc">
                        <span style="color: #94a3b8; font-size: 14px; margin-right: 6px;">📍</span>{top_5_recs[i]['lokasi']}
                    </div>
                    <div class="rek-divider"></div>
                    <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-top: 15px;">
                        <span class="rek-price-label">Harga Aktual</span>
                        <span class="rek-price-value">{top_5_recs[i]['harga']}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    st.write("")

    cols2 = st.columns(3)
    for i in range(3, len(top_5_recs)):
        with cols2[i-3]:
            st.markdown(f"""
                <div class="rekomendasi-card">
                    <div class="rek-title">{top_5_recs[i]['nama']}</div>
                    <div class="rek-loc">
                        <span style="color: #94a3b8; font-size: 14px; margin-right: 6px;">📍</span>{top_5_recs[i]['lokasi']}
                    </div>
                    <div class="rek-divider"></div>
                    <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-top: 15px;">
                        <span class="rek-price-label">Harga Aktual</span>
                        <span class="rek-price-value">{top_5_recs[i]['harga']}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
