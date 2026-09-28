import os, io, pickle, sqlite3
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASS_FILE = os.path.join(BASE_DIR, 'class_indices.pkl')
WEIGHTS_FILE = os.path.join(BASE_DIR, 'model_weights.weights.h5')
EXCEL_FILE = os.path.join(BASE_DIR, 'sci -123.xlsx')
DB_FILE = os.path.join(BASE_DIR, 'plantmed.db')

st.set_page_config(page_title='PLANTMED', page_icon='🌿', layout='wide', initial_sidebar_state='expanded')

@st.cache_data

def load_data():
    df = pd.read_excel(EXCEL_FILE).fillna('')
    return df

@st.cache_data

def load_classes():
    with open(CLASS_FILE, 'rb') as f:
        ci = pickle.load(f)
    return {int(v): k for k, v in ci.items()}

@st.cache_resource

def load_model():
    idx_to_class = load_classes()
    base = keras.applications.InceptionV3(weights=None, include_top=False, input_shape=(224,224,3))
    model = keras.Sequential([base, layers.GlobalAveragePooling2D(), layers.Dense(256, activation='relu'), layers.Dropout(.5), layers.Dense(128, activation='relu'), layers.Dropout(.3), layers.Dense(len(idx_to_class), activation='softmax')])
    model.build((None,224,224,3))
    model.load_weights(WEIGHTS_FILE)
    return model

def db_init():
    with sqlite3.connect(DB_FILE) as con:
        con.execute('''CREATE TABLE IF NOT EXISTS observations (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, plant TEXT, confidence REAL, latitude REAL, longitude REAL, verified INTEGER DEFAULT 0, notes TEXT)''')
        con.execute('''CREATE TABLE IF NOT EXISTS expert_reviews (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, plant TEXT, reviewer TEXT, status TEXT, comments TEXT)''')

def save_observation(plant, confidence, lat, lon, notes=''):
    with sqlite3.connect(DB_FILE) as con:
        con.execute('INSERT INTO observations(created_at,plant,confidence,latitude,longitude,notes) VALUES(?,?,?,?,?,?)', (datetime.utcnow().isoformat(), plant, confidence, lat, lon, notes))

def save_review(plant, reviewer, status, comments):
    with sqlite3.connect(DB_FILE) as con:
        con.execute('INSERT INTO expert_reviews(created_at,plant,reviewer,status,comments) VALUES(?,?,?,?,?)', (datetime.utcnow().isoformat(), plant, reviewer, status, comments))

def prepare_image(upload):
    img = Image.open(upload).convert('RGB')
    arr = keras_image.img_to_array(img.resize((224,224)))
    return img, preprocess_input(np.expand_dims(arr, 0))

def plant_details(df, name):
    m = df[df['Scientific_name'].astype(str).str.strip().str.lower() == name.strip().lower()]
    return m.iloc[0].to_dict() if not m.empty else {}

def confidence_label(c):
    if c >= 85: return 'High confidence', 'success'
    if c >= 60: return 'Moderate confidence', 'warning'
    return 'Low confidence — verify before use', 'error'

def safety_for(name, confidence):
    # Conservative, project-level safety layer. It does not diagnose, prescribe, or determine toxicity.
    text = f'Identification confidence is {confidence:.1f}%. Do not consume a plant based only on an AI prediction. Confirm species and plant part with a qualified botanist/health professional.'
    known = {'Abrus precatorius': 'HIGH CAUTION: seeds are known to contain potent toxins. Do not ingest based on this application.'}
    return known.get(name, text)

def evidence_links(name):
    q = name.replace(' ', '+')
    return [f'https://pubmed.ncbi.nlm.nih.gov/?term={q}', f'https://scholar.google.com/scholar?q={q}+medicinal+plant']

def assistant_answer(question, df):
    q = question.lower().strip()
    if not q: return 'Ask me about a plant, traditional uses, plant parts, preparation, safety, or research.'
    for _, r in df.iterrows():
        name = str(r.get('Scientific_name',''))
        common = str(r.get('Kannada Name',''))
        if name and (name.lower() in q or common.lower() in q):
            if 'use' in q or 'medic' in q: return f"{name}: documented project database uses include {r.get('Uses','N/A')}. These are database/traditional-use notes, not a prescription."
            if 'part' in q: return f"{name}: listed parts are {r.get('Parts_used','N/A')}. Verify the correct species and part before use."
            if 'where' in q or 'grow' in q: return f"{name}: the database lists {r.get('Grown_Area','N/A')}."
            return f"{name} ({common}): {r.get('Uses','N/A')}. Parts: {r.get('Parts_used','N/A')}."
    return 'I could not match that plant in the local PLANTMED database. Try the scientific name or use the Identify Plant page.'

# Init
_db_ok = True
try: db_init()
except Exception: _db_ok = False

df = load_data()
idx_to_class = load_classes()

st.markdown('''<style> .hero{padding:1.3rem 1.5rem;border-radius:18px;border:1px solid rgba(128,128,128,.25);background:linear-gradient(135deg,rgba(46,125,50,.14),rgba(76,175,80,.04));} .small{opacity:.75;font-size:.9rem} .card{padding:1rem;border:1px solid rgba(128,128,128,.25);border-radius:14px;margin-bottom:.8rem} </style>''', unsafe_allow_html=True)

with st.sidebar:
    st.title('🌿 PLANTMED')
    st.caption('AI-powered medicinal plant identification')
    page = st.radio('Navigate', ['🏠 Home','📸 Identify Plant','🌿 Plant Explorer','🤖 Plant Assistant','🗺️ Observations','👨‍🔬 Expert Verification','📊 Dashboard','🔬 Research & Evidence','⚙️ System Info'])
    language = st.selectbox('Language', ['English','ಕನ್ನಡ (Kannada)'])
    st.divider()
    st.caption('Educational tool — not medical diagnosis or prescription.')

if page == '🏠 Home':
    st.markdown('<div class="hero"><h1>🌿 PLANTMED</h1><h3>AI-Powered Medicinal Plant Identification System</h3><p>Identify plants, explore medicinal knowledge, record observations, and support expert verification.</p></div>', unsafe_allow_html=True)
    st.write('')
    a,b,c,d = st.columns(4)
    a.metric('Plant classes', len(idx_to_class)); b.metric('Database records', len(df)); c.metric('AI input', '224 × 224'); d.metric('Modes', '8+')
    st.subheader('Global-ready capabilities')
    cols = st.columns(3)
    features = [('📸','Multi-image/camera-ready identification'),('🤖','Top-5 predictions + confidence'),('⚠️','Unknown/low-confidence warnings'),('💊','Medicinal knowledge database'),('🛡️','Safety guidance layer'),('🌍','English + Kannada UI'),('🗺️','GPS observation capture'),('👨‍🔬','Expert review workflow'),('🔬','Research/evidence links'),('📊','Analytics dashboard'),('🔌','REST API-ready'),('📱','Responsive Streamlit UI')]
    for i,(icon,t) in enumerate(features): cols[i%3].markdown(f'<div class="card"><b>{icon} {t}</b></div>', unsafe_allow_html=True)

elif page == '📸 Identify Plant':
    st.title('📸 Identify a Medicinal Plant')
    st.info('For better accuracy, use a clear image with the plant part visible. Never rely on AI alone to decide whether a plant is safe to consume.')
    source = st.radio('Image source', ['Upload image','Camera'], horizontal=True)
    uploaded = st.file_uploader('Choose an image', type=['jpg','jpeg','png','webp']) if source == 'Upload image' else st.camera_input('Take a plant photo')
    if uploaded:
        img, arr = prepare_image(uploaded)
        st.image(img, caption='Input image', width=320)
        if st.button('🔍 Identify Plant', type='primary', use_container_width=True):
            with st.spinner('Running PLANTMED AI...'):
                model = load_model()
                probs = model.predict(arr, verbose=0)[0]
            top = np.argsort(probs)[::-1][:5]
            results = [{'Scientific Name': idx_to_class[int(i)], 'Confidence (%)': round(float(probs[i])*100,2)} for i in top]
            st.session_state['prediction'] = results
        if 'prediction' in st.session_state:
            results = st.session_state['prediction']; top = results[0]; name = top['Scientific Name']; conf = top['Confidence (%)']; d = plant_details(df,name)
            st.subheader(f'🌿 {name}')
            label,_ = confidence_label(conf)
            st.metric('AI confidence', f'{conf:.2f}%')
            st.caption(label)
            st.dataframe(pd.DataFrame(results), hide_index=True, use_container_width=True)
            if d:
                x,y = st.columns(2)
                with x:
                    st.markdown('### Plant profile'); st.write('**Local name:**', d.get('Kannada Name','N/A')); st.write('**Parts used:**', d.get('Parts_used','N/A')); st.write('**Grown area:**', d.get('Grown_Area','N/A'))
                with y:
                    st.markdown('### Medicinal information'); st.write('**Documented uses:**', d.get('Uses','N/A')); st.write('**Preparation (database):**', d.get('Preparation_method','N/A'))
                st.warning(safety_for(name, conf))
                st.markdown('### Research starting points')
                for u in evidence_links(name): st.write(u)
                if _db_ok and st.button('🗺️ Save observation'):
                    save_observation(name, conf, None, None); st.success('Observation saved. You can add GPS coordinates from the Observations page.')
            else: st.error('Plant was predicted but no matching record exists in the local database.')

elif page == '🌿 Plant Explorer':
    st.title('🌿 Plant Explorer')
    query = st.text_input('Search scientific or local name')
    filtered = df if not query else df[df.astype(str).apply(lambda col: col.str.contains(query, case=False, na=False)).any(axis=1)]
    st.write(f'{len(filtered)} record(s) found')
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    selected = st.selectbox('Open plant profile', ['—'] + filtered['Scientific_name'].astype(str).tolist())
    if selected != '—':
        d = plant_details(df, selected)
        st.subheader(selected); st.write('**Local name:**', d.get('Kannada Name','N/A')); st.write('**Parts:**',d.get('Parts_used','N/A')); st.write('**Uses:**',d.get('Uses','N/A')); st.write('**Preparation:**',d.get('Preparation_method','N/A')); st.write('**Area:**',d.get('Grown_Area','N/A')); st.warning('Database information is educational and should not be treated as a treatment recommendation.')

elif page == '🤖 Plant Assistant':
    st.title('🤖 PLANTMED Assistant')
    st.caption('Retrieval-based assistant using the supplied PLANTMED database; it does not diagnose or prescribe.')
    if 'chat' not in st.session_state: st.session_state.chat=[]
    for role,msg in st.session_state.chat: st.chat_message(role).write(msg)
    q=st.chat_input('Ask about a plant, uses, parts, area, or safety...')
    if q:
        st.session_state.chat.append(('user',q)); ans=assistant_answer(q,df); st.session_state.chat.append(('assistant',ans)); st.rerun()

elif page == '🗺️ Observations':
    st.title('🗺️ Global Observation Capture')
    st.write('Record where an AI identification was observed. GPS can be entered manually or supplied by a future mobile/browser geolocation integration.')
    names=df['Scientific_name'].astype(str).tolist(); plant=st.selectbox('Plant',names); lat=st.number_input('Latitude', value=0.0, format='%.6f'); lon=st.number_input('Longitude', value=0.0, format='%.6f'); notes=st.text_area('Observation notes')
    if st.button('Save GPS observation', type='primary'):
        if _db_ok: save_observation(plant,0,lat,lon,notes); st.success('Observation saved.')
        else: st.error('Database unavailable.')
    if _db_ok:
        with sqlite3.connect(DB_FILE) as con: obs=pd.read_sql_query('SELECT * FROM observations ORDER BY id DESC',con)
        if not obs.empty:
            st.map(obs.dropna(subset=['latitude','longitude']).rename(columns={'latitude':'lat','longitude':'lon'})[['lat','lon']])
            st.dataframe(obs,hide_index=True,use_container_width=True)

elif page == '👨‍🔬 Expert Verification':
    st.title('👨‍🔬 Expert Verification Workflow')
    st.write('Structured workflow for botanists/researchers to review AI observations. This local demo does not authenticate professional credentials.')
    plant=st.selectbox('Plant to review',df['Scientific_name'].astype(str).tolist()); reviewer=st.text_input('Reviewer name / organization'); status=st.selectbox('Review result',['Pending','Verified','Rejected','Needs more evidence']); comments=st.text_area('Comments')
    if st.button('Submit expert review',type='primary'):
        if reviewer.strip() and _db_ok: save_review(plant,reviewer,status,comments); st.success('Review submitted.')
        else: st.error('Enter reviewer details and ensure the database is available.')
    if _db_ok:
        with sqlite3.connect(DB_FILE) as con: reviews=pd.read_sql_query('SELECT * FROM expert_reviews ORDER BY id DESC',con)
        st.dataframe(reviews,hide_index=True,use_container_width=True)

elif page == '📊 Dashboard':
    st.title('📊 PLANTMED Dashboard')
    if _db_ok:
        with sqlite3.connect(DB_FILE) as con:
            obs=pd.read_sql_query('SELECT * FROM observations',con); rev=pd.read_sql_query('SELECT * FROM expert_reviews',con)
        a,b,c=st.columns(3); a.metric('Observations',len(obs)); b.metric('Expert reviews',len(rev)); c.metric('Verified reviews',int((rev.status=='Verified').sum()) if not rev.empty else 0)
        if not obs.empty:
            st.subheader('Most observed plants'); st.bar_chart(obs['plant'].value_counts().head(15))
            st.subheader('Confidence distribution'); st.line_chart(obs['confidence'])
        else: st.info('No observations yet.')
    else: st.error('Database unavailable.')

elif page == '🔬 Research & Evidence':
    st.title('🔬 Research & Evidence')
    plant=st.selectbox('Select plant',df['Scientific_name'].astype(str).tolist()); d=plant_details(df,plant)
    st.write('**Plant:**',plant); st.write('**Database uses:**',d.get('Uses','N/A')); st.write('**Parts:**',d.get('Parts_used','N/A'))
    st.markdown('### Research search portals')
    for u in evidence_links(plant): st.write(u)
    st.info('For a production release, connect a curated literature database/API and store DOI, publication year, study type, evidence level, and provenance for each claim.')

elif page == '⚙️ System Info':
    st.title('⚙️ System Information')
    st.write('**Model:** InceptionV3 feature extractor + dense classifier'); st.write('**Input:** 224 × 224 RGB'); st.write('**Classes:**',len(idx_to_class)); st.write('**Metadata source:** supplied Excel database'); st.write('**Storage:** SQLite for observations and expert reviews'); st.write('**API:** FastAPI service available in `api.py`'); st.write('**Deployment:** Streamlit Cloud / Render / Docker-ready files included.')
    st.code('streamlit run streamlit_app.py')
