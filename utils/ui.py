"""Componentes visuais e estados da interface de classificação."""
from html import escape
import streamlit as st
import settings
from services.classifier import Prediction

CSS = '''
<style>
:root { --ink:#182D29; --muted:#586B65; --line:#DEE6E1; --green:#185C46; }
html, body, [class*="st-"] { font-family: 'Segoe UI', system-ui, sans-serif; }
.stApp { background:#F5F7F5; }
.block-container { max-width:1160px; padding:2.5rem 2rem 2rem; }
h1,h2,h3 { color:var(--ink); letter-spacing:-.035em; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility:hidden; }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stVerticalBlock"] { gap:1rem; }
.hero { padding:0 0 1.6rem; }
.brand { display:flex; align-items:center; gap:.6rem; color:var(--green); font-weight:650; font-size:.85rem; margin-bottom:1.25rem; }
.brand-mark { display:grid; place-items:center; width:28px; height:28px; background:var(--green); color:white; border-radius:8px; font-size:18px; }
.hero h1 { font-size:clamp(2rem,4vw,2.8rem); font-weight:700; line-height:1.15; margin:0 0 .65rem; padding:0; }
.hero p { color:var(--muted); font-size:1.05rem; line-height:1.6; margin:0; }
.fruit-list { display:flex; flex-wrap:wrap; gap:.45rem; margin-top:1rem; }
.fruit-tag { color:#40564D; font-size:.78rem; padding:.25rem .65rem; background:#EAF0EB; border:1px solid #DFE7E0; border-radius:30px; }
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > .st-key-image_panel), [data-testid="stVerticalBlockBorderWrapper"]:has(> div > .st-key-result_panel) { background:white; border-color:var(--line)!important; border-radius:16px!important; }
.panel-head { display:flex; align-items:center; gap:.7rem; margin-bottom:.2rem; }
.step { display:grid; place-items:center; width:28px; height:28px; background:#EDF3EF; color:var(--green); border-radius:50%; font-size:.85rem; font-weight:700; flex-shrink:0; }
.panel-head h2 { font-size:1.1rem; font-weight:650; margin:0; padding:0; letter-spacing:-.02em; }
.panel-copy { font-size:.88rem; color:var(--muted); line-height:1.5; margin:.3rem 0 .7rem; }
[data-testid="stFileUploaderDropzone"] { border:1px dashed #A8BCB0; border-radius:12px; background:#F7FAF8; padding:1rem; flex-wrap:wrap; gap:.7rem; }
[data-testid="stFileUploaderDropzone"] button { white-space:nowrap; min-width:110px; }
[data-testid="stFileUploaderDropzoneInstructions"] { min-width:140px; }
[data-testid="stImage"] { width:100%; }
[data-testid="stImage"] img { width:100%; max-height:310px; object-fit:contain; border-radius:10px; background:#F6F8F6; }
[data-testid="stImageCaption"] { font-size:.75rem; color:var(--muted); }
.stButton > button { border-radius:10px; min-height:44px; font-weight:600; border-color:#C9D7CF; }
.stButton > button[kind="primary"] { background:var(--green); border-color:var(--green); }
.stButton > button[kind="primary"]:disabled { background:#E4EBE6; border-color:#E4EBE6; color:#6A7B70; opacity:1; }
.stButton > button:focus-visible, summary:focus-visible { outline:3px solid #75AD93!important; outline-offset:3px; }
.state-empty { min-height:240px; display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center; padding:2rem 1rem; }
.state-icon { display:grid; place-items:center; width:54px; height:54px; border-radius:16px; background:#EDF3EF; color:#487761; font-size:1.6rem; margin-bottom:1rem; }
.state-empty h3 { margin:0 0 .5rem; padding:0; font-size:1.15rem; letter-spacing:-.02em; }
.state-empty p { margin:0; color:var(--muted); line-height:1.65; font-size:.9rem; max-width:280px; }
.photo-placeholder { min-height:130px; border-radius:10px; background:#F7F9F7; display:grid; place-items:center; color:#62766B; font-size:.85rem; border:1px solid #ECF0EC; }
.result-card { border-radius:12px; padding:1.35rem; background:#F0F7F2; border:1px solid #DAE8DE; }
.status { display:inline-flex; align-items:center; gap:.4rem; font-size:.75rem; font-weight:650; color:#215740; background:#DDEDE2; border-radius:30px; padding:.3rem .65rem; }
.result-name { display:flex; gap:.7rem; align-items:center; margin:1rem 0 1.2rem; }
.result-name h3 { font-size:2rem; line-height:1.2; margin:0; padding:0; }
.result-emoji { font-size:2rem; }
.score-line { display:flex; align-items:baseline; justify-content:space-between; gap:1rem; }
.score-line span { color:var(--muted); font-size:.82rem; }
.score-line strong { color:var(--ink); font-size:1.25rem; }
.score-track { margin:.5rem 0 .75rem; height:7px; border-radius:8px; background:#D9E6DC; overflow:hidden; }
.score-track div { height:100%; background:var(--green); border-radius:8px; }
.result-note { color:#4B6258; font-size:.8rem; line-height:1.55; margin:0; }
.result-card.uncertain { background:#FFFAEF; border-color:#EDDFBC; }
.uncertain .status { background:#F5E9C9; color:#795514; }
.uncertain h3 { font-size:1.45rem; margin:1rem 0 .6rem; padding:0; letter-spacing:-.02em; }
.uncertain p { color:#685C42; font-size:.9rem; line-height:1.6; }
.prob-row { display:grid; grid-template-columns:80px 1fr 55px; gap:.8rem; align-items:center; padding:.45rem 0; font-size:.85rem; color:#3D5349; }
.prob-row strong { text-align:right; font-weight:500; font-variant-numeric:tabular-nums; }
.prob-track { height:6px; background:#ECF1ED; border-radius:6px; overflow:hidden; }
.prob-track span { display:block; height:100%; background:#5D9379; border-radius:6px; }
.tips { border-top:1px solid var(--line); margin-top:1.2rem; padding-top:1rem; }
.tips h3 { font-size:.85rem; letter-spacing:0; margin:0 0 .45rem; padding:0; }
.tips p { font-size:.82rem; line-height:1.65; color:var(--muted); margin:0; }
.footer-note { padding:1.1rem 0 0; color:var(--muted); font-size:.8rem; line-height:1.7; }
.footer-meta { display:flex; flex-wrap:wrap; justify-content:space-between; gap:.4rem 1rem; margin-top:1.1rem; padding-top:1rem; border-top:1px solid var(--line); font-size:.75rem; color:#62736B; }
@media(max-width:700px) { .block-container { padding:1.3rem 1rem 2rem; } .hero { padding-bottom:1.2rem; } .brand { margin-bottom:1rem; } .state-empty { min-height:170px; padding:1rem; } .result-name h3 { font-size:1.8rem; } }
</style>
'''


def pct(value):
    return f'{value*100:.1f}'.replace('.',',')+'%'


def inject_css():
    st.markdown(CSS,unsafe_allow_html=True)


def header():
    tags=''.join(f'<span class="fruit-tag">{escape(f["name"])}</span>' for f in settings.FRUITS.values())
    st.markdown('<div class="hero"><div class="brand"><span class="brand-mark" aria-hidden="true">✳</span>FRUTAS · LABORATÓRIO DE IA</div>'
                '<h1>Qual é a fruta?</h1><p>Envie uma imagem e confira a sugestão do modelo.</p>'
                f'<div class="fruit-list" aria-label="Categorias disponíveis">{tags}</div></div>',unsafe_allow_html=True)


def panel_header(number,title,description):
    st.markdown(f'<div class="panel-head"><span class="step">{number}</span><h2>{escape(title)}</h2></div>'
                f'<p class="panel-copy">{escape(description)}</p>',unsafe_allow_html=True)


def empty_state(ready=False):
    title='Tudo pronto para analisar' if ready else 'Aguardando uma imagem'
    message='Clique em “Analisar imagem” para ver a sugestão e as pontuações.' if ready else 'Escolha uma foto ao lado. O resultado da análise aparecerá aqui.'
    st.markdown(f'<div class="state-empty"><div class="state-icon" aria-hidden="true">⌕</div><h3>{title}</h3><p>{message}</p></div>',unsafe_allow_html=True)


def result_card(pred:Prediction):
    if pred.confident and pred.best:
        fruit=settings.FRUITS[pred.best.key]
        st.markdown('<div class="result-card" role="status"><span class="status">Sugestão do modelo</span>'
                    f'<div class="result-name"><span class="result-emoji" aria-hidden="true">{fruit["emoji"]}</span><h3>{escape(pred.best.label)}</h3></div>'
                    f'<div class="score-line"><span>Pontuação do modelo</span><strong>{pct(pred.best.score)}</strong></div>'
                    f'<div class="score-track" aria-hidden="true"><div style="width:{pred.best.score*100:.2f}%"></div></div>'
                    '<p class="result-note">Essa pontuação não garante que a identificação esteja correta.</p></div>',unsafe_allow_html=True)
    else:
        guess=f'A categoria mais próxima foi <strong>{escape(pred.best.label)}</strong>, com {pct(pred.best.score)}.' if pred.best else 'Nenhuma categoria pôde ser sugerida.'
        st.markdown('<div class="result-card uncertain" role="status"><span class="status">Baixa confiança</span>'
                    f'<h3>Resultado inconclusivo</h3><p>{guess}</p>'
                    '<p class="result-note">Tente outra foto: fruta inteira, bem iluminada e com fundo simples.</p></div>',unsafe_allow_html=True)


def details(pred:Prediction):
    with st.expander('Ver pontuações das 6 frutas',expanded=False):
        rows=''
        for c in pred.candidates:
            rows+=f'<div class="prob-row"><span>{escape(c.label)}</span><div class="prob-track" aria-hidden="true"><span style="width:{c.score*100:.2f}%"></span></div><strong>{pct(c.score)}</strong></div>'
        st.markdown(rows,unsafe_allow_html=True)
        st.caption(f'Abaixo de {pct(pred.threshold)}, o resultado é apresentado como inconclusivo.')


def tips():
    st.markdown('<div class="tips"><h3>Para uma foto melhor</h3><p>Uma fruta por vez · Boa iluminação · Fundo simples<br>Prefira a fruta inteira, sem cortes ou objetos sobrepostos.</p></div>',unsafe_allow_html=True)


def footer():
    st.markdown('<div class="footer-note"><strong>Em fase de testes.</strong> O modelo pode errar, inclusive com pontuação alta. Fotos reais ainda apresentam limitações de reconhecimento.</div>'
                '<div class="footer-meta"><span>FAMETRO · Engenharia de Software</span><span>Projeto acadêmico · Inteligência Artificial</span></div>',unsafe_allow_html=True)
