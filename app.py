"""Interface Streamlit para envio de imagem e leitura das previsões."""
from __future__ import annotations
import hashlib
import logging
import streamlit as st
import settings
from services.classifier import FruitClassifier,trained_model_available
from utils import ui
from utils.image_utils import ImageValidationError,load_image

st.set_page_config(page_title='Identificador de frutas | IA',page_icon='🍎',layout='wide')

@st.cache_resource(show_spinner='Carregando o modelo…')
def get_classifier(model_stamp:int,classes_stamp:int)->FruitClassifier:
    return FruitClassifier()


def reset():
    st.session_state.uploader_key+=1
    st.session_state.prediction=None
    st.session_state.signature=None
    st.session_state.notice=None

for key,value in [('uploader_key',0),('prediction',None),('signature',None),('notice',None)]:
    st.session_state.setdefault(key,value)

ui.inject_css()
ui.header()
available=trained_model_available()
if not available:
    st.warning('O modelo está indisponível no momento. Conclua o treinamento para habilitar a análise.')

left,right=st.columns([1.1,1],gap='large')
image=None
with left:
    with st.container(border=True,key="image_panel"):
        ui.panel_header(1,'Sua imagem','Selecione uma foto para começar.')
        uploaded=st.file_uploader('Enviar imagem',type=settings.ALLOWED_EXTENSIONS,
                                  key=f'uploader_{st.session_state.uploader_key}',
                                  label_visibility='collapsed',
                                  help=f'JPG, JPEG, PNG ou WEBP · até {settings.MAX_FILE_MB} MB')
        st.caption(f'JPG, JPEG, PNG ou WEBP · até {settings.MAX_FILE_MB} MB')
        signature=hashlib.sha256(uploaded.getvalue()).hexdigest() if uploaded is not None else None
        if signature!=st.session_state.signature:
            st.session_state.prediction=None
            st.session_state.notice=None
            st.session_state.signature=signature
        if uploaded is not None:
            try:
                image=load_image(uploaded)
            except ImageValidationError as exc:
                st.session_state.prediction=None
                st.error(str(exc))
        if image is not None:
            st.image(image,use_container_width=True)
            st.caption(f'{uploaded.name} · {image.width} × {image.height} px')
        elif uploaded is None:
            st.markdown('<div class="photo-placeholder">A prévia da sua foto aparecerá aqui</div>',unsafe_allow_html=True)
        action,remove=st.columns([2,1])
        analyze=action.button('Analisar imagem',type='primary',use_container_width=True,
                              disabled=image is None or not available)
        remove.button('Limpar',use_container_width=True,disabled=uploaded is None,on_click=reset)
        if analyze and image is not None:
            try:
                with st.spinner('Analisando sua imagem…'):
                    classifier=get_classifier(settings.MODEL_PATH.stat().st_mtime_ns,settings.CLASSES_PATH.stat().st_mtime_ns)
                    st.session_state.prediction=classifier.predict(image)
                    st.session_state.notice=classifier.notice
            except Exception:
                logging.exception('Falha durante identificação')
                st.session_state.prediction=None
                st.error('Não foi possível analisar esta imagem. Tente novamente ou envie outra foto.')

with right:
    with st.container(border=True,key="result_panel"):
        ui.panel_header(2,'Resultado','Confira a sugestão e avalie se ela faz sentido.')
        pred=st.session_state.prediction
        if pred is None:
            ui.empty_state(ready=image is not None)
        else:
            if st.session_state.notice:
                st.warning(st.session_state.notice)
            ui.result_card(pred)
            ui.details(pred)
        ui.tips()
ui.footer()
