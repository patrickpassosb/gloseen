# -*- coding: utf-8 -*-
from flask import Flask, render_template, jsonify, request
import cv2
import numpy as np
import requests
import time
import os
import glob
import sqlite3
import json
import base64
from datetime import datetime
from dotenv import load_dotenv

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# --- IMPORTAÇÕES IAs ---
from ultralytics import YOLO
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.vision.face import FaceClient
from openai import AzureOpenAI

app = Flask(__name__, static_folder='static')

@app.template_filter('from_json')
def from_json_filter(s):
    try: return json.loads(s)
    except: return []

# ==========================================
# CONFIGURAÇÕES E CONSTANTES
# ==========================================
# Azure Credentials from environment
AZURE_VISION_ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT")
AZURE_VISION_KEY = os.getenv("AZURE_VISION_KEY")
AZURE_FACE_ENDPOINT = os.getenv("AZURE_FACE_ENDPOINT")
AZURE_FACE_KEY = os.getenv("AZURE_FACE_KEY")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")

URL_ESP32 = os.getenv("URL_ESP32", "http://192.168.15.9/capture")
PASTA_TEMP = "static"
if not os.path.exists(PASTA_TEMP): os.makedirs(PASTA_TEMP)

NOME_DB = "vision_memory.db"
# Nomes de arquivos separados para PT e EN
NOME_ARQUIVO_AUDIO_PT = "audio_pt.wav"
NOME_ARQUIVO_AUDIO_EN = "audio_en.wav"
NOME_ARQUIVO_AUDIO_RESPOSTA = "audio_resposta.wav"
NOME_ARQUIVO_CUE = "pode_falar.wav"
NOME_ARQUIVO_ALARME = "alarme.wav"

CAMINHO_AUDIO_PT = os.path.join(PASTA_TEMP, NOME_ARQUIVO_AUDIO_PT)
CAMINHO_AUDIO_EN = os.path.join(PASTA_TEMP, NOME_ARQUIVO_AUDIO_EN)
CAMINHO_AUDIO_RESPOSTA_COMPLETO = os.path.join(PASTA_TEMP, NOME_ARQUIVO_AUDIO_RESPOSTA)
CAMINHO_AUDIO_CUE_COMPLETO = os.path.join(PASTA_TEMP, NOME_ARQUIVO_CUE)
CAMINHO_AUDIO_ALARME_COMPLETO = os.path.join(PASTA_TEMP, NOME_ARQUIVO_ALARME)

# --- VOZES NEURAIS ---
VOZ_PT = 'pt-BR-FranciscaNeural'
VOZ_EN = 'en-US-JennyNeural'

# --- LISTA DE PERIGO ---
PERIGO_IMEDIATO_PT = [
    'faca', 'tesoura',  'arma', 'fogo', 'incêndio', 'chama', 'explosivo',
    'fumaça', 'vapor', 'escada', 'degrau', 'quina', 'buraco', 'obstáculo', 'vão', 'desnível'
]


CONTEXTO_ULTIMA_ANALISE = None

print("--- INICIALIZANDO SISTEMA FINAL (PITCH): ÁUDIOS SEPARADOS E ROBUSTOS ---")
try:
    computervision_client = ComputerVisionClient(AZURE_VISION_ENDPOINT, CognitiveServicesCredentials(AZURE_VISION_KEY))
    face_client = FaceClient(AZURE_FACE_ENDPOINT, CognitiveServicesCredentials(AZURE_FACE_KEY))
    openai_client = AzureOpenAI(api_key=AZURE_OPENAI_KEY, api_version="2024-02-01", azure_endpoint=AZURE_OPENAI_ENDPOINT)
    model = YOLO('yolov8x.pt')
    print("Todas as IAs Azure Carregadas.")
except Exception as e: print(f"Erro IAs Azure: {e}")

# --- FUNÇÕES DE BANCO DE DADOS ---
def get_db_connection(): conn = sqlite3.connect(NOME_DB); conn.row_factory = sqlite3.Row; return conn
def init_db():
    with get_db_connection() as conn: conn.execute('''CREATE TABLE IF NOT EXISTS historico (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, data_hora_formated TEXT, imagem_path TEXT, resumo_gpt TEXT, objetos_json TEXT, tem_perigo INTEGER)'''); conn.commit()
init_db()

# --- GERADORES DE ÁUDIOS ESTÁTICOS ---
def gerar_audios_estaticos():
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
    speech_config.speech_synthesis_voice_name='pt-BR-AntonioNeural'
    if not os.path.exists(CAMINHO_AUDIO_CUE_COMPLETO):
        try:
            audio_config = speechsdk.audio.AudioOutputConfig(filename=CAMINHO_AUDIO_CUE_COMPLETO)
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
            synthesizer.speak_text_async("Pode falar.").get()
        except: pass
    if not os.path.exists(CAMINHO_AUDIO_ALARME_COMPLETO):
        try:
            audio_config = speechsdk.audio.AudioOutputConfig(filename=CAMINHO_AUDIO_ALARME_COMPLETO)
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
            synthesizer.speak_text_async("Perigo crítico! Atenção!").get() 
        except: pass
gerar_audios_estaticos()

# --- FUNÇÕES AUXILIARES ---
def limpar_imagens_antigas():
    try:
        for arq in glob.glob(os.path.join(PASTA_TEMP, "captura_*.jpg")): os.remove(arq)
        for arq in glob.glob(os.path.join(PASTA_TEMP, "processada_*.jpg")): os.remove(arq)
    except: pass

# --- NOVA FUNÇÃO DE GERAÇÃO DE ÁUDIO SIMPLES E SEGURA ---
def gerar_audio_neural_simples(texto, voz, arquivo_saida, nome_web):
    """Gera um arquivo de áudio simples com uma única voz. Muito mais robusto."""
    if not texto: return None
    try:
        # Remove o arquivo anterior se existir para garantir que não toque áudio velho
        if os.path.exists(arquivo_saida):
            try: os.remove(arquivo_saida)
            except: pass # Ignora erro se o arquivo estiver sendo usado, o Azure sobrescreve

        speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
        speech_config.speech_synthesis_voice_name = voz
        audio_config = speechsdk.audio.AudioOutputConfig(filename=arquivo_saida)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        
        result = synthesizer.speak_text_async(texto).get()
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            # Adiciona timestamp para evitar cache do navegador
            return f"/static/{nome_web}?t={int(time.time())}_{np.random.randint(0,1000)}"
        else:
            print(f"Erro Azure TTS ({voz}): {result.cancellation_details.reason}")
            return None
    except Exception as e:
        print(f"Exceção TTS: {e}")
        return None

# --- GPT-4o VISION (PERIGOS AMBIENTAIS) ---
def analisar_imagem_com_gpt4o_vision(caminho_imagem):
    try:
        with open(caminho_imagem, "rb") as image_file: encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        system_prompt = """Você é um sistema de visão de segurança CRÍTICA.
        SUA MISSÃO PRIORITÁRIA: Identificar perigos físicos no ambiente e objetos perigosos.
        INSTRUÇÕES DE DETECÇÃO:
        1. PERIGOS AMBIENTAIS (Prioridade Máxima): Procure obsessivamente por: 'escadas', 'degraus', 'quinas' vivas, 'buracos', 'desníveis', 'fumaça', 'vapor', 'obstáculos'.
        2. OBJETOS PERIGOSOS: Identifique 'faca', 'arma', 'fogo'.
        3. OBJETOS PESSOAIS: Identifique celulares, chaves, óculos.
        Retorne APENAS uma lista de objetos/perigos identificados, separados por vírgula, em Português."""
        response = openai_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": [{"type": "text", "text": "Analise a imagem em busca de perigos ambientais e objetos."},
                                                   {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}]}],
            temperature=0.0, max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e: print(f"❌ Erro GPT Vision: {e}"); return "Erro na análise visual."

# --- RESUMO BILÍNGUE (Gera os dois textos) ---
def gerar_resumo_inteligente_bilingue(dados_da_cena, status_alerta):
    instrucao_pt, instrucao_en = "", ""
    if status_alerta == "critico":
        instrucao_pt = "ALERTA CRÍTICO: Um alarme já tocou. Comece com um aviso direto sobre o perigo detectado."
        instrucao_en = "CRITICAL ALERT: An alarm has sounded. Start with a direct warning about the danger detected."

    system_prompt = f"""Você é um assistente visual bilíngue (Português e Inglês).
    Use 'DETECCAO_VISUAL_GPT_VISION' como base.
    TAREFA: Gere o resumo DUAS VEZES. Primeiro em Português, depois em Inglês.
    Separe os dois textos EXATAMENTE com a string "###SPLIT###".
    Diretrizes PT: {instrucao_pt} Seja direto.
    Diretrizes EN: {instrucao_en} Be direct.
    """
    user_prompt = f"""Dados da Cena: {dados_da_cena}. Gere o resumo PT ###SPLIT### EN."""
    try:
        response = openai_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.3, max_tokens=600
        )
        texto_completo = response.choices[0].message.content
        parts = texto_completo.split("###SPLIT###")
        if len(parts) >= 2: return parts[0].strip(), parts[1].strip()
        else: return texto_completo, "Error generating translation."
    except: return "Erro no resumo.", "Error in summary."

# --- ROTAS WEB ---
@app.route('/')
def index(): return render_template('index.html')

@app.route('/historico')
def historico():
    conn = get_db_connection(); capturas = conn.execute('SELECT * FROM historico ORDER BY id DESC LIMIT 20').fetchall(); conn.close()
    return render_template('memory.html', capturas=capturas)

@app.route('/analisar', methods=['POST'])
def analisar():
    global CONTEXTO_ULTIMA_ANALISE
    print(">>> INICIANDO ANÁLISE (PERIGOS + ÁUDIOS SEPARADOS) <<<")
    limpar_imagens_antigas(); timestamp_agora = int(time.time()); data_hora_formated = datetime.now().strftime("%d/%m/%Y %H:%M")
    nome_arq = f"captura_{timestamp_agora}.jpg"; caminho_img = os.path.join(PASTA_TEMP, nome_arq)

    # Captura ESP32
    try:
        response = requests.get(URL_ESP32, timeout=10)
        if response.status_code == 200 and len(response.content) > 0:
            with open(caminho_img, "wb") as f: f.write(response.content); f.flush(); os.fsync(f.fileno()); time.sleep(0.2)
        else: return jsonify({'status': 'erro', 'mensagem': 'Erro captura ESP32.'})
    except Exception as e: return jsonify({'status': 'erro', 'mensagem': f'Falha ESP32: {e}'})

    # Execução IAs
    print("ENVIANDO PARA GPT-4o VISION...")
    deteccao_visual_gpt = analisar_imagem_com_gpt4o_vision(caminho_img)

    # Alerta Crítico
    status_alerta = "nenhum"
    if any(perigo in deteccao_visual_gpt.lower() for perigo in PERIGO_IMEDIATO_PT): status_alerta = "critico"

    dados_para_o_cerebro = {'DETECCAO_VISUAL_GPT_VISION': deteccao_visual_gpt, 'STATUS_ALERTA_ATUAL': status_alerta}
    CONTEXTO_ULTIMA_ANALISE = dados_para_o_cerebro

    # --- GERAÇÃO DOS DOIS ÁUDIOS SEPARADOS ---
    texto_pt, texto_en = gerar_resumo_inteligente_bilingue(dados_para_o_cerebro, status_alerta)
    
    # Gera áudio PT
    url_audio_pt = gerar_audio_neural_simples(texto_pt, VOZ_PT, CAMINHO_AUDIO_PT, NOME_ARQUIVO_AUDIO_PT)
    # Gera áudio EN (com um pequeno delay para não sobrecarregar a API)
    time.sleep(0.1)
    url_audio_en = gerar_audio_neural_simples(texto_en, VOZ_EN, CAMINHO_AUDIO_EN, NOME_ARQUIVO_AUDIO_EN)
    
    # Salva no histórico (apenas PT)
    try:
        lista_objs = [item.strip() for item in deteccao_visual_gpt.split(',') if item.strip()]
        with get_db_connection() as conn: conn.execute('''INSERT INTO historico VALUES (NULL, ?, ?, ?, ?, ?, ?)''', (str(timestamp_agora), data_hora_formated, f"/static/{nome_arq}", texto_pt, json.dumps(lista_objs, ensure_ascii=False), 1 if status_alerta != "nenhum" else 0)); conn.commit()
    except: pass

    # RETORNA AS DUAS URLS SEPARADAS
    return jsonify({
        'status': 'sucesso',
        'yolo': deteccao_visual_gpt, 
        'azure_desc': "Análise GPT-4o Vision (Ambiente+Objetos)", 
        'status_alerta': status_alerta,
        'imagem_url': f"/static/{nome_arq}",
        'audio_url_pt': url_audio_pt, # URL 1
        'audio_url_en': url_audio_en  # URL 2
    })

@app.route('/perguntar', methods=['POST'])
def perguntar():
    global CONTEXTO_ULTIMA_ANALISE
    if CONTEXTO_ULTIMA_ANALISE is None: return jsonify({'status': 'erro', 'resposta': 'Analise primeiro.'})
    dados = request.get_json(); pergunta = dados.get('pergunta')
    if not pergunta: return jsonify({'status': 'erro', 'resposta': 'Vazio.'})
    
    # Detecta idioma da pergunta
    is_english = any(word in pergunta.lower() for word in ['what', 'where', 'how', 'is there', 'do you see', 'take', 'photo'])
    lang_instruction = "Responda em Inglês." if is_english else "Responda em Português."
    
    # Gera resposta
    try:
        response = openai_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME, messages=[{"role": "system", "content": f"Use 'DETECCAO_VISUAL_GPT_VISION'. {lang_instruction} Seja breve."}, {"role": "user", "content": f"Contexto: {CONTEXTO_ULTIMA_ANALISE}. Pergunta: {pergunta}"}],
            temperature=0.5, max_tokens=200
        )
        resposta_texto = response.choices[0].message.content
    except: resposta_texto = "Erro."; is_english = False;

    # Gera áudio da resposta com a voz certa
    voice_to_use = VOZ_EN if is_english else VOZ_PT
    url_audio_resp = gerar_audio_neural_simples(resposta_texto, voice_to_use, CAMINHO_AUDIO_RESPOSTA_COMPLETO, NOME_ARQUIVO_AUDIO_RESPOSTA)

    return jsonify({'status': 'sucesso', 'resposta_texto': resposta_texto, 'audio_url': url_audio_resp})

@app.route('/ler_texto', methods=['POST'])
def ler_texto(): return jsonify({'status':'erro'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)