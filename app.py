import streamlit as st
import requests
import pandas as pd
import numpy as np
import unicodedata
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from scipy.stats import poisson, norm

# Configuración inicial de la página web
st.set_page_config(page_title="NBA Analytics & Backtesting Engine", page_icon="🏀", layout="wide")

# Estilos CSS avanzados
st.markdown("""
    <style>
    .stMetric {
        background-color: #1e2530;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2d3748;
    }
    .card-over {
        background-color: rgba(16, 185, 129, 0.15) !important;
        border: 1px solid #10b981 !important;
        padding: 15px;
        border-radius: 10px;
    }
    .card-under {
        background-color: rgba(239, 68, 68, 0.15) !important;
        border: 1px solid #ef4444 !important;
        padding: 15px;
        border-radius: 10px;
    }
    .card-push {
        background-color: rgba(245, 158, 11, 0.15) !important;
        border: 1px solid #f59e0b !important;
        padding: 15px;
        border-radius: 10px;
    }
    .card-live {
        background-color: rgba(59, 130, 246, 0.12) !important;
        border: 1px solid #3b82f6 !important;
        padding: 14px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .card-high-conf {
        box-shadow: 0 0 14px rgba(250, 204, 21, 0.35) !important;
        border-width: 2px !important;
    }
    .badge-high-conf {
        display: inline-block;
        background: linear-gradient(90deg, #f59e0b, #eab308);
        color: #0f172a;
        font-weight: 800;
        font-size: 11px;
        padding: 3px 9px;
        border-radius: 999px;
        letter-spacing: 0.4px;
        margin-bottom: 6px;
        text-transform: uppercase;
    }
    .badge-normal-conf {
        display: inline-block;
        background-color: rgba(148, 163, 184, 0.2);
        color: #cbd5e1;
        font-weight: 600;
        font-size: 11px;
        padding: 3px 9px;
        border-radius: 999px;
        margin-bottom: 6px;
    }
    .dual-prob-container {
        width: 100%;
        height: 28px;
        border-radius: 8px;
        overflow: hidden;
        display: flex;
        font-weight: 700;
        font-size: 13px;
        border: 1px solid #334155;
        margin-top: 6px;
        margin-bottom: 10px;
    }
    .dual-prob-t1 {
        background: linear-gradient(90deg, #2563eb, #3b82f6);
        color: #ffffff;
        display: flex;
        align-items: center;
        justify-content: flex-start;
        padding-left: 12px;
        white-space: nowrap;
        transition: width 0.4s ease;
    }
    .dual-prob-t2 {
        background: linear-gradient(90deg, #ea580c, #f97316);
        color: #ffffff;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 12px;
        white-space: nowrap;
        transition: width 0.4s ease;
    }
    .status-active {
        background-color: rgba(16, 185, 129, 0.2);
        border-left: 5px solid #10b981;
        padding: 10px;
        border-radius: 5px;
        color: #d1fae5;
    }
    .status-questionable {
        background-color: rgba(245, 158, 11, 0.2);
        border-left: 5px solid #f59e0b;
        padding: 10px;
        border-radius: 5px;
        color: #fef3c7;
    }
    .status-injured {
        background-color: rgba(239, 68, 68, 0.2);
        border-left: 5px solid #ef4444;
        padding: 10px;
        border-radius: 5px;
        color: #fee2e2;
    }
    </style>
""", unsafe_allow_html=True)

# Constantes de Calibración y Mercado
UMBRAL_ALTA_CONFIANZA = 58.0
BREAK_EVEN_ODDS_110 = 52.38  # Probabilidad implícita estándar de momio -110 (1.91)
UMBRAL_EDGE_OPERABLE = 5.0   # Edge mínimo neto para considerar un mercado operable

ESTATUS_NO_DISPONIBLE = [
    'out', 'injured', 'injury', 'doubtful', 'suspension', 'suspended',
    'inactive', 'g-league', 'g league', 'not with team', 'personal', 'rest'
]


def normalizar_texto(texto):
    if not texto:
        return ""
    nfkd = unicodedata.normalize('NFKD', str(texto))
    return "".join(c for c in nfkd if not unicodedata.combining(c)).strip().lower()


def es_estatus_baja(status_str):
    if not status_str:
        return False
    s = str(status_str).strip().lower()
    return any(palabra in s for palabra in ESTATUS_NO_DISPONIBLE)


# ==============================================================================
# 0. MOTOR DE BÚSQUEDA WEB DE NOTICIAS Y ESTADO DE JUGADORES (GOOGLE NEWS + ESPN)
# ==============================================================================

@st.cache_data(ttl=300)
def buscar_noticias_web_nba(consulta, idioma="en", max_resultados=12):
    if not consulta or not str(consulta).strip():
        return []

    query_codificada = urllib.parse.quote(f"{consulta.strip()} NBA")
    if idioma == "es":
        url_rss = f"https://news.google.com/rss/search?q={query_codificada}&hl=es-419&gl=US&ceid=US:es-419"
    else:
        url_rss = f"https://news.google.com/rss/search?q={query_codificada}&hl=en-US&gl=US&ceid=US:en"

    headers = {'User-Agent': 'Mozilla/5.0'}
    resultados = []

    try:
        resp = requests.get(url_rss, headers=headers, timeout=8)
        if resp.status_code == 200:
            root = ET.fromstring(resp.content)
            for item in root.findall('.//item')[:max_resultados]:
                titulo_raw = item.findtext('title', default='')
                link = item.findtext('link', default='')
                pub_date = item.findtext('pubDate', default='')
                fuente_el = item.find('source')
                fuente = fuente_el.text if fuente_el is not None else "Google News"

                titulo_limpio = titulo_raw.rsplit(' - ', 1)[0] if ' - ' in titulo_raw else titulo_raw

                try:
                    dt_pub = datetime.strptime(pub_date[:25], "%a, %d %b %Y %H:%M:%S")
                    dt_pub_ect = dt_pub - timedelta(hours=5)
                    fecha_fmt = dt_pub_ect.strftime("%d/%m %H:%M ECT")
                except Exception:
                    fecha_fmt = pub_date[:16]

                resultados.append({
                    'titulo': titulo_limpio,
                    'fuente': fuente,
                    'fecha': fecha_fmt,
                    'url': link
                })
    except Exception as e:
        print(f"Error buscador noticias web: {e}")

    return resultados


@st.cache_data(ttl=300)
def obtener_ultimas_noticias_espn_nba():
    url_news = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/news?limit=15"
    headers = {'User-Agent': 'Mozilla/5.0'}
    noticias = []
    try:
        res = requests.get(url_news, headers=headers, timeout=8).json()
        for art in res.get('articles', []):
            noticias.append({
                'titulo': art.get('headline', ''),
                'descripcion': art.get('description', ''),
                'fecha': art.get('published', '')[:16].replace('T', ' '),
                'url': art.get('links', {}).get('web', {}).get('href', 'https://www.espn.com/nba/')
            })
    except Exception as e:
        print(f"Error noticias ESPN: {e}")
    return noticias


@st.cache_data(ttl=600)
def obtener_tabla_completa_lesiones_nba():
    url_inj = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
    headers = {'User-Agent': 'Mozilla/5.0'}
    filas = []
    try:
        res = requests.get(url_inj, headers=headers, timeout=8).json()
        for eq_block in res.get('injuries', []):
            equipo_nom = eq_block.get('displayName', '')
            for item in eq_block.get('injuries', []):
                atl = item.get('athlete', {})
                det = item.get('details', {})
                filas.append({
                    'Equipo': equipo_nom,
                    'Jugador': atl.get('displayName', ''),
                    'Posición': atl.get('position', {}).get('abbreviation', '-'),
                    'Estatus': item.get('status', ''),
                    'Lesión / Motivo': det.get('type', '') or item.get('shortComment', ''),
                    'Comentario Oficial': item.get('longComment') or item.get('shortComment', '')
                })
    except Exception as e:
        print(f"Error tabla lesiones: {e}")
    return pd.DataFrame(filas)


# ==============================================================================
# 1. EXTRACCIÓN DE DATOS BASE (CARTELERA, REPORTES MÉDICOS, B2B Y ROSTERS)
# ==============================================================================

@st.cache_data(ttl=90)
def obtener_cartelera_por_fecha(fecha_str):
    url_scoreboard = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={fecha_str}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    partidos = []
   
    try:
        res = requests.get(url_scoreboard, headers=headers, timeout=10).json()
        for evento in res.get('events', []):
            date_str = evento.get('date', '')
            try:
                dt_utc = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                dt_ecuador = dt_utc.astimezone(timezone(timedelta(hours=-5)))
                hora_ecuador_str = dt_ecuador.strftime("%H:%M")
                dt_orden = dt_ecuador
            except Exception:
                hora_ecuador_str = "Por definir"
                dt_orden = datetime.min

            competitions = evento.get('competitions', [{}])[0]
            competidores = competitions.get('competitors', [])

            odds_list = competitions.get('odds', [])
            linea_mercado_ou = None
            spread_mercado = None
            if odds_list and isinstance(odds_list, list):
                ou_val = odds_list[0].get('overUnder')
                if ou_val is not None:
                    try:
                        linea_mercado_ou = float(ou_val)
                    except (ValueError, TypeError):
                        linea_mercado_ou = None
                sp_val = odds_list[0].get('spread')
                if sp_val is not None:
                    try:
                        spread_mercado = abs(float(sp_val))
                    except (ValueError, TypeError):
                        spread_mercado = None

            bajas_evento_ids = set()
            bajas_evento_nombres = set()
            for grp_inj in competitions.get('injuries', []):
                for inj_item in grp_inj.get('injuries', []):
                    st_inj = str(inj_item.get('status', '')).lower()
                    atl_inj = inj_item.get('athlete', {})
                    id_inj = str(atl_inj.get('id', ''))
                    nom_inj = atl_inj.get('displayName') or atl_inj.get('fullName', '')
                    if es_estatus_baja(st_inj):
                        if id_inj:
                            bajas_evento_ids.add(id_inj)
                        if nom_inj:
                            bajas_evento_nombres.add(normalizar_texto(nom_inj))

            status_obj = evento.get('status', {})
            status_type = status_obj.get('type', {})
            estado_desc = status_type.get('shortDetail') or status_type.get('description', 'Programado')
           
            if len(competidores) == 2:
                t1 = competidores[0]
                t2 = competidores[1]
               
                score_1 = t1.get('score', '0')
                score_2 = t2.get('score', '0')
               
                partidos.append({
                    'event_id': str(evento.get('id')),
                    'dt_orden': dt_orden,
                    'hora_ecuador': hora_ecuador_str,
                    'id_t1': str(t1.get('team', {}).get('id')),
                    'abrev_t1': t1.get('team', {}).get('abbreviation'),
                    'nombre_t1': t1.get('team', {}).get('displayName'),
                    'score_t1': int(score_1) if str(score_1).isdigit() else 0,
                   
                    'id_t2': str(t2.get('team', {}).get('id')),
                    'abrev_t2': t2.get('team', {}).get('abbreviation'),
                    'nombre_t2': t2.get('team', {}).get('displayName'),
                    'score_t2': int(score_2) if str(score_2).isdigit() else 0,
                   
                    'encuentro': f"{t1.get('team', {}).get('abbreviation')} vs {t2.get('team', {}).get('abbreviation')}",
                    'detalle': f"{t1.get('team', {}).get('displayName')} vs {t2.get('team', {}).get('displayName')}",
                    'estado': estado_desc,
                    'estado_codigo': status_type.get('state', 'pre'),
                    'cuarto': int(status_obj.get('period', 0)),
                    'linea_mercado_ou': linea_mercado_ou,
                    'spread_mercado': spread_mercado,
                    'bajas_evento_ids': list(bajas_evento_ids),
                    'bajas_evento_nombres': list(bajas_evento_nombres)
                })
       
        partidos = sorted(partidos, key=lambda x: x['dt_orden'])
    except Exception as e:
        print(f"Error cartelera fecha {fecha_str}: {e}")
       
    return partidos


@st.cache_data(ttl=1800)
def obtener_reporte_lesiones_liga():
    url_inj = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
    headers = {'User-Agent': 'Mozilla/5.0'}
    bajas_globales = {}
    try:
        res = requests.get(url_inj, headers=headers, timeout=8).json()
        for eq_block in res.get('injuries', []):
            for item in eq_block.get('injuries', []):
                status_txt = item.get('status', '')
                atl = item.get('athlete', {})
                id_atl = str(atl.get('id', ''))
                if id_atl and status_txt:
                    bajas_globales[id_atl] = status_txt
    except Exception as e:
        print(f"Error reporte lesiones liga: {e}")
    return bajas_globales


def detectar_equipos_back_to_back(fecha_dt):
    fecha_ayer = fecha_dt - timedelta(days=1)
    fecha_ayer_str = fecha_ayer.strftime("%Y%m%d")
    partidos_ayer = obtener_cartelera_por_fecha(fecha_ayer_str)
    equipos_b2b = set()
    for p in partidos_ayer:
        equipos_b2b.add(str(p['id_t1']))
        equipos_b2b.add(str(p['id_t2']))
    return equipos_b2b


@st.cache_data(ttl=21600)
def obtener_metricas_avanzadas_equipos():
    url_equipos = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams?limit=50"
    headers = {'User-Agent': 'Mozilla/5.0'}
    metricas = {}

    try:
        res_eq = requests.get(url_equipos, headers=headers, timeout=10).json()
        equipos = res_eq.get('sports', [{}])[0].get('leagues', [{}])[0].get('teams', [])

        for item in equipos:
            team_obj = item.get('team', {})
            id_eq = str(team_obj.get('id', ''))
            abrev = team_obj.get('abbreviation', '')
            if not id_eq:
                continue

            url_stats = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{id_eq}/statistics"
            try:
                res_st = requests.get(url_stats, headers=headers, timeout=6).json()
                categorias = res_st.get('results', {}).get('stats', {}).get('categories', [])
                
                stats_map = {}
                for cat in categorias:
                    for st_item in cat.get('stats', []):
                        nombre_st = st_item.get('name', '')
                        val_st = st_item.get('value', None)
                        if nombre_st and val_st is not None:
                            stats_map[nombre_st] = float(val_st)

                fga = stats_map.get('avgFieldGoalsAttempted', 88.5)
                fta = stats_map.get('avgFreeThrowsAttempted', 22.0)
                orb = stats_map.get('avgOffensiveRebounds', 10.5)
                tov = stats_map.get('avgTurnovers', 13.8)
                pts_favor = stats_map.get('avgPoints', 114.0)
                pts_contra = stats_map.get('avgPointsAgainst', stats_map.get('opponentPoints', 114.0))

                pace_est = fga + (0.44 * fta) - orb + tov
                if pace_est < 85.0 or pace_est > 115.0:
                    pace_est = 99.5

                off_rtg = (pts_favor / pace_est) * 100.0
                def_rtg = (pts_contra / pace_est) * 100.0

                metricas[id_eq] = {
                    'abrev': abrev,
                    'pace': round(pace_est, 2),
                    'off_rtg': round(off_rtg, 2),
                    'def_rtg': round(def_rtg, 2),
                    'pts_favor': round(pts_favor, 1),
                    'pts_contra': round(pts_contra, 1)
                }
            except Exception:
                metricas[id_eq] = {
                    'abrev': abrev,
                    'pace': 99.5,
                    'off_rtg': 114.5,
                    'def_rtg': 114.5,
                    'pts_favor': 114.0,
                    'pts_contra': 114.0
                }
    except Exception as e:
        print(f"Error métricas equipos: {e}")

    if metricas:
        pace_liga = float(np.mean([v['pace'] for v in metricas.values()]))
        off_liga = float(np.mean([v['off_rtg'] for v in metricas.values()]))
        def_liga = float(np.mean([v['def_rtg'] for v in metricas.values()]))
    else:
        pace_liga, off_liga, def_liga = 99.5, 114.5, 114.5

    return metricas, {'pace_liga': pace_liga, 'off_liga': off_liga, 'def_liga': def_liga}


@st.cache_data(ttl=3600)
def obtener_mapa_jugadores_completo():
    url_equipos = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams?limit=50"
    headers = {'User-Agent': 'Mozilla/5.0'}
    mapa_jugadores = {}
    lesiones_liga = obtener_reporte_lesiones_liga()
   
    try:
        res_eq = requests.get(url_equipos, headers=headers, timeout=10).json()
        for item in res_eq.get('sports', [{}])[0].get('leagues', [{}])[0].get('teams', []):
            id_equipo = str(item['team']['id'])
            nombre_equipo = item['team']['displayName']
            abrev_equipo = item['team']['abbreviation']
           
            url_roster = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{id_equipo}/roster"
            res_roster = requests.get(url_roster, headers=headers, timeout=5).json()
           
            for atleta in res_roster.get('athletes', []):
                nombre_completo = atleta.get('fullName', '')
                id_atleta = str(atleta.get('id', ''))
                status_roster = atleta.get('status', {}).get('name', 'Active') or 'Active'
                
                status_lesion_global = lesiones_liga.get(id_atleta, '')
                if status_lesion_global and es_estatus_baja(status_lesion_global):
                    status_final = f"Out ({status_lesion_global})"
                elif status_lesion_global and status_roster.lower() == 'active':
                    status_final = status_lesion_global
                else:
                    status_final = status_roster

                pos_abrev = atleta.get('position', {}).get('abbreviation', 'F')
               
                if nombre_completo and id_atleta:
                    mapa_jugadores[nombre_completo] = {
                        'id': id_atleta,
                        'team_id': id_equipo,
                        'team_abbrev': abrev_equipo,
                        'team_name': nombre_equipo,
                        'status': status_final,
                        'position': pos_abrev
                    }
    except Exception as e:
        print(f"Error roster: {e}")
       
    return mapa_jugadores


@st.cache_data(ttl=3600)
def consultar_gamelog_jugador(id_espn):
    url_gamelog = f"https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/{id_espn}/gamelog"
    headers = {'User-Agent': 'Mozilla/5.0'}
   
    try:
        respuesta = requests.get(url_gamelog, headers=headers, timeout=10).json()
        nombres_columnas = respuesta.get('names', [])
        eventos_dict = respuesta.get('events', {})

        filas = []
        for temporada in respuesta.get('seasonTypes', []):
            for categoria in temporada.get('categories', []):
                for evento in categoria.get('events', []):
                    id_partido = evento.get('eventId')
                    stats = dict(zip(nombres_columnas, evento.get('stats', [])))
                    info_partido = eventos_dict.get(id_partido, {})
                   
                    fecha_raw = info_partido.get('gameDate', '')[:10]
                    minutos_str = str(stats.get('minutes', 0))
                    try:
                        min_val = float(minutos_str)
                    except Exception:
                        min_val = 0.0

                    pts_v = int(stats.get('points', 0))
                    reb_v = int(stats.get('totalRebounds', 0))
                    ast_v = int(stats.get('assists', 0))

                    filas.append({
                        'FECHA': fecha_raw,
                        'RIVAL': info_partido.get('opponent', {}).get('abbreviation', ''),
                        'MIN': min_val,
                        'PTS': pts_v,
                        'REB': reb_v,
                        'AST': ast_v,
                        'PRA': pts_v + reb_v + ast_v
                    })

        df = pd.DataFrame(filas)
        if not df.empty:
            df = df.sort_values(by='FECHA', ascending=False).drop_duplicates()
            return df
    except Exception as e:
        print(f"Error gamelog: {e}")
    return None


def obtener_historial_previo_sin_leakage(df_jugador, fecha_limite_str):
    if df_jugador is None or df_jugador.empty:
        return df_jugador
    df_previo = df_jugador[df_jugador['FECHA'] < fecha_limite_str]
    return df_previo if not df_previo.empty else df_jugador


def calcular_estadisticas_ponderadas_jugador(df_jugador_previo):
    if df_jugador_previo is None or df_jugador_previo.empty:
        return {'min': 28.0, 'pts': 15.0, 'reb': 5.0, 'ast': 3.0, 'pra': 23.0, 'tasa_pts': 0.50, 'tasa_reb': 0.18, 'tasa_ast': 0.11}

    df_v = df_jugador_previo[df_jugador_previo['MIN'] > 0]
    if df_v.empty:
        return {'min': 28.0, 'pts': 15.0, 'reb': 5.0, 'ast': 3.0, 'pra': 23.0, 'tasa_pts': 0.50, 'tasa_reb': 0.18, 'tasa_ast': 0.11}

    df_5 = df_v.head(5)
    df_10 = df_v.head(10)

    def pond_col(col):
        v5 = df_5[col].mean()
        v10 = df_10[col].mean()
        vall = df_v[col].mean()
        return (0.55 * v5) + (0.30 * v10) + (0.15 * vall)

    min_p = pond_col('MIN')
    pts_p = pond_col('PTS')
    reb_p = pond_col('REB')
    ast_p = pond_col('AST')
    pra_p = pond_col('PRA') if 'PRA' in df_v.columns else (pts_p + reb_p + ast_p)

    tasa_pts_5 = df_5['PTS'].sum() / max(df_5['MIN'].sum(), 1.0)
    tasa_pts_10 = df_10['PTS'].sum() / max(df_10['MIN'].sum(), 1.0)
    tasa_pts = (0.55 * tasa_pts_5) + (0.45 * tasa_pts_10)

    tasa_reb_5 = df_5['REB'].sum() / max(df_5['MIN'].sum(), 1.0)
    tasa_reb_10 = df_10['REB'].sum() / max(df_10['MIN'].sum(), 1.0)
    tasa_reb = (0.55 * tasa_reb_5) + (0.45 * tasa_reb_10)

    tasa_ast_5 = df_5['AST'].sum() / max(df_5['MIN'].sum(), 1.0)
    tasa_ast_10 = df_10['AST'].sum() / max(df_10['MIN'].sum(), 1.0)
    tasa_ast = (0.55 * tasa_ast_5) + (0.45 * tasa_ast_10)

    return {
        'min': round(min_p, 1),
        'pts': round(pts_p, 1),
        'reb': round(reb_p, 1),
        'ast': round(ast_p, 1),
        'pra': round(pra_p, 1),
        'tasa_pts': tasa_pts,
        'tasa_reb': tasa_reb,
        'tasa_ast': tasa_ast
    }


def calcular_lineas_automaticas_jugador(df_jugador_previo):
    if df_jugador_previo is None or df_jugador_previo.empty:
        return {'pts': 15.5, 'reb': 5.5, 'ast': 3.5, 'pra': 24.5}

    df_v = df_jugador_previo[df_jugador_previo['MIN'] > 0]
    if df_v.empty:
        return {'pts': 15.5, 'reb': 5.5, 'ast': 3.5, 'pra': 24.5}

    st_pond = calcular_estadisticas_ponderadas_jugador(df_jugador_previo)
    df_10 = df_v.head(10)

    pts_ref = (0.65 * st_pond['pts']) + (0.35 * df_10['PTS'].median())
    reb_ref = (0.65 * st_pond['reb']) + (0.35 * df_10['REB'].median())
    ast_ref = (0.65 * st_pond['ast']) + (0.35 * df_10['AST'].median())
    pra_series = df_10['PRA'] if 'PRA' in df_10.columns else (df_10['PTS'] + df_10['REB'] + df_10['AST'])
    pra_ref = (0.65 * st_pond['pra']) + (0.35 * pra_series.median())

    def a_medio_punto(val):
        return max(np.floor(val) + 0.5, 0.5)

    return {
        'pts': round(a_medio_punto(pts_ref), 1),
        'reb': round(a_medio_punto(reb_ref), 1),
        'ast': round(a_medio_punto(ast_ref), 1),
        'pra': round(a_medio_punto(pra_ref), 1)
    }


def calcular_consistencia_y_estabilidad(df_jugador_previo, col, linea, direccion="MÁS"):
    if df_jugador_previo is None or df_jugador_previo.empty:
        return 50.0, 28.0, "🟡 Moderado"

    df_v = df_jugador_previo[df_jugador_previo['MIN'] > 0].head(10)
    if df_v.empty:
        return 50.0, 28.0, "🟡 Moderado"

    if col == 'PRA' and 'PRA' not in df_v.columns:
        serie = df_v['PTS'] + df_v['REB'] + df_v['AST']
    else:
        serie = df_v[col]

    if direccion == "MÁS":
        aciertos = (serie > linea).sum()
    else:
        aciertos = (serie < linea).sum()

    hit_rate = (aciertos / len(serie)) * 100.0
    media = max(float(serie.mean()), 0.5)
    std = float(serie.std()) if len(serie) > 1 else media * 0.25
    if pd.isna(std):
        std = media * 0.25

    cv_pct = (std / media) * 100.0

    if cv_pct < 24.0:
        etiqueta_estab = "🟢 Estable"
    elif cv_pct <= 33.0:
        etiqueta_estab = "🟡 Moderado"
    else:
        etiqueta_estab = "🔴 Volátil"

    return round(hit_rate, 0), round(cv_pct, 1), etiqueta_estab


# ==============================================================================
# 2. CAPA 5 CALIBRADA: ALINEACIONES, REDISTRIBUCIÓN DE MINUTOS Y USAGE BUMP
# ==============================================================================

def normalizar_grupo_posicion(pos_str):
    p = str(pos_str).upper()
    if any(k in p for k in ['PG', 'SG', 'G']):
        return 'G'
    if any(k in p for k in ['C']):
        return 'C'
    return 'F'


def calcular_ajustes_rotacion_capa5(
    team_id,
    catalogo_jugadores,
    bajas_manuales_out,
    altas_manuales_in,
    fecha_limite_str,
    incluir_bajas_automaticas_espn=True,
    bajas_evento_ids=None,
    bajas_evento_nombres=None,
    datos_vivo_partido=None
):
    bajas_ev_ids_set = set(str(x) for x in (bajas_evento_ids or []))
    bajas_ev_noms_set = set(bajas_evento_nombres or [])

    try:
        dt_limite = datetime.strptime(fecha_limite_str, "%Y-%m-%d")
        fecha_corte_reciente = (dt_limite - timedelta(days=18)).strftime("%Y-%m-%d")
    except Exception:
        fecha_corte_reciente = "2000-01-01"

    roster_equipo = [
        (nom, inf) for nom, inf in catalogo_jugadores.items()
        if str(inf.get('team_id')) == str(team_id)
    ]

    jugadores_activos = []
    jugadores_out = []

    for nom, inf in roster_equipo:
        id_j = str(inf.get('id', ''))
        status_raw = inf.get('status', 'Active')

        es_baja_roster = es_estatus_baja(status_raw)
        es_baja_scoreboard = (id_j in bajas_ev_ids_set) or (normalizar_texto(nom) in bajas_ev_noms_set)
        es_baja_en_vivo = False
        if datos_vivo_partido and datos_vivo_partido.get("estado") in ("in", "post"):
            j_v = datos_vivo_partido["jugadores_por_id"].get(id_j) or datos_vivo_partido["jugadores_por_nombre"].get(normalizar_texto(nom))
            if j_v is None or j_v.get("dnp", False):
                es_baja_en_vivo = True
            elif datos_vivo_partido.get("cuarto", 1) >= 2 and j_v.get("MIN_ACTUAL", 0.0) == 0.0:
                es_baja_en_vivo = True

        es_baja_detectada = incluir_bajas_automaticas_espn and (es_baja_roster or es_baja_scoreboard or es_baja_en_vivo)

        if nom in bajas_manuales_out:
            esta_out = True
        elif nom in altas_manuales_in:
            esta_out = False
        else:
            esta_out = es_baja_detectada

        df_j = consultar_gamelog_jugador(id_j)
        if df_j is not None and not df_j.empty:
            df_prev = obtener_historial_previo_sin_leakage(df_j, fecha_limite_str)
            df_v = df_prev[df_prev['MIN'] > 0]
            if not df_v.empty:
                st_p = calcular_estadisticas_ponderadas_jugador(df_prev)
                min_base = st_p['min']
                ultima_fecha_jugada = str(df_v.iloc[0]['FECHA'])
                es_baja_reciente = (ultima_fecha_jugada >= fecha_corte_reciente) or (nom in bajas_manuales_out)

                usg_est = min(max((st_p['tasa_pts'] * 36.0 + st_p['tasa_ast'] * 14.0), 10.0), 38.0)
                grupo_pos = normalizar_grupo_posicion(inf.get('position', 'F'))

                registro = {
                    'nombre': nom,
                    'id': id_j,
                    'pos': inf.get('position', 'F'),
                    'grupo_pos': grupo_pos,
                    'min_base': min_base,
                    'usg_est': round(usg_est, 1),
                    'status_raw': status_raw,
                    'es_baja_reciente': es_baja_reciente
                }
                if esta_out and min_base >= 14.0:
                    jugadores_out.append(registro)
                elif not esta_out and min_base >= 10.0:
                    jugadores_activos.append(registro)

    ajustes = {}
    for j in jugadores_activos:
        ajustes[j['nombre']] = {
            'esta_out': False,
            'min_base': round(j['min_base'], 1),
            'min_ajustados': round(j['min_base'], 1),
            'extra_min': 0.0,
            'factor_usage_bump': 1.0,
            'multiplicador_combinado': 1.0,
            'usg_est': j['usg_est'],
            'motivos': []
        }

    for j_out in jugadores_out:
        ajustes[j_out['nombre']] = {
            'esta_out': True,
            'min_base': round(j_out['min_base'], 1),
            'min_ajustados': 0.0,
            'extra_min': -round(j_out['min_base'], 1),
            'factor_usage_bump': 0.0,
            'multiplicador_combinado': 0.0,
            'usg_est': j_out['usg_est'],
            'motivos': ['Jugador descartado / OUT']
        }

    bajas_redistribuibles = [b for b in jugadores_out if b['es_baja_reciente']]

    if not bajas_redistribuibles or not jugadores_activos:
        return ajustes, jugadores_out

    for baja in bajas_redistribuibles:
        min_vacantes_efectivos = baja['min_base'] * 0.35
        grupo_baja = baja['grupo_pos']
        usg_baja = baja['usg_est']

        misma_pos = sorted(
            [j for j in jugadores_activos if j['grupo_pos'] == grupo_baja],
            key=lambda x: x['min_base'],
            reverse=True
        )[:3]

        if misma_pos:
            pesos_dir = [0.55, 0.30, 0.15][:len(misma_pos)]
            suma_p = sum(pesos_dir)
            pesos_dir = [p / suma_p for p in pesos_dir]

            for idx, cand in enumerate(misma_pos):
                nom_c = cand['nombre']
                ganancia = min_vacantes_efectivos * pesos_dir[idx]
                tope_personal = min(cand['min_base'] + 3.0, 36.5)
                nuevo_min = min(ajustes[nom_c]['min_ajustados'] + ganancia, tope_personal)
                real_ganado = nuevo_min - ajustes[nom_c]['min_ajustados']
                ajustes[nom_c]['min_ajustados'] = round(nuevo_min, 1)
                ajustes[nom_c]['extra_min'] = round(ajustes[nom_c]['extra_min'] + real_ganado, 1)
                if real_ganado >= 0.4:
                    ajustes[nom_c]['motivos'].append(f"+{real_ganado:.1f} min por baja de {baja['nombre']}")

        if usg_baja > 26.0:
            exceso_usg = (usg_baja - 26.0) / 100.0
            principales = sorted(jugadores_activos, key=lambda x: x['usg_est'], reverse=True)[:3]
            for cand in principales:
                nom_c = cand['nombre']
                bump = min(0.02 + (exceso_usg * 0.25), 0.05)
                ajustes[nom_c]['factor_usage_bump'] = round(min(ajustes[nom_c]['factor_usage_bump'] + bump, 1.07), 3)
                ajustes[nom_c]['motivos'].append(f"Usage Bump +{bump*100:.1f}% (Baja {baja['nombre']})")

    for nom_c, dat in ajustes.items():
        if dat['esta_out']:
            continue
        ratio_min = dat['min_ajustados'] / max(dat['min_base'], 1.0)
        mult_bruto = ratio_min * dat['factor_usage_bump']
        mult_tope = min(mult_bruto, 1.10)
        dat['multiplicador_combinado'] = round(mult_tope, 3)

    return ajustes, jugadores_out


# ==============================================================================
# 3. CAPAS 1, 2 Y 3 CALIBRADAS: CLASIFICACIÓN POR EDGE NETO Y PRA
# ==============================================================================

def clasificar_oportunidad_partido(abrev_ganador, edge_ml, dir_total, linea_ou, edge_ou, umbral_edge=UMBRAL_EDGE_OPERABLE):
    ml_valido = edge_ml >= umbral_edge
    ou_valido = edge_ou >= umbral_edge

    if ml_valido and ou_valido:
        nivel = "💎 DOBLE EDGE (ML + O/U)"
        mercado_op = f"{abrev_ganador} ML + {dir_total} {linea_ou}"
        score_orden = max(edge_ml, edge_ou) + 12.0
    elif ou_valido and not ml_valido:
        nivel = "🎯 EDGE SOLO TOTAL (O/U)"
        mercado_op = f"Solo {dir_total} {linea_ou} (Evitar ML)"
        score_orden = edge_ou + 4.0
    elif ml_valido and not ou_valido:
        nivel = "🏆 EDGE SOLO GANADOR (ML)"
        mercado_op = f"Solo {abrev_ganador} ML (Evitar O/U)"
        score_orden = edge_ml + 4.0
    else:
        nivel = "⚖️ Sin Ventaja Clara"
        mercado_op = "No operar (Edge insuficiente)"
        score_orden = max(edge_ml, edge_ou)

    return {
        'nivel': nivel,
        'mercado_operable': mercado_op,
        'ml_valido': ml_valido,
        'ou_valido': ou_valido,
        'es_operable': (ml_valido or ou_valido),
        'score_orden': round(score_orden, 2)
    }


def calcular_contexto_colectivo_capa1(
    id_t1,
    id_t2,
    metricas_equipos,
    promedios_liga,
    linea_mercado_ou=None,
    spread_mercado=None,
    t1_en_b2b=False,
    t2_en_b2b=False
):
    default_eq = {'pace': 99.5, 'off_rtg': 114.5, 'def_rtg': 114.5, 'pts_favor': 114.0, 'pts_contra': 114.0}
    eq1 = metricas_equipos.get(str(id_t1), default_eq)
    eq2 = metricas_equipos.get(str(id_t2), default_eq)

    pace_liga = promedios_liga.get('pace_liga', 99.5)
    def_liga = promedios_liga.get('def_liga', 114.5)

    pace_1 = eq1['pace'] * (0.99 if t1_en_b2b else 1.0)
    off_1 = eq1['off_rtg'] * (0.982 if t1_en_b2b else 1.0)

    pace_2 = eq2['pace'] * (0.99 if t2_en_b2b else 1.0)
    off_2 = eq2['off_rtg'] * (0.982 if t2_en_b2b else 1.0)

    posesiones_esperadas = (pace_1 * pace_2) / pace_liga

    lambda_t1 = (posesiones_esperadas / 100.0) * off_1 * (eq2['def_rtg'] / def_liga)
    lambda_t2 = (posesiones_esperadas / 100.0) * off_2 * (eq1['def_rtg'] / def_liga)

    total_proyectado = lambda_t1 + lambda_t2

    if linea_mercado_ou is not None and linea_mercado_ou > 150.0:
        linea_evaluada = linea_mercado_ou
        origen_linea = "Mercado Oficial"
    else:
        base_bruta = ((eq1['pts_favor'] + eq2['pts_contra']) / 2.0) + ((eq2['pts_favor'] + eq1['pts_contra']) / 2.0)
        linea_evaluada = round(base_bruta) + 0.5
        origen_linea = "Auto-Referencia"

    std_diferencia = 13.5
    std_total_partido = 16.0

    diferencia_esperada = lambda_t1 - lambda_t2
    spread_modelo = abs(diferencia_esperada)
    spread_ref = max(spread_modelo, spread_mercado) if spread_mercado is not None else spread_modelo

    if spread_ref >= 12.0:
        riesgo_paliza = "ALTO"
        factor_recorte_paliza = 0.91
    elif spread_ref >= 8.5:
        riesgo_paliza = "MODERADO"
        factor_recorte_paliza = 0.95
    else:
        riesgo_paliza = "BAJO"
        factor_recorte_paliza = 1.0

    prob_t1 = norm.cdf(diferencia_esperada / std_diferencia) * 100.0
    prob_t1 = min(max(prob_t1, 5.0), 95.0)
    prob_t2 = 100.0 - prob_t1

    z_total = (linea_evaluada - total_proyectado) / std_total_partido
    prob_over_total = (1.0 - norm.cdf(z_total)) * 100.0

    return {
        'posesiones_esperadas': round(posesiones_esperadas, 1),
        'lambda_t1': round(lambda_t1, 1),
        'lambda_t2': round(lambda_t2, 1),
        'total_proyectado': round(total_proyectado, 1),
        'linea_evaluada': linea_evaluada,
        'origen_linea': origen_linea,
        'prob_t1': round(prob_t1, 1),
        'prob_t2': round(prob_t2, 1),
        'prob_over_total': round(prob_over_total, 1),
        'spread_proyectado': round(spread_ref, 1),
        'riesgo_paliza': riesgo_paliza,
        'factor_recorte_paliza': factor_recorte_paliza,
        't1_en_b2b': t1_en_b2b,
        't2_en_b2b': t2_en_b2b,
        'eq1_stats': eq1,
        'eq2_stats': eq2
    }


def aplicar_ajustes_fatiga_y_paliza_minutos(
    minutos_base,
    en_b2b=False,
    factor_recorte_paliza=1.0
):
    min_aj = minutos_base
    etiquetas = []

    if en_b2b and minutos_base >= 25.0:
        min_aj *= 0.935
        etiquetas.append("B2B (-6.5% min)")

    if factor_recorte_paliza < 1.0 and minutos_base >= 24.0:
        min_aj *= factor_recorte_paliza
        pct_recorte = int(round((1.0 - factor_recorte_paliza) * 100))
        etiquetas.append(f"Riesgo Paliza (-{pct_recorte}% min)")

    return round(min_aj, 1), etiquetas


def calcular_proyeccion_jugador_capa2_3(
    df_jugador,
    minutos_proyectados,
    linea_pts,
    linea_reb,
    linea_ast,
    linea_pra=None,
    posesiones_partido=None,
    def_rtg_rival=None,
    promedios_liga=None,
    factor_usage_bump=1.0,
    min_base_sin_bajas=None,
    en_b2b=False,
    ventana_juegos=10
):
    df_valid = df_jugador[df_jugador['MIN'] > 0].head(ventana_juegos)
    if df_valid.empty:
        return None

    if linea_pra is None:
        linea_pra = round(linea_pts + linea_reb + linea_ast - 0.5, 1)

    st_pond = calcular_estadisticas_ponderadas_jugador(df_jugador)
    tasa_pts_min = st_pond['tasa_pts']
    tasa_reb_min = st_pond['tasa_reb']
    tasa_ast_min = st_pond['tasa_ast']
    min_promedio_hist = st_pond['min']

    pace_liga = promedios_liga.get('pace_liga', 99.5) if promedios_liga else 99.5
    def_liga = promedios_liga.get('def_liga', 114.5) if promedios_liga else 114.5

    factor_ritmo_raw = (posesiones_partido / pace_liga) if posesiones_partido else 1.0
    factor_ritmo = 1.0 + ((factor_ritmo_raw - 1.0) * 0.65)

    factor_rival_raw = (def_rtg_rival / def_liga) if def_rtg_rival else 1.0
    factor_rival = 1.0 + ((factor_rival_raw - 1.0) * 0.75)

    factor_fatiga_eficiencia = 0.975 if en_b2b else 1.0

    min_ref = min_base_sin_bajas if min_base_sin_bajas else min_promedio_hist
    lambda_pts_sin_bajas = tasa_pts_min * min_ref * factor_ritmo * factor_rival
    lambda_reb_sin_bajas = tasa_reb_min * min_ref * factor_ritmo
    lambda_ast_sin_bajas = tasa_ast_min * min_ref * factor_ritmo * (1.0 + (factor_rival - 1.0) * 0.5)

    lambda_pts_bruto = tasa_pts_min * minutos_proyectados * factor_ritmo * factor_rival * factor_usage_bump * factor_fatiga_eficiencia
    lambda_reb_bruto = tasa_reb_min * minutos_proyectados * factor_ritmo * (1.0 + (factor_usage_bump - 1.0) * 0.3)
    lambda_ast_bruto = tasa_ast_min * minutos_proyectados * factor_ritmo * (1.0 + (factor_rival - 1.0) * 0.5) * factor_usage_bump * factor_fatiga_eficiencia

    lambda_pts_tope = min(lambda_pts_bruto, lambda_pts_sin_bajas * 1.10) if (factor_usage_bump > 1.0 or minutos_proyectados > min_ref) else lambda_pts_bruto
    lambda_reb_tope = min(lambda_reb_bruto, lambda_reb_sin_bajas * 1.10) if (factor_usage_bump > 1.0 or minutos_proyectados > min_ref) else lambda_reb_bruto
    lambda_ast_tope = min(lambda_ast_bruto, lambda_ast_sin_bajas * 1.10) if (factor_usage_bump > 1.0 or minutos_proyectados > min_ref) else lambda_ast_bruto

    lambda_pts = (0.80 * lambda_pts_tope) + (0.20 * linea_pts)
    lambda_reb = (0.80 * lambda_reb_tope) + (0.20 * linea_reb)
    lambda_ast = (0.80 * lambda_ast_tope) + (0.20 * linea_ast)
    lambda_pra = lambda_pts + lambda_reb + lambda_ast

    std_hist_pts = df_valid['PTS'].std() if len(df_valid) > 1 else 6.0
    if pd.isna(std_hist_pts):
        std_hist_pts = 6.0
    std_pts = max(float(std_hist_pts), lambda_pts * 0.26, 5.2)
    z_score = (linea_pts - lambda_pts) / std_pts
    prob_more_pts = min(max((1.0 - norm.cdf(z_score)) * 100.0, 25.0), 75.0)
    prob_less_pts = 100.0 - prob_more_pts

    std_reb = max(float(df_valid['REB'].std()) if len(df_valid) > 1 else 2.5, np.sqrt(max(lambda_reb, 1.0)) * 1.25, 2.0)
    z_reb = (linea_reb - lambda_reb) / std_reb
    prob_more_reb = min(max((1.0 - norm.cdf(z_reb)) * 100.0, 25.0), 75.0)
    prob_less_reb = 100.0 - prob_more_reb

    std_ast = max(float(df_valid['AST'].std()) if len(df_valid) > 1 else 2.2, np.sqrt(max(lambda_ast, 1.0)) * 1.25, 1.8)
    z_ast = (linea_ast - lambda_ast) / std_ast
    prob_more_ast = min(max((1.0 - norm.cdf(z_ast)) * 100.0, 25.0), 75.0)
    prob_less_ast = 100.0 - prob_more_ast

    pra_series = df_valid['PRA'] if 'PRA' in df_valid.columns else (df_valid['PTS'] + df_valid['REB'] + df_valid['AST'])
    std_hist_pra = float(pra_series.std()) if len(pra_series) > 1 else 7.5
    if pd.isna(std_hist_pra):
        std_hist_pra = 7.5
    std_pra = max(std_hist_pra, lambda_pra * 0.22, 6.5)
    z_pra = (linea_pra - lambda_pra) / std_pra
    prob_more_pra = min(max((1.0 - norm.cdf(z_pra)) * 100.0, 25.0), 75.0)
    prob_less_pra = 100.0 - prob_more_pra

    mercados_eval = []
    configs = [
        ('PTS', lambda_pts, linea_pts, prob_more_pts, prob_less_pts),
        ('REB', lambda_reb, linea_reb, prob_more_reb, prob_less_reb),
        ('AST', lambda_ast, linea_ast, prob_more_ast, prob_less_ast),
        ('PRA', lambda_pra, linea_pra, prob_more_pra, prob_less_pra),
    ]
    for m_nom, m_lam, m_lin, p_m, p_l in configs:
        dir_m = "MÁS" if p_m >= 50.0 else "MENOS"
        prob_m = max(p_m, p_l)
        edge_m = round(prob_m - BREAK_EVEN_ODDS_110, 1)
        hr_m, cv_m, estab_m = calcular_consistencia_y_estabilidad(df_jugador, m_nom, m_lin, dir_m)
        penalizacion_anomalia = max(edge_m - 13.5, 0.0) * 0.35
        score_calidad = edge_m - penalizacion_anomalia + (hr_m - 50.0) * 0.14 - max(cv_m - 25.0, 0.0) * 0.15
        mercados_eval.append({
            'mercado': m_nom,
            'direccion': dir_m,
            'linea': m_lin,
            'lambda': round(m_lam, 1),
            'prob': round(prob_m, 1),
            'edge': edge_m,
            'hit_rate': hr_m,
            'cv': cv_m,
            'estabilidad': estab_m,
            'score_calidad': score_calidad
        })

    mejor_mercado = max(mercados_eval, key=lambda x: x['score_calidad'])

    return {
        'min_promedio_hist': round(min_promedio_hist, 1),
        'tasa_pts_min': round(tasa_pts_min, 3),
        'tasa_reb_min': round(tasa_reb_min, 3),
        'tasa_ast_min': round(tasa_ast_min, 3),
        'factor_ritmo': round(factor_ritmo, 3),
        'factor_rival': round(factor_rival, 3),
        'factor_usage_bump': round(factor_usage_bump, 3),
        'factor_fatiga': round(factor_fatiga_eficiencia, 3),
        'lambda_pts': round(lambda_pts, 2),
        'prob_more_pts': round(prob_more_pts, 1),
        'prob_less_pts': round(prob_less_pts, 1),
        'lambda_reb': round(lambda_reb, 2),
        'prob_more_reb': round(prob_more_reb, 1),
        'prob_less_reb': round(prob_less_reb, 1),
        'lambda_ast': round(lambda_ast, 2),
        'prob_more_ast': round(prob_more_ast, 1),
        'prob_less_ast': round(prob_less_ast, 1),
        'lambda_pra': round(lambda_pra, 2),
        'linea_pra': linea_pra,
        'prob_more_pra': round(prob_more_pra, 1),
        'prob_less_pra': round(prob_less_pra, 1),
        'mercados_eval': {m['mercado']: m for m in mercados_eval},
        'mejor_mercado': mejor_mercado
    }


# ==============================================================================
# 4. CAPA 4: EXTRACCIÓN Y MOTOR PROBABILÍSTICO EN VIVO (BOXSCORE OFICIAL)
# ==============================================================================

def parsear_reloj_a_minutos(reloj_str):
    if not reloj_str:
        return 0.0
    try:
        if ":" in str(reloj_str):
            mins, secs = str(reloj_str).split(":")
            return float(mins) + (float(secs) / 60.0)
        return float(reloj_str) / 60.0
    except Exception:
        return 0.0


def parsear_minutos_jugados_boxscore(val_min):
    if not val_min:
        return 0.0
    s = str(val_min).strip()
    try:
        if ":" in s:
            m, sec = s.split(":")
            return float(m) + (float(sec) / 60.0)
        return float(s)
    except Exception:
        return 0.0


@st.cache_data(ttl=25)
def obtener_datos_en_vivo_espn(event_id):
    if not event_id:
        return None
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={event_id}"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        res = requests.get(url, headers=headers, timeout=8).json()
        header_comp = res.get("header", {}).get("competitions", [{}])[0]
        status_obj = header_comp.get("status", {})
        status_type = status_obj.get("type", {})
        
        estado_tipo = status_type.get("state", "pre")
        detalle_estado = status_type.get("shortDetail") or status_type.get("description", "")
        cuarto = int(status_obj.get("period", 1))
        reloj_str = status_obj.get("displayClock", "12:00")
        min_reloj = parsear_reloj_a_minutos(reloj_str)
        
        competidores = header_comp.get("competitors", [])
        marcadores = {}
        for comp in competidores:
            abrev = comp.get("team", {}).get("abbreviation", "")
            marcadores[abrev] = int(comp.get("score", 0)) if str(comp.get("score", "0")).isdigit() else 0
            
        if estado_tipo == "pre":
            min_restantes_partido = 48.0
        elif estado_tipo == "post":
            min_restantes_partido = 0.0
        else:
            if cuarto <= 4:
                min_restantes_partido = ((4 - cuarto) * 12.0) + min_reloj
            else:
                min_restantes_partido = min_reloj

        stats_por_id = {}
        stats_por_nombre_norm = {}
        boxscore = res.get("boxscore", {}).get("players", [])
        
        for equipo_box in boxscore:
            abrev_eq = equipo_box.get("team", {}).get("abbreviation", "")
            estadisticas_lista = equipo_box.get("statistics", [])
            
            for bloque_stats in estadisticas_lista:
                nombres_cols = bloque_stats.get("names", [])
                atletas = bloque_stats.get("athletes", [])
                
                for atl in atletas:
                    info_atl = atl.get("athlete", {})
                    nombre = info_atl.get("displayName", "")
                    id_atl = str(info_atl.get("id", ""))
                    valores = atl.get("stats", [])
                    did_not_play = atl.get("didNotPlay", False)
                    
                    if valores and len(valores) == len(nombres_cols):
                        mapa_s = dict(zip(nombres_cols, valores))
                        min_jugados = parsear_minutos_jugados_boxscore(mapa_s.get("MIN", 0))
                        pts_a = int(mapa_s.get("PTS", 0)) if str(mapa_s.get("PTS", "0")).isdigit() else 0
                        reb_a = int(mapa_s.get("REB", 0)) if str(mapa_s.get("REB", "0")).isdigit() else 0
                        ast_a = int(mapa_s.get("AST", 0)) if str(mapa_s.get("AST", "0")).isdigit() else 0
                        registro_vivo = {
                            "id": id_atl,
                            "nombre": nombre,
                            "equipo": abrev_eq,
                            "dnp": did_not_play,
                            "MIN_ACTUAL": round(min_jugados, 1),
                            "PTS_ACTUAL": pts_a,
                            "REB_ACTUAL": reb_a,
                            "AST_ACTUAL": ast_a,
                            "PRA_ACTUAL": pts_a + reb_a + ast_a,
                            "FALTAS": int(mapa_s.get("PF", 0)) if str(mapa_s.get("PF", "0")).isdigit() else 0,
                        }
                    else:
                        registro_vivo = {
                            "id": id_atl,
                            "nombre": nombre,
                            "equipo": abrev_eq,
                            "dnp": did_not_play,
                            "MIN_ACTUAL": 0.0,
                            "PTS_ACTUAL": 0,
                            "REB_ACTUAL": 0,
                            "AST_ACTUAL": 0,
                            "PRA_ACTUAL": 0,
                            "FALTAS": 0,
                        }

                    if id_atl:
                        stats_por_id[id_atl] = registro_vivo
                    if nombre:
                        stats_por_nombre_norm[normalizar_texto(nombre)] = registro_vivo
                    
        return {
            "estado": estado_tipo,
            "detalle_estado": detalle_estado,
            "cuarto": cuarto,
            "reloj": reloj_str,
            "min_reloj": min_reloj,
            "min_restantes_partido": min_restantes_partido,
            "marcadores": marcadores,
            "jugadores_por_id": stats_por_id,
            "jugadores_por_nombre": stats_por_nombre_norm
        }
    except Exception as e:
        print(f"Error live summary: {e}")
        return None


def calcular_ajuste_vivo_capa4(
    datos_vivo,
    id_jugador_espn,
    nombre_jugador,
    df_historial,
    minutos_proyectados_totales=33.0,
    linea_pts=22.5,
    linea_reb=6.5,
    linea_ast=4.5,
    factor_ritmo=1.0,
    factor_rival=1.0,
    factor_usage_bump=1.0,
    penalizacion_faltas=0.20,
    penalizacion_blowout=0.85
):
    if not datos_vivo:
        return None

    j_vivo = datos_vivo["jugadores_por_id"].get(str(id_jugador_espn))
    if not j_vivo:
        j_vivo = datos_vivo["jugadores_por_nombre"].get(normalizar_texto(nombre_jugador))
    if not j_vivo:
        return None
        
    cuarto = datos_vivo["cuarto"]
    min_reloj = datos_vivo["min_reloj"]
    min_restantes_partido = datos_vivo["min_restantes_partido"]
    
    st_p = calcular_estadisticas_ponderadas_jugador(df_historial)
    tasa_pts = st_p['tasa_pts'] * factor_ritmo * factor_rival * factor_usage_bump
    tasa_reb = st_p['tasa_reb'] * factor_ritmo
    tasa_ast = st_p['tasa_ast'] * factor_ritmo * factor_usage_bump

    df_valid = df_historial[df_historial["MIN"] > 0].head(10)
    std_pts_hist = max(float(df_valid["PTS"].std()) if len(df_valid) > 1 else 6.0, 5.2)

    ratio_rotacion = min(minutos_proyectados_totales / 48.0, 1.0)
    min_restantes_jugador = min(min_restantes_partido * ratio_rotacion, min_restantes_partido)
    
    alertas = []
    factor_multiplicador = 1.0
    
    faltas = j_vivo["FALTAS"]
    en_foul_trouble = (
        (cuarto == 1 and faltas >= 2) or
        (cuarto == 2 and faltas >= 3) or
        (cuarto == 3 and faltas >= 4)
    )
    if en_foul_trouble and min_restantes_partido > 0:
        factor_multiplicador *= (1.0 - penalizacion_faltas)
        alertas.append(f"⚠️ Foul Trouble ({faltas} faltas en Q{cuarto}): -{int(penalizacion_faltas*100)}% min restantes")

    scores = list(datos_vivo["marcadores"].values())
    diferencia = abs(scores[0] - scores[1]) if len(scores) == 2 else 0
    
    en_blowout = False
    if cuarto == 4 and min_reloj >= 6.0 and diferencia >= 20:
        en_blowout = True
    elif cuarto >= 4 and min_reloj < 6.0 and diferencia >= 15:
        en_blowout = True
        
    if en_blowout and min_restantes_partido > 0:
        factor_multiplicador *= (1.0 - penalizacion_blowout)
        alertas.append(f"🚨 Blowout Factor (Diferencia de {diferencia} pts en Q{cuarto}): -{int(penalizacion_blowout*100)}% min titulares")

    min_restantes_ajustados = min_restantes_jugador * factor_multiplicador

    proj_pts = j_vivo["PTS_ACTUAL"] + (tasa_pts * min_restantes_ajustados)
    proj_reb = j_vivo["REB_ACTUAL"] + (tasa_reb * min_restantes_ajustados)
    proj_ast = j_vivo["AST_ACTUAL"] + (tasa_ast * min_restantes_ajustados)

    fraccion_tiempo_restante = max(min_restantes_ajustados / max(minutos_proyectados_totales, 1.0), 0.05)
    std_pts_restante = std_pts_hist * np.sqrt(fraccion_tiempo_restante)
    
    if j_vivo["PTS_ACTUAL"] > linea_pts:
        prob_over_pts = 100.0
    elif min_restantes_ajustados == 0:
        prob_over_pts = 100.0 if j_vivo["PTS_ACTUAL"] > linea_pts else 0.0
    else:
        z_vivo = (linea_pts - proj_pts) / std_pts_restante
        prob_over_pts = (1 - norm.cdf(z_vivo)) * 100.0

    rem_reb_necesario = int(np.floor(linea_reb - j_vivo["REB_ACTUAL"]))
    if rem_reb_necesario < 0:
        prob_over_reb = 100.0
    else:
        lambda_reb_rem = max(tasa_reb * min_restantes_ajustados, 1e-6)
        prob_over_reb = (1 - poisson.cdf(rem_reb_necesario, lambda_reb_rem)) * 100.0

    rem_ast_necesario = int(np.floor(linea_ast - j_vivo["AST_ACTUAL"]))
    if rem_ast_necesario < 0:
        prob_over_ast = 100.0
    else:
        lambda_ast_rem = max(tasa_ast * min_restantes_ajustados, 1e-6)
        prob_over_ast = (1 - poisson.cdf(rem_ast_necesario, lambda_ast_rem)) * 100.0

    return {
        "actual": j_vivo,
        "min_restantes_ajustados": round(min_restantes_ajustados, 1),
        "alertas": alertas,
        "proj_pts": round(proj_pts, 1),
        "prob_over_pts": round(prob_over_pts, 1),
        "proj_reb": round(proj_reb, 1),
        "prob_over_reb": round(prob_over_reb, 1),
        "proj_ast": round(proj_ast, 1),
        "prob_over_ast": round(prob_over_ast, 1),
    }


# ==============================================================================
# 5. COMPONENTES VISUALES
# ==============================================================================

def renderizar_barra_probabilidad_dual(abrev_t1, prob_t1, abrev_t2, prob_t2):
    p1 = min(max(float(prob_t1), 5.0), 95.0)
    p2 = round(100.0 - p1, 1)
    p1 = round(p1, 1)

    badge_ganador = ""
    edge_fav = round(max(p1, p2) - BREAK_EVEN_ODDS_110, 1)
    if edge_fav >= UMBRAL_EDGE_OPERABLE:
        fav = abrev_t1 if p1 >= p2 else abrev_t2
        badge_ganador = f' <span class="badge-high-conf">🏆 VENTAJA ML ({fav} {max(p1, p2):.1f}% | Edge +{edge_fav}%)</span>'
    else:
        badge_ganador = f' <span class="badge-normal-conf">⚠️ ML Sin Edge ({edge_fav:+.1f}%)</span>'

    st.markdown(f"""
        <div style="margin-top: 6px;">
            <span style="font-weight: 700; font-size: 14px;">Probabilidad de Victoria:</span>
            <span style="color: #60a5fa; font-weight: 700;">{abrev_t1} ({p1:.1f}%)</span> vs
            <span style="color: #fb923c; font-weight: 700;">{abrev_t2} ({p2:.1f}%)</span>
            {badge_ganador}
        </div>
        <div class="dual-prob-container">
            <div class="dual-prob-t1" style="width: {p1}%;">🏀 {abrev_t1} {p1:.1f}%</div>
            <div class="dual-prob-t2" style="width: {p2}%;">{p2:.1f}% {abrev_t2} 🏀</div>
        </div>
    """, unsafe_allow_html=True)


def obtener_html_badge_confianza(prob_rec):
    edge_val = round(prob_rec - BREAK_EVEN_ODDS_110, 1)
    if prob_rec >= UMBRAL_ALTA_CONFIANZA and edge_val >= UMBRAL_EDGE_OPERABLE:
        return f'<div class="badge-high-conf">🔥 ALTA CONFIANZA ({prob_rec:.1f}% | Edge +{edge_val}%)</div>'
    return f'<div class="badge-normal-conf">⚖️ Estándar ({prob_rec:.1f}% | Edge {edge_val:+.1f}%)</div>'


# ==============================================================================
# 6. INTERFAZ VISUAL PRINCIPAL Y CONTROLES EN BARRA LATERAL
# ==============================================================================

st.title("🏀 NBA Real-Time Analytics & Backtesting Engine")
st.markdown("Motor Probabilístico Institucional + **Buscador Web en Vivo de Noticias y Reporte Médico NBA**.")

st.sidebar.header("⚙️ Configuración y Auditoría")

col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    if st.button("📅 Hoy"):
        st.session_state['fecha_input'] = datetime.now(timezone(timedelta(hours=-5))).date()
        st.rerun()
with col_sb2:
    if st.button("🔄 Refrescar"):
        st.cache_data.clear()
        st.rerun()

fecha_defecto = st.session_state.get('fecha_input', datetime.now(timezone(timedelta(hours=-5))).date())
fecha_seleccionada = st.sidebar.date_input("Seleccionar Fecha de Cartelera", value=fecha_defecto, key='fecha_input')
fecha_str_api = fecha_seleccionada.strftime("%Y%m%d")
fecha_busqueda_str = fecha_seleccionada.strftime("%Y-%m-%d")

with st.spinner(f"Sincronizando cartelera, reportes médicos, fatiga B2B y ratings para el {fecha_seleccionada}..."):
    cartelera_fecha = obtener_cartelera_por_fecha(fecha_str_api)
    equipos_b2b_hoy = detectar_equipos_back_to_back(fecha_seleccionada)
    catalogo_jugadores = obtener_mapa_jugadores_completo()
    metricas_equipos, promedios_liga = obtener_metricas_avanzadas_equipos()

if not catalogo_jugadores:
    catalogo_jugadores = {
        "Nikola Jokic": {"id": "3112335", "team_id": "7", "team_abbrev": "DEN", "team_name": "Denver Nuggets", "status": "Active", "position": "C"}
    }

nombres_disponibles = sorted(list(catalogo_jugadores.keys()))
jugador_seleccionado = st.sidebar.selectbox("Selecciona o Busca un Jugador (Opcional)", ["-- Selecciona un jugador --"] + nombres_disponibles)

if jugador_seleccionado != "-- Selecciona un jugador --":
    info_sel = catalogo_jugadores[jugador_seleccionado]
    status_actual = info_sel.get('status', 'Active')
    status_lower = status_actual.lower()

    if es_estatus_baja(status_lower):
        st.sidebar.markdown(f'<div class="status-injured">❌ <b>Estatus:</b> {status_actual} (Lesionado / Fuera)</div>', unsafe_allow_html=True)
    elif any(word in status_lower for word in ['questionable', 'duda', 'day-to-day', 'probable']):
        st.sidebar.markdown(f'<div class="status-questionable">⚠️ <b>Estatus:</b> {status_actual} (En Duda)</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown(f'<div class="status-active">✅ <b>Estatus:</b> {status_actual} (Disponible)</div>', unsafe_allow_html=True)

# --- CAPA 5: CONTROLES MANUALES OUT / IN EN EL SIDEBAR ---
st.sidebar.markdown("---")
st.sidebar.subheader("🏥 Alineaciones y Bajas (Capa 5)")

equipos_en_cartelera = set()
for p in cartelera_fecha:
    equipos_en_cartelera.add(str(p['id_t1']))
    equipos_en_cartelera.add(str(p['id_t2']))

jugadores_jornada = sorted([
    nom for nom, inf in catalogo_jugadores.items()
    if str(inf.get('team_id')) in equipos_en_cartelera
]) if equipos_en_cartelera else nombres_disponibles

incluir_bajas_espn = st.sidebar.checkbox("Aplicar bajas automáticas del reporte médico", value=True)
bajas_manuales_out = st.sidebar.multiselect(
    "Marcar Jugadores como OUT (Baja de última hora)",
    options=jugadores_jornada,
    default=[]
)
altas_manuales_in = st.sidebar.multiselect(
    "Forzar Jugadores como IN (Activos hoy)",
    options=jugadores_jornada,
    default=[]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Filtros de Calidad de Mercado")
filtro_equipos_modo = st.sidebar.selectbox(
    "Filtrar Tabla de Partidos (Equipos)",
    [
        "Todos los partidos (Ordenados por Calidad)",
        "💎 Solo Doble Edge (ML + O/U)",
        "🎯 Mercados con Edge en Total (O/U ≥ +5%)",
        "🏆 Mercados con Edge en Ganador (ML ≥ +5%)",
        "✅ Solo Partidos Operables (Al menos 1 Edge ≥ +5%)"
    ],
    index=0
)
modo_mercado_resumen = st.sidebar.radio(
    "Estrategia en Tabla de Jugadores",
    ["🏆 Mejor Mercado Auto (PTS/REB/AST/PRA)", "🏀 Solo Mercado de Puntos (PTS)"],
    index=0
)
min_edge_filtro = st.sidebar.slider("Edge Mínimo vs Break-Even (%)", min_value=0.0, max_value=15.0, value=5.0, step=0.5)
min_hit_rate_filtro = st.sidebar.slider("Consistencia Mínima L10 (%)", min_value=0, max_value=90, value=60, step=10)
excluir_volatiles = st.sidebar.checkbox("Excluir jugadores 🔴 Volátiles (CV > 33%)", value=True)
aplicar_filtros_calidad = st.sidebar.checkbox("Activar Filtros de Calidad en Tabla Jugadores", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Fatiga y Automatización")
aplicar_b2b_y_paliza = st.sidebar.checkbox("Ajuste Automático Back-to-Back y Palizas", value=True)
usar_min_automaticos = st.sidebar.checkbox("Usar Minutos Ajustados Automáticos", value=True)
min_proj_manual = st.sidebar.slider("Minutos Manuales (si desactivas auto)", min_value=15.0, max_value=42.0, value=32.0, step=0.5)
modo_vivo = st.sidebar.toggle("Activar Polling en Vivo cada 30s", value=True)

with st.sidebar.expander("🛠️ Sobreescribir Líneas del Jugador (Opcional)"):
    sobreescribir_lineas = st.checkbox("Activar edición manual para jugador seleccionado", value=False)
    linea_pts_manual = st.number_input("Línea Puntos Manual", min_value=0.5, max_value=60.0, value=22.5, step=0.5)
    linea_reb_manual = st.number_input("Línea Rebotes Manual", min_value=0.5, max_value=30.0, value=6.5, step=0.5)
    linea_ast_manual = st.number_input("Línea Asistencias Manual", min_value=0.5, max_value=20.0, value=4.5, step=0.5)

st.sidebar.caption(
    f"📊 Promedios Liga: Pace {promedios_liga['pace_liga']:.1f} | "
    f"OffRtg {promedios_liga['off_liga']:.1f} | DefRtg {promedios_liga['def_liga']:.1f}"
)

# ==============================================================================
# PESTAÑAS PRINCIPALES DE LA APLICACIÓN
# ==============================================================================

tab_modelo, tab_buscador = st.tabs([
    "📊 Motor Probabilístico y Cartelera",
    "🔍 Buscador Web de Noticias y Lesiones"
])

# ------------------------------------------------------------------------------
# PESTAÑA 1: MOTOR PROBABILÍSTICO Y CARTELERA
# ------------------------------------------------------------------------------
with tab_modelo:
    st.subheader(f"📅 Cartelera de Partidos (Hora de Ecuador - UTC-5) — Fecha: {fecha_seleccionada}")

    resumen_equipos = []
    resumen_jugadores = []
    cache_ajustes_equipos = {}

    def obtener_ajustes_equipo_cached(team_id, bajas_ev_ids=None, bajas_ev_noms=None, datos_vivo=None):
        tid = str(team_id)
        if tid not in cache_ajustes_equipos:
            cache_ajustes_equipos[tid] = calcular_ajustes_rotacion_capa5(
                team_id=tid,
                catalogo_jugadores=catalogo_jugadores,
                bajas_manuales_out=bajas_manuales_out,
                altas_manuales_in=altas_manuales_in,
                fecha_limite_str=fecha_busqueda_str,
                incluir_bajas_automaticas_espn=incluir_bajas_espn,
                bajas_evento_ids=bajas_ev_ids,
                bajas_evento_nombres=bajas_ev_noms,
                datos_vivo_partido=datos_vivo
            )
        return cache_ajustes_equipos[tid]

    def hallar_top_jugadores_activos(lista_jugadores, ajustes_equipo, top_n=2):
        candidatos = []
        for nom, inf in lista_jugadores[:25]:
            aj = ajustes_equipo.get(nom, {})
            if aj.get('esta_out', False):
                continue
            if es_estatus_baja(inf.get('status', 'Active')) and nom not in altas_manuales_in:
                continue

            df_t = consultar_gamelog_jugador(inf['id'])
            if df_t is not None and len(df_t) >= 3:
                df_prev = obtener_historial_previo_sin_leakage(df_t, fecha_busqueda_str)
                df_v = df_prev[df_prev['MIN'] > 0].head(10)
                if df_v.empty:
                    continue
                st_p = calcular_estadisticas_ponderadas_jugador(df_prev)
                min_prom = st_p['min']
                pts_prom = st_p['pts']
                reb_prom = st_p['reb']
                ast_prom = st_p['ast']
               
                if min_prom >= 18.0:
                    score_jerarquia = pts_prom * 1.25 + reb_prom + ast_prom
                    candidatos.append((
                        score_jerarquia,
                        (nom, pts_prom, reb_prom, ast_prom, min_prom, inf.get('status', 'Active'), str(inf['id']), df_t, df_prev)
                    ))

        candidatos.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in candidatos[:top_n]]

    def evaluar_resultado_linea(valor_real, linea_evaluada, recomendacion_str):
        if valor_real == linea_evaluada:
            return "🟡 Push (Empate)"
        if recomendacion_str == "MÁS":
            return "🟢 Acierto" if valor_real > linea_evaluada else "🔴 Fallo"
        else:
            return "🟢 Acierto" if valor_real < linea_evaluada else "🔴 Fallo"

    if cartelera_fecha:
        for partido in cartelera_fecha:
            with st.container():
                t1_id = partido['id_t1']
                t2_id = partido['id_t2']

                partido_en_vivo = partido['estado_codigo'] == 'in'
                partido_finalizado = partido['estado_codigo'] == 'post' or partido['estado'].lower() in ['finalized', 'final', 'closed']

                datos_vivo_p = obtener_datos_en_vivo_espn(partido['event_id']) if (partido_en_vivo or partido_finalizado) else None

                t1_b2b = (str(t1_id) in equipos_b2b_hoy) if aplicar_b2b_y_paliza else False
                t2_b2b = (str(t2_id) in equipos_b2b_hoy) if aplicar_b2b_y_paliza else False

                ctx_partido = calcular_contexto_colectivo_capa1(
                    t1_id, t2_id, metricas_equipos, promedios_liga,
                    linea_mercado_ou=partido.get('linea_mercado_ou'),
                    spread_mercado=partido.get('spread_mercado'),
                    t1_en_b2b=t1_b2b,
                    t2_en_b2b=t2_b2b
                )
                linea_ou_partido = ctx_partido['linea_evaluada']
                origen_ou_partido = ctx_partido['origen_linea']
                factor_paliza_partido = ctx_partido['factor_recorte_paliza'] if aplicar_b2b_y_paliza else 1.0

                ajustes_t1, bajas_t1 = obtener_ajustes_equipo_cached(
                    t1_id, partido.get('bajas_evento_ids'), partido.get('bajas_evento_nombres'), datos_vivo_p
                )
                ajustes_t2, bajas_t2 = obtener_ajustes_equipo_cached(
                    t2_id, partido.get('bajas_evento_ids'), partido.get('bajas_evento_nombres'), datos_vivo_p
                )

                badges_ctx = []
                if t1_b2b:
                    badges_ctx.append(f"🔋 {partido['abrev_t1']} en B2B")
                if t2_b2b:
                    badges_ctx.append(f"🔋 {partido['abrev_t2']} en B2B")
                if ctx_partido['riesgo_paliza'] != "BAJO" and aplicar_b2b_y_paliza:
                    badges_ctx.append(f"🚨 Riesgo Paliza {ctx_partido['riesgo_paliza']} (Spread ~{ctx_partido['spread_proyectado']} pts)")
                str_badges = f" | {' · '.join(badges_ctx)}" if badges_ctx else ""

                if partido_en_vivo:
                    marcador_str = f" 🔴 EN VIVO ({partido['estado']}): **{partido['abrev_t1']} {partido['score_t1']} - {partido['score_t2']} {partido['abrev_t2']}**"
                elif partido_finalizado:
                    marcador_str = f" 🏁 FINAL: **{partido['abrev_t1']} {partido['score_t1']} - {partido['score_t2']} {partido['abrev_t2']}**"
                else:
                    marcador_str = f" | Estado: `{partido['estado']}`"

                st.markdown(f"### ⏰ [{partido['hora_ecuador']} ECT] 🏟️ {partido['encuentro']} — *{partido['detalle']}*{marcador_str}")
                st.caption(
                    f"⚡ **Ritmo Esperado:** {ctx_partido['posesiones_esperadas']} pos | "
                    f"**{partido['abrev_t1']}:** Λ = {ctx_partido['lambda_t1']} pts | "
                    f"**{partido['abrev_t2']}:** Λ = {ctx_partido['lambda_t2']} pts{str_badges}"
                )

                if bajas_t1 or bajas_t2:
                    txt_b1 = f"**{partido['abrev_t1']} OUT:** {', '.join([b['nombre'] for b in bajas_t1])}" if bajas_t1 else ""
                    txt_b2 = f"**{partido['abrev_t2']} OUT:** {', '.join([b['nombre'] for b in bajas_t2])}" if bajas_t2 else ""
                    separador = " | " if (txt_b1 and txt_b2) else ""
                    st.warning(f"🏥 **Bajas excluidas automáticamente (Capa 5 activa):** {txt_b1}{separador}{txt_b2}")
               
                jugadores_t1 = [(nom, inf) for nom, inf in catalogo_jugadores.items() if str(inf.get('team_id')) == t1_id]
                jugadores_t2 = [(nom, inf) for nom, inf in catalogo_jugadores.items() if str(inf.get('team_id')) == t2_id]

                top_t1 = hallar_top_jugadores_activos(jugadores_t1, ajustes_t1, top_n=2)
                top_t2 = hallar_top_jugadores_activos(jugadores_t2, ajustes_t2, top_n=2)

                def procesar_jugadores_equipo_resumen(lista_top, abrev_eq, ajustes_eq, en_b2b_eq, def_rtg_op):
                    for cand_j in lista_top:
                        nom_j, pts_j, reb_j, ast_j, min_j, stat_j, id_j, df_j_full, df_j_prev = cand_j
                        lineas_auto = calcular_lineas_automaticas_jugador(df_j_prev)

                        aj = ajustes_eq.get(nom_j, {})
                        min_aj_c5 = aj.get('min_ajustados', min_j)
                        min_aj_ctx, _ = aplicar_ajustes_fatiga_y_paliza_minutos(
                            min_aj_c5, en_b2b=en_b2b_eq, factor_recorte_paliza=factor_paliza_partido
                        )
                        bump = aj.get('factor_usage_bump', 1.0)
                        min_eval = min_aj_ctx if usar_min_automaticos else min_proj_manual

                        proy = calcular_proyeccion_jugador_capa2_3(
                            df_jugador=df_j_prev,
                            minutos_proyectados=min_eval,
                            linea_pts=lineas_auto['pts'],
                            linea_reb=lineas_auto['reb'],
                            linea_ast=lineas_auto['ast'],
                            linea_pra=lineas_auto['pra'],
                            posesiones_partido=ctx_partido['posesiones_esperadas'],
                            def_rtg_rival=def_rtg_op,
                            promedios_liga=promedios_liga,
                            factor_usage_bump=bump,
                            min_base_sin_bajas=min_j,
                            en_b2b=en_b2b_eq
                        )
                        if not proy:
                            continue

                        if "Mejor Mercado Auto" in modo_mercado_resumen:
                            m_sel = proy['mejor_mercado']
                        else:
                            m_sel = proy['mercados_eval']['PTS']

                        tipo_m = m_sel['mercado']
                        dir_m = m_sel['direccion']
                        linea_m = m_sel['linea']
                        lambda_m = m_sel['lambda']
                        prob_m = m_sel['prob']
                        edge_m = m_sel['edge']
                        hr_m = m_sel['hit_rate']
                        cv_m = m_sel['cv']
                        estab_m = m_sel['estabilidad']

                        es_alta_conf = (prob_m >= UMBRAL_ALTA_CONFIANZA and edge_m >= 5.0 and hr_m >= 60.0 and "Volátil" not in estab_m)

                        real_str = "-"
                        desglose_str = "-"
                        res_estado = "Pendiente"

                        if partido_finalizado:
                            partido_j_fecha = df_j_full[df_j_full['FECHA'] == fecha_busqueda_str]
                            if not partido_j_fecha.empty and partido_j_fecha.iloc[0]['MIN'] > 0:
                                p_r = int(partido_j_fecha.iloc[0]['PTS'])
                                r_r = int(partido_j_fecha.iloc[0]['REB'])
                                a_r = int(partido_j_fecha.iloc[0]['AST'])
                                pra_r = p_r + r_r + a_r
                                mapa_reales = {'PTS': p_r, 'REB': r_r, 'AST': a_r, 'PRA': pra_r}
                                val_eval = mapa_reales[tipo_m]
                                real_str = f"{val_eval} {tipo_m}"
                                desglose_str = f"{p_r}P / {r_r}R / {a_r}A"
                                res_estado = evaluar_resultado_linea(val_eval, linea_m, dir_m)
                            elif datos_vivo_p:
                                jv = datos_vivo_p["jugadores_por_id"].get(str(id_j)) or datos_vivo_p["jugadores_por_nombre"].get(normalizar_texto(nom_j))
                                if jv and jv["MIN_ACTUAL"] > 0:
                                    mapa_reales = {'PTS': jv['PTS_ACTUAL'], 'REB': jv['REB_ACTUAL'], 'AST': jv['AST_ACTUAL'], 'PRA': jv['PRA_ACTUAL']}
                                    val_eval = mapa_reales[tipo_m]
                                    real_str = f"{val_eval} {tipo_m}"
                                    desglose_str = f"{jv['PTS_ACTUAL']}P / {jv['REB_ACTUAL']}R / {jv['AST_ACTUAL']}A"
                                    res_estado = evaluar_resultado_linea(val_eval, linea_m, dir_m)
                                else:
                                    res_estado = "No jugó"
                            else:
                                res_estado = "No jugó"

                        elif partido_en_vivo and datos_vivo_p:
                            jv = datos_vivo_p["jugadores_por_id"].get(str(id_j)) or datos_vivo_p["jugadores_por_nombre"].get(normalizar_texto(nom_j))
                            if jv:
                                mapa_v = {'PTS': jv['PTS_ACTUAL'], 'REB': jv['REB_ACTUAL'], 'AST': jv['AST_ACTUAL'], 'PRA': jv['PRA_ACTUAL']}
                                val_v = mapa_v[tipo_m]
                                real_str = f"🔴 {val_v} {tipo_m} ({jv['MIN_ACTUAL']:.0f}m)"
                                desglose_str = f"{jv['PTS_ACTUAL']}P / {jv['REB_ACTUAL']}R / {jv['AST_ACTUAL']}A"
                                if dir_m == "MÁS" and val_v > linea_m:
                                    res_estado = "🟢 Superada (Vivo)"
                                elif dir_m == "MENOS" and val_v > linea_m:
                                    res_estado = "🔴 Superada en contra"
                                else:
                                    res_estado = f"⏳ En juego ({partido['estado']})"

                        resumen_jugadores.append({
                            'Nivel': '🔥 TOP PICK' if es_alta_conf else 'Estándar',
                            'Encuentro': partido['encuentro'],
                            'Jugador': nom_j,
                            'Equipo': abrev_eq,
                            'Mercado': tipo_m,
                            'Min Proy.': round(min_eval, 1),
                            'Λ Proy.': lambda_m,
                            'Línea': linea_m,
                            'Recomendación': f"{dir_m} {linea_m} {tipo_m}",
                            'Prob.': f"{prob_m:.1f}%",
                            'Edge %': f"{edge_m:+.1f}%",
                            'Hit Rate L10': f"{int(hr_m)}%",
                            'Estabilidad': f"{estab_m} ({cv_m}%)",
                            'Real / Vivo': real_str,
                            'Boxscore': desglose_str,
                            'Resultado': res_estado,
                            '_edge_num': edge_m,
                            '_hr_num': hr_m,
                            '_estab_txt': estab_m,
                            '_score_calidad': m_sel['score_calidad']
                        })

                procesar_jugadores_equipo_resumen(top_t1, partido['abrev_t1'], ajustes_t1, t1_b2b, ctx_partido['eq2_stats']['def_rtg'])
                procesar_jugadores_equipo_resumen(top_t2, partido['abrev_t2'], ajustes_t2, t2_b2b, ctx_partido['eq1_stats']['def_rtg'])
               
                col_p1, col_p2 = st.columns(2)
               
                with col_p1:
                    if top_t1:
                        nom_1, pts_1, reb_1, ast_1, min_1, stat_1, id_1, df_j1_full, df_j1_prev = top_t1[0]
                        lineas_auto_1 = calcular_lineas_automaticas_jugador(df_j1_prev)
                        aj_1 = ajustes_t1.get(nom_1, {})
                        min_aj_ctx_1, notas_min_1 = aplicar_ajustes_fatiga_y_paliza_minutos(
                            aj_1.get('min_ajustados', min_1), en_b2b=t1_b2b, factor_recorte_paliza=factor_paliza_partido
                        )
                        min_eval_1 = min_aj_ctx_1 if usar_min_automaticos else min_proj_manual
                        proy_1 = calcular_proyeccion_jugador_capa2_3(
                            df_jugador=df_j1_prev, minutos_proyectados=min_eval_1,
                            linea_pts=lineas_auto_1['pts'], linea_reb=lineas_auto_1['reb'],
                            linea_ast=lineas_auto_1['ast'], linea_pra=lineas_auto_1['pra'],
                            posesiones_partido=ctx_partido['posesiones_esperadas'], def_rtg_rival=ctx_partido['eq2_stats']['def_rtg'],
                            promedios_liga=promedios_liga, factor_usage_bump=aj_1.get('factor_usage_bump', 1.0),
                            min_base_sin_bajas=min_1, en_b2b=t1_b2b
                        )
                        mm1 = proy_1['mejor_mercado']
                        etiqueta_extra_1 = f" | {', '.join(notas_min_1)}" if notas_min_1 else ""
                        sello_exp_1 = f" 🔥 {mm1['direccion']} {mm1['linea']} {mm1['mercado']} (Edge {mm1['edge']:+.1f}%)" if mm1['edge'] >= 4.5 else ""
                        with st.expander(f"⭐ {partido['abrev_t1']} - {nom_1} (PTS Λ: {proy_1['lambda_pts']:.1f} vs {lineas_auto_1['pts']}{etiqueta_extra_1}){sello_exp_1}"):
                            st.markdown(obtener_html_badge_confianza(mm1['prob']), unsafe_allow_html=True)
                            st.markdown(
                                f"**Mejor Mercado:** `{mm1['direccion']} {mm1['linea']} {mm1['mercado']}` (Λ: {mm1['lambda']}) | "
                                f"**Hit Rate L10:** {int(mm1['hit_rate'])}% | **Estabilidad:** {mm1['estabilidad']}"
                            )
                            st.metric(
                                f"Proyección Completa ({lineas_auto_1['pts']}P / {lineas_auto_1['reb']}R / {lineas_auto_1['ast']}A / {lineas_auto_1['pra']} PRA)",
                                f"{proy_1['lambda_pts']:.1f} P | {proy_1['lambda_reb']:.1f} R | {proy_1['lambda_ast']:.1f} A | {proy_1['lambda_pra']:.1f} PRA"
                            )
                            if st.button(f"Analizar a {nom_1} en Vivo / Detalle", key=f"btn_{id_1}_{partido['event_id']}"):
                                st.session_state['jugador_seleccionado'] = nom_1
                                st.rerun()
                    else:
                        st.info(f"Sin jugadores sanos activos para {partido['abrev_t1']}")

                with col_p2:
                    if top_t2:
                        nom_2, pts_2, reb_2, ast_2, min_2, stat_2, id_2, df_j2_full, df_j2_prev = top_t2[0]
                        lineas_auto_2 = calcular_lineas_automaticas_jugador(df_j2_prev)
                        aj_2 = ajustes_t2.get(nom_2, {})
                        min_aj_ctx_2, notas_min_2 = aplicar_ajustes_fatiga_y_paliza_minutos(
                            aj_2.get('min_ajustados', min_2), en_b2b=t2_b2b, factor_recorte_paliza=factor_paliza_partido
                        )
                        min_eval_2 = min_aj_ctx_2 if usar_min_automaticos else min_proj_manual
                        proy_2 = calcular_proyeccion_jugador_capa2_3(
                            df_jugador=df_j2_prev, minutos_proyectados=min_eval_2,
                            linea_pts=lineas_auto_2['pts'], linea_reb=lineas_auto_2['reb'],
                            linea_ast=lineas_auto_2['ast'], linea_pra=lineas_auto_2['pra'],
                            posesiones_partido=ctx_partido['posesiones_esperadas'], def_rtg_rival=ctx_partido['eq1_stats']['def_rtg'],
                            promedios_liga=promedios_liga, factor_usage_bump=aj_2.get('factor_usage_bump', 1.0),
                            min_base_sin_bajas=min_2, en_b2b=t2_b2b
                        )
                        mm2 = proy_2['mejor_mercado']
                        etiqueta_extra_2 = f" | {', '.join(notas_min_2)}" if notas_min_2 else ""
                        sello_exp_2 = f" 🔥 {mm2['direccion']} {mm2['linea']} {mm2['mercado']} (Edge {mm2['edge']:+.1f}%)" if mm2['edge'] >= 4.5 else ""
                        with st.expander(f"⭐ {partido['abrev_t2']} - {nom_2} (PTS Λ: {proy_2['lambda_pts']:.1f} vs {lineas_auto_2['pts']}{etiqueta_extra_2}){sello_exp_2}"):
                            st.markdown(obtener_html_badge_confianza(mm2['prob']), unsafe_allow_html=True)
                            st.markdown(
                                f"**Mejor Mercado:** `{mm2['direccion']} {mm2['linea']} {mm2['mercado']}` (Λ: {mm2['lambda']}) | "
                                f"**Hit Rate L10:** {int(mm2['hit_rate'])}% | **Estabilidad:** {mm2['estabilidad']}"
                            )
                            st.metric(
                                f"Proyección Completa ({lineas_auto_2['pts']}P / {lineas_auto_2['reb']}R / {lineas_auto_2['ast']}A / {lineas_auto_2['pra']} PRA)",
                                f"{proy_2['lambda_pts']:.1f} P | {proy_2['lambda_reb']:.1f} R | {proy_2['lambda_ast']:.1f} A | {proy_2['lambda_pra']:.1f} PRA"
                            )
                            if st.button(f"Analizar a {nom_2} en Vivo / Detalle", key=f"btn_{id_2}_{partido['event_id']}"):
                                st.session_state['jugador_seleccionado'] = nom_2
                                st.rerun()
                    else:
                        st.info(f"Sin jugadores sanos activos para {partido['abrev_t2']}")
               
                prob_t1 = ctx_partido['prob_t1']
                prob_t2 = ctx_partido['prob_t2']
                total_proyectado_partido = ctx_partido['total_proyectado']
                is_more_partido = total_proyectado_partido > linea_ou_partido
                verdict_partido_total = "MÁS" if is_more_partido else "MENOS"
                prob_total_rec = ctx_partido['prob_over_total'] if is_more_partido else (100.0 - ctx_partido['prob_over_total'])
                ganador_recomendado = partido['abrev_t1'] if prob_t1 >= prob_t2 else partido['abrev_t2']
                max_prob_val = max(prob_t1, prob_t2)
                edge_ml = round(max_prob_val - BREAK_EVEN_ODDS_110, 1)
                edge_ou = round(prob_total_rec - BREAK_EVEN_ODDS_110, 1)

                clasif_eq = clasificar_oportunidad_partido(
                    abrev_ganador=ganador_recomendado,
                    edge_ml=edge_ml,
                    dir_total=verdict_partido_total,
                    linea_ou=linea_ou_partido,
                    edge_ou=edge_ou,
                    umbral_edge=UMBRAL_EDGE_OPERABLE
                )

                ganador_display = (
                    f"{ganador_recomendado} ({max_prob_val:.1f}% | Edge {edge_ml:+.1f}%)"
                    if clasif_eq['ml_valido']
                    else f"⚠️ {ganador_recomendado} ({max_prob_val:.1f}% | Edge {edge_ml:+.1f}% - Sin Edge)"
                )
                total_display = (
                    f"{verdict_partido_total} ({prob_total_rec:.1f}% | Edge {edge_ou:+.1f}%)"
                    if clasif_eq['ou_valido']
                    else f"⚠️ {verdict_partido_total} ({prob_total_rec:.1f}% | Edge {edge_ou:+.1f}% - Ruido)"
                )

                s1 = partido['score_t1']
                s2 = partido['score_t2']
                total_actual = s1 + s2

                if partido_finalizado:
                    ganador_real = partido['abrev_t1'] if s1 > s2 else partido['abrev_t2']
                    cumplio_total = evaluar_resultado_linea(total_actual, linea_ou_partido, verdict_partido_total)
                    acerto_ganador = "🟢 Acierto" if ganador_recomendado == ganador_real else "🔴 Fallo"
                   
                    resumen_equipos.append({
                        'Nivel': clasif_eq['nivel'],
                        'Mercado Operable': clasif_eq['mercado_operable'],
                        'Hora (ECT)': partido['hora_ecuador'],
                        'Encuentro': partido['encuentro'],
                        'Marcador': f"🏁 {partido['abrev_t1']} {s1} - {s2} {partido['abrev_t2']}",
                        'Ganador Sugerido': ganador_display,
                        'Resultado Ganador': acerto_ganador if clasif_eq['ml_valido'] else f"No operable ({acerto_ganador})",
                        'Línea O/U': f"{linea_ou_partido} ({origen_ou_partido})",
                        'Pts Proyectados': total_proyectado_partido,
                        'Predicción Total': total_display,
                        'Total Real / Vivo': f"{total_actual} pts",
                        'Resultado Total': cumplio_total if clasif_eq['ou_valido'] else f"No operable ({cumplio_total})",
                        'Estado': 'Finalizado',
                        '_ml_valido': clasif_eq['ml_valido'],
                        '_ou_valido': clasif_eq['ou_valido'],
                        '_es_operable': clasif_eq['es_operable'],
                        '_score_orden': clasif_eq['score_orden']
                    })

                    st.info(f"📊 **Resultado Final Real:** {partido['abrev_t1']} **{s1} - {s2}** {partido['abrev_t2']} (Total: {total_actual} pts)")
                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.markdown(f"**Ganador:** {ganador_display} | **Real:** {ganador_real} → {acerto_ganador}")
                    with col_res2:
                        st.markdown(
                            f"**Total Partido ({linea_ou_partido}):** Proy. {total_proyectado_partido:.1f} "
                            f"({total_display}) | Real: **{total_actual}** → {cumplio_total}"
                        )
                elif partido_en_vivo:
                    lider_actual = partido['abrev_t1'] if s1 > s2 else (partido['abrev_t2'] if s2 > s1 else "Empate")
                    estado_ganador_vivo = "🟢 Liderando" if lider_actual == ganador_recomendado else ("🟡 Empatado" if lider_actual == "Empate" else "🔴 Abajo")

                    resumen_equipos.append({
                        'Nivel': clasif_eq['nivel'],
                        'Mercado Operable': clasif_eq['mercado_operable'],
                        'Hora (ECT)': partido['hora_ecuador'],
                        'Encuentro': partido['encuentro'],
                        'Marcador': f"🔴 {partido['abrev_t1']} {s1} - {s2} {partido['abrev_t2']}",
                        'Ganador Sugerido': ganador_display,
                        'Resultado Ganador': estado_ganador_vivo,
                        'Línea O/U': f"{linea_ou_partido} ({origen_ou_partido})",
                        'Pts Proyectados': total_proyectado_partido,
                        'Predicción Total': total_display,
                        'Total Real / Vivo': f"🔴 {total_actual} pts",
                        'Resultado Total': f"⏳ {partido['estado']}",
                        'Estado': f"🔴 {partido['estado']}",
                        '_ml_valido': clasif_eq['ml_valido'],
                        '_ou_valido': clasif_eq['ou_valido'],
                        '_es_operable': clasif_eq['es_operable'],
                        '_score_orden': clasif_eq['score_orden']
                    })

                    col_r1, col_r2 = st.columns(2)
                    with col_r1:
                        if clasif_eq['ml_valido']:
                            st.success(f"🏆 **Ganador Operable:** **{ganador_display}** | Vivo: **{estado_ganador_vivo}**")
                        else:
                            st.info(f"⚖️ **Ganador Sin Edge:** {ganador_display} | Vivo: **{estado_ganador_vivo}**")
                    with col_r2:
                        if clasif_eq['ou_valido']:
                            st.warning(f"📈 **Total Operable ({linea_ou_partido}):** Proy. {total_proyectado_partido:.1f} → **{total_display}** | Acumulado: **{total_actual} pts**")
                        else:
                            st.info(f"⚖️ **Total Sin Edge ({linea_ou_partido}):** Proy. {total_proyectado_partido:.1f} → {total_display} | Acumulado: **{total_actual} pts**")
                else:
                    resumen_equipos.append({
                        'Nivel': clasif_eq['nivel'],
                        'Mercado Operable': clasif_eq['mercado_operable'],
                        'Hora (ECT)': partido['hora_ecuador'],
                        'Encuentro': partido['encuentro'],
                        'Marcador': '-',
                        'Ganador Sugerido': ganador_display,
                        'Resultado Ganador': 'Pendiente' if clasif_eq['ml_valido'] else 'No operar',
                        'Línea O/U': f"{linea_ou_partido} ({origen_ou_partido})",
                        'Pts Proyectados': total_proyectado_partido,
                        'Predicción Total': total_display,
                        'Total Real / Vivo': '-',
                        'Resultado Total': 'Pendiente' if clasif_eq['ou_valido'] else 'No operar',
                        'Estado': partido['estado'],
                        '_ml_valido': clasif_eq['ml_valido'],
                        '_ou_valido': clasif_eq['ou_valido'],
                        '_es_operable': clasif_eq['es_operable'],
                        '_score_orden': clasif_eq['score_orden']
                    })

                    col_r1, col_r2 = st.columns(2)
                    with col_r1:
                        if clasif_eq['ml_valido']:
                            st.success(f"🏆 **Ganador Operable (Moneyline):** **{ganador_display}**")
                        else:
                            st.info(f"⚖️ **Ganador Descartado:** {ganador_display}")
                    with col_r2:
                        if clasif_eq['ou_valido']:
                            st.warning(f"📈 **Total Operable ({linea_ou_partido} - {origen_ou_partido}):** Proy. {total_proyectado_partido:.1f} → **{total_display}**")
                        else:
                            st.info(f"⚖️ **Total Descartado ({linea_ou_partido}):** Proy. {total_proyectado_partido:.1f} → {total_display}")

                renderizar_barra_probabilidad_dual(partido['abrev_t1'], prob_t1, partido['abrev_t2'], prob_t2)
                st.markdown("---")
    else:
        st.info(f"No hay partidos registrados en la cartelera oficial para la fecha: {fecha_seleccionada}.")

    # --- RESUMEN EJECUTIVO DE EQUIPOS Y JUGADORES ---
    if resumen_equipos:
        st.subheader("📋 Resumen Ejecutivo: Pronósticos de Equipos (Clasificados por Edge Real)")
        st.markdown(
            f"Separa automáticamente **💎 Doble Edge**, **🎯 Edge Solo Total (O/U)** y **🏆 Edge Solo Ganador (ML)** "
            f"(umbral mínimo **Edge ≥ +{UMBRAL_EDGE_OPERABLE:.1f}%**)."
        )
       
        df_equipos = pd.DataFrame(resumen_equipos)
        df_equipos = df_equipos.sort_values(by='_score_orden', ascending=False)

        if "Doble Edge" in filtro_equipos_modo:
            df_equipos = df_equipos[df_equipos['_ml_valido'] & df_equipos['_ou_valido']]
        elif "Solo Total" in filtro_equipos_modo:
            df_equipos = df_equipos[df_equipos['_ou_valido']]
        elif "Solo Ganador" in filtro_equipos_modo:
            df_equipos = df_equipos[df_equipos['_ml_valido']]
        elif "Solo Partidos Operables" in filtro_equipos_modo:
            df_equipos = df_equipos[df_equipos['_es_operable']]

        cols_ocultas_eq = ['_ml_valido', '_ou_valido', '_es_operable', '_score_orden']
        df_equipos_export = df_equipos.drop(columns=cols_ocultas_eq, errors='ignore').copy()
        if not df_equipos_export.empty:
            csv_equipos = df_equipos_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar Resumen de Partidos (CSV)",
                data=csv_equipos,
                file_name=f"nba_equipos_{fecha_seleccionada}.csv",
                mime="text/csv",
            )
            st.dataframe(df_equipos_export, use_container_width=True, hide_index=True)
        else:
            st.info("Ningún encuentro cumple el filtro seleccionado en el sidebar.")
        st.markdown("---")

    if resumen_jugadores:
        st.subheader("⭐ Resumen Ejecutivo: Mejores Mercados de Jugadores (Ordenados por Calidad)")
        st.markdown(
            "Ordenado automáticamente de **mayor a menor calidad estadística** combinando **Edge %**, **Hit Rate L10** y **Estabilidad (CV%)**."
        )
       
        df_jugadores_res = pd.DataFrame(resumen_jugadores)
        df_jugadores_res = df_jugadores_res.sort_values(by='_score_calidad', ascending=False)

        if aplicar_filtros_calidad:
            df_jugadores_res = df_jugadores_res[
                (df_jugadores_res['_edge_num'] >= min_edge_filtro) &
                (df_jugadores_res['_hr_num'] >= min_hit_rate_filtro)
            ]
            if excluir_volatiles:
                df_jugadores_res = df_jugadores_res[~df_jugadores_res['_estab_txt'].str.contains("Volátil", na=False)]

        cols_ocultas = ['_edge_num', '_hr_num', '_estab_txt', '_score_calidad']
        df_jugadores_export = df_jugadores_res.drop(columns=cols_ocultas, errors='ignore').copy()

        if not df_jugadores_export.empty:
            csv_jugadores = df_jugadores_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar Resumen de Jugadores (CSV)",
                data=csv_jugadores,
                file_name=f"nba_jugadores_{fecha_seleccionada}.csv",
                mime="text/csv",
            )
            st.dataframe(df_jugadores_export, use_container_width=True, hide_index=True)
        else:
            st.info("Ningún jugador cumple los filtros estrictos de calidad seleccionados en el sidebar. Prueba bajar el Edge Mínimo o desactivar el filtro.")
        st.markdown("---")

    # --- ANÁLISIS DETALLADO DEL JUGADOR ---
    if 'jugador_seleccionado' in st.session_state and st.session_state['jugador_seleccionado'] in nombres_disponibles:
        jugador_activo = st.session_state['jugador_seleccionado']
    elif jugador_seleccionado != "-- Selecciona un jugador --":
        jugador_activo = jugador_seleccionado
    else:
        jugador_activo = None

    if jugador_activo:
        info_sel = catalogo_jugadores[jugador_activo]
        id_seleccionado = str(info_sel['id'])
        equipo_jugador = info_sel['team_name']
        team_id_jugador = str(info_sel['team_id'])

        evento_jugador = next(
            (p for p in cartelera_fecha if team_id_jugador in (str(p["id_t1"]), str(p["id_t2"]))),
            None
        )
        datos_vivo_ev = obtener_datos_en_vivo_espn(evento_jugador["event_id"]) if evento_jugador else None

        ajustes_eq_jugador, bajas_eq_jugador = obtener_ajustes_equipo_cached(
            team_id_jugador,
            evento_jugador.get('bajas_evento_ids') if evento_jugador else None,
            evento_jugador.get('bajas_evento_nombres') if evento_jugador else None,
            datos_vivo_ev
        )
        ajuste_indiv_capa5 = ajustes_eq_jugador.get(jugador_activo, {})

        if ajuste_indiv_capa5.get('esta_out', False):
            st.error(f"🚫 **{jugador_activo}** está detectado actualmente como **OUT / LESIONADO / INACTIVO** ({info_sel.get('status')}). Desmárcalo en el sidebar ('Forzar Jugadores como IN') si deseas proyectar su línea.")

        with st.spinner(f"Analizando estadísticas en tiempo real para {jugador_activo}..."):
            df_jugador_full = consultar_gamelog_jugador(id_seleccionado)

        if df_jugador_full is not None and not df_jugador_full.empty:
            df_jugador_previo = obtener_historial_previo_sin_leakage(df_jugador_full, fecha_busqueda_str)

            lineas_auto_j = calcular_lineas_automaticas_jugador(df_jugador_previo)
            linea_pts_eval = linea_pts_manual if sobreescribir_lineas else lineas_auto_j['pts']
            linea_reb_eval = linea_reb_manual if sobreescribir_lineas else lineas_auto_j['reb']
            linea_ast_eval = linea_ast_manual if sobreescribir_lineas else lineas_auto_j['ast']
            linea_pra_eval = round(linea_pts_eval + linea_reb_eval + linea_ast_eval - 0.5, 1) if sobreescribir_lineas else lineas_auto_j['pra']
            origen_lineas_j = "Manual" if sobreescribir_lineas else "Auto-Calculada"

            st.subheader(f"📊 Historial y Rendimiento Real: {jugador_activo} ({equipo_jugador})")
            st.dataframe(df_jugador_full.head(10), use_container_width=True)

            if ajuste_indiv_capa5.get('motivos') and not ajuste_indiv_capa5.get('esta_out', False):
                st.info(
                    f"🔄 **Ajuste de Rotación y Usage Bump (Capa 5):** "
                    f"Minutos Base: **{ajuste_indiv_capa5['min_base']} min** → Ajustados: **{ajuste_indiv_capa5['min_ajustados']} min** | "
                    f"Multiplicador de Uso: **x{ajuste_indiv_capa5['factor_usage_bump']:.2f}** | "
                    f"Detalle: {'; '.join(ajuste_indiv_capa5['motivos'])}"
                )

            posesiones_ctx = None
            def_rtg_rival_ctx = None
            rival_abrev_ctx = "Liga (Neutro)"
            en_b2b_jugador = (team_id_jugador in equipos_b2b_hoy) if aplicar_b2b_y_paliza else False
            factor_paliza_jugador = 1.0
            riesgo_paliza_txt = "BAJO"

            if evento_jugador:
                es_t1 = str(evento_jugador["id_t1"]) == team_id_jugador
                id_rival = str(evento_jugador["id_t2"] if es_t1 else evento_jugador["id_t1"])
                rival_abrev_ctx = evento_jugador["abrev_t2"] if es_t1 else evento_jugador["abrev_t1"]

                t1_b2b_ev = (str(evento_jugador["id_t1"]) in equipos_b2b_hoy) if aplicar_b2b_y_paliza else False
                t2_b2b_ev = (str(evento_jugador["id_t2"]) in equipos_b2b_hoy) if aplicar_b2b_y_paliza else False

                ctx_j = calcular_contexto_colectivo_capa1(
                    str(evento_jugador["id_t1"]),
                    str(evento_jugador["id_t2"]),
                    metricas_equipos,
                    promedios_liga,
                    linea_mercado_ou=evento_jugador.get('linea_mercado_ou'),
                    spread_mercado=evento_jugador.get('spread_mercado'),
                    t1_en_b2b=t1_b2b_ev,
                    t2_en_b2b=t2_b2b_ev
                )
                posesiones_ctx = ctx_j['posesiones_esperadas']
                def_rtg_rival_ctx = metricas_equipos.get(id_rival, {}).get('def_rtg', promedios_liga['def_liga'])
                factor_paliza_jugador = ctx_j['factor_recorte_paliza'] if aplicar_b2b_y_paliza else 1.0
                riesgo_paliza_txt = ctx_j['riesgo_paliza']

            st_pond_j = calcular_estadisticas_ponderadas_jugador(df_jugador_previo)
            min_auto_sin_bajas = st_pond_j['min']
            min_auto_con_bajas = ajuste_indiv_capa5.get('min_ajustados', min_auto_sin_bajas) if not ajuste_indiv_capa5.get('esta_out', False) else 0.0
            min_auto_con_fatiga, notas_fatiga_j = aplicar_ajustes_fatiga_y_paliza_minutos(
                min_auto_con_bajas, en_b2b=en_b2b_jugador, factor_recorte_paliza=factor_paliza_jugador
            )
            minutos_proyectados_final = min_auto_con_fatiga if usar_min_automaticos else min_proj_manual
            factor_bump_final = ajuste_indiv_capa5.get('factor_usage_bump', 1.0) if not ajuste_indiv_capa5.get('esta_out', False) else 1.0

            if notas_fatiga_j and not ajuste_indiv_capa5.get('esta_out', False):
                st.warning(f"⚡ **Ajuste Automático de Carga / Descanso Prematuro:** {', '.join(notas_fatiga_j)} → Minutos finales proyectados: **{minutos_proyectados_final:.1f} min**")

            proy_detallada = calcular_proyeccion_jugador_capa2_3(
                df_jugador=df_jugador_previo,
                minutos_proyectados=minutos_proyectados_final,
                linea_pts=linea_pts_eval,
                linea_reb=linea_reb_eval,
                linea_ast=linea_ast_eval,
                linea_pra=linea_pra_eval,
                posesiones_partido=posesiones_ctx,
                def_rtg_rival=def_rtg_rival_ctx,
                promedios_liga=promedios_liga,
                factor_usage_bump=factor_bump_final,
                min_base_sin_bajas=min_auto_sin_bajas,
                en_b2b=en_b2b_jugador
            )

            # --- PANEL DINÁMICO EN VIVO (CAPA 4 - POLLING CADA 30s) ---
            if evento_jugador and modo_vivo and proy_detallada and not ajuste_indiv_capa5.get('esta_out', False):
                @st.fragment(run_every=30)
                def panel_en_vivo_capa4():
                    obtener_datos_en_vivo_espn.clear()
                    datos_vivo = obtener_datos_en_vivo_espn(evento_jugador["event_id"])
                    if not datos_vivo:
                        return

                    marc_str = " vs ".join([f"{eq} {pts}" for eq, pts in datos_vivo["marcadores"].items()])

                    if datos_vivo["estado"] == "in":
                        res_vivo = calcular_ajuste_vivo_capa4(
                            datos_vivo=datos_vivo,
                            id_jugador_espn=id_seleccionado,
                            nombre_jugador=jugador_activo,
                            df_historial=df_jugador_previo,
                            minutos_proyectados_totales=minutos_proyectados_final,
                            linea_pts=linea_pts_eval,
                            linea_reb=linea_reb_eval,
                            linea_ast=linea_ast_eval,
                            factor_ritmo=proy_detallada['factor_ritmo'],
                            factor_rival=proy_detallada['factor_rival'],
                            factor_usage_bump=factor_bump_final
                        )
                        st.markdown("---")
                        st.markdown(f"""
                            <div class="card-live">
                                <h3 style="margin:0; color:#93c5fd;">🔴 SEGUIMIENTO EN VIVO — Q{datos_vivo['cuarto']} ({datos_vivo['reloj']}) | Marcador: {marc_str}</h3>
                                <p style="margin:4px 0 0 0; font-size:13px; color:#cbd5e1;">Sincronización automática cada 30 segundos directamente desde Boxscore oficial.</p>
                            </div>
                        """, unsafe_allow_html=True)

                        if res_vivo:
                            for alerta in res_vivo["alertas"]:
                                st.warning(alerta)
                                
                            act = res_vivo["actual"]
                            st.caption(
                                f"📊 **Progreso Actual en Cancha:** **{act['PTS_ACTUAL']} PTS, {act['REB_ACTUAL']} REB, {act['AST_ACTUAL']} AST ({act['PRA_ACTUAL']} PRA)** "
                                f"en {act['MIN_ACTUAL']:.1f} min ({act['FALTAS']} faltas) | "
                                f"Minutos restantes estimados: **{res_vivo['min_restantes_ajustados']} min**"
                            )
                            
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                st.metric(
                                    f"PTS: {act['PTS_ACTUAL']} / {linea_pts_eval}",
                                    f"Proy. Final: {res_vivo['proj_pts']} pts",
                                    f"Over {linea_pts_eval}: {res_vivo['prob_over_pts']}%"
                                )
                                st.progress(min(max(int((act['PTS_ACTUAL'] / max(linea_pts_eval, 0.5)) * 100), 0), 100))
                            with c2:
                                st.metric(
                                    f"REB: {act['REB_ACTUAL']} / {linea_reb_eval}",
                                    f"Proy. Final: {res_vivo['proj_reb']} reb",
                                    f"Over {linea_reb_eval}: {res_vivo['prob_over_reb']}%"
                                )
                                st.progress(min(max(int((act['REB_ACTUAL'] / max(linea_reb_eval, 0.5)) * 100), 0), 100))
                            with c3:
                                st.metric(
                                    f"AST: {act['AST_ACTUAL']} / {linea_ast_eval}",
                                    f"Proy. Final: {res_vivo['proj_ast']} ast",
                                    f"Over {linea_ast_eval}: {res_vivo['prob_over_ast']}%"
                                )
                                st.progress(min(max(int((act['AST_ACTUAL'] / max(linea_ast_eval, 0.5)) * 100), 0), 100))
                        else:
                            st.info(f"⏳ El partido está en vivo ({marc_str}), pero {jugador_activo} aún no registra aparición en el boxscore.")

                    elif datos_vivo["estado"] == "pre":
                        st.info(f"⏳ Partido aún no inicia ({evento_jugador['hora_ecuador']} ECT vs {rival_abrev_ctx}). El progreso en vivo se activará automáticamente en el Q1.")

                panel_en_vivo_capa4()

            if proy_detallada:
                mm_j = proy_detallada['mejor_mercado']
                st.markdown("---")
                st.subheader(f"🎯 Reporte Probabilístico Integral ({origen_lineas_j}): {jugador_activo}")
                st.success(
                    f"🏆 **Mejor Mercado Recomendado:** **{mm_j['direccion']} {mm_j['linea']} {mm_j['mercado']}** "
                    f"(Λ: {mm_j['lambda']} | Prob: {mm_j['prob']}% | **Edge {mm_j['edge']:+.1f}%** | "
                    f"Hit Rate L10: **{int(mm_j['hit_rate'])}%** | Estabilidad: **{mm_j['estabilidad']}**)"
                )
                st.caption(
                    f"⚙️ **Contexto Aplicado:** Minutos Proyectados = **{minutos_proyectados_final:.1f} min** | "
                    f"Rival = **{rival_abrev_ctx}** | Ritmo = **{proy_detallada['factor_ritmo']}x** | "
                    f"Defensa = **{proy_detallada['factor_rival']}x** | "
                    f"B2B = **{'Sí' if en_b2b_jugador else 'No'}** | "
                    f"Riesgo Paliza = **{riesgo_paliza_txt}**"
                )

                col1, col2, col3, col4 = st.columns(4)

                for col_ui, m_key, titulo_m in [
                    (col1, 'PTS', '🏀 PUNTOS'),
                    (col2, 'REB', '📊 REBOTES'),
                    (col3, 'AST', '🎯 ASISTENCIAS'),
                    (col4, 'PRA', '🔥 PRA (P+R+A)')
                ]:
                    with col_ui:
                        m_dat = proy_detallada['mercados_eval'][m_key]
                        is_more = m_dat['direccion'] == "MÁS"
                        extra_glow = " card-high-conf" if (m_dat['prob'] >= UMBRAL_ALTA_CONFIANZA and m_dat['edge'] >= 4.5) else ""
                        css_box = ("card-over" if is_more else "card-under") + extra_glow
                        badge_m = obtener_html_badge_confianza(m_dat['prob'])
                        verdict_m = f"🟢 {m_dat['direccion']} {m_dat['linea']}" if is_more else f"🔴 {m_dat['direccion']} {m_dat['linea']}"
                        p_more_bar = m_dat['prob'] if is_more else (100.0 - m_dat['prob'])

                        st.markdown(f"""
                            <div class="{css_box}">
                                {badge_m}
                                <h4 style="margin:0; color:white;">{titulo_m} (Línea: {m_dat['linea']})</h4>
                                <p style="margin:5px 0 0 0; font-size:18px; font-weight:bold;">Λ Proy: {m_dat['lambda']}</p>
                                <p style="margin:2px 0 0 0; font-size:12px; color:#94a3b8;">Hit Rate L10: {int(m_dat['hit_rate'])}% | {m_dat['estabilidad']}</p>
                                <p style="margin:5px 0 0 0; font-size:17px; font-weight:bold;">{verdict_m}</p>
                                <p style="margin:0; font-size:12px; color:#cbd5e1;">Prob: {m_dat['prob']:.1f}% | Edge: {m_dat['edge']:+.1f}%</p>
                            </div>
                        """, unsafe_allow_html=True)
                        st.progress(min(max(int(p_more_bar), 0), 100))

# ------------------------------------------------------------------------------
# PESTAÑA 2: BUSCADOR WEB DE NOTICIAS Y ESTADO DE JUGADORES
# ------------------------------------------------------------------------------
with tab_buscador:
    st.subheader("🔍 Buscador en Tiempo Real: Estado de Jugadores y Noticias NBA")
    st.markdown(
        "Consulta titulares en tiempo real desde **Google News** (en inglés para reportes inmediatos de *beat writers* / Rotowire o en español) "
        "y revisa el **Monitor Oficial de Lesiones** de la NBA sin salir de la aplicación."
    )

    col_b1, col_b2, col_b3 = st.columns([3, 1.2, 1])
    with col_b1:
        query_usuario = st.text_input(
            "Buscar jugador, equipo o consulta libre:",
            value=st.session_state.get('query_web_nba', 'injury report ruled out tonight'),
            placeholder="Ej: LeBron James injury, jugadores fuera nba hoy, Bam Adebayo..."
        )
    with col_b2:
        idioma_busqueda = st.selectbox("Idioma de fuentes", ["Inglés (Más rápido/NBA)", "Español"], index=0)
    with col_b3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔎 Buscar en la Red", use_container_width=True):
            st.session_state['query_web_nba'] = query_usuario

    st.caption("⚡ **Consultas Rápidas de Última Hora (1 clic):**")
    c_q1, c_q2, c_q3, c_q4 = st.columns(4)
    if c_q1.button("🏥 Jugadores Fuera Hoy", use_container_width=True):
        st.session_state['query_web_nba'] = "injury report ruled out tonight"
        st.rerun()
    if c_q2.button("⏳ Game-Time Decisions", use_container_width=True):
        st.session_state['query_web_nba'] = "game-time decision questionable tonight"
        st.rerun()
    if c_q3.button("📋 Quintetos Titulares", use_container_width=True):
        st.session_state['query_web_nba'] = "starting lineup tonight"
        st.rerun()
    if c_q4.button("🇪🇸 Noticias NBA Hoy", use_container_width=True):
        st.session_state['query_web_nba'] = "jugadores lesionados fuera hoy"
        st.rerun()

    st.markdown("---")
    col_res_web, col_res_espn = st.columns([1.4, 1])

    with col_res_web:
        st.markdown(f"### 🌐 Resultados Web para: *`{query_usuario}`*")
        cod_lang = "en" if "Inglés" in idioma_busqueda else "es"
        noticias_web = buscar_noticias_web_nba(query_usuario, idioma=cod_lang, max_resultados=10)

        if noticias_web:
            for n in noticias_web:
                st.markdown(f"""
                    <div class="card-live" style="padding: 10px 14px; margin-bottom: 8px;">
                        <a href="{n['url']}" target="_blank" style="color: #60a5fa; font-weight: 700; font-size: 15px; text-decoration: none;">
                            {n['titulo']} ↗
                        </a>
                        <div style="font-size: 12px; color: #94a3b8; margin-top: 4px;">
                            📰 <b>{n['fuente']}</b> &nbsp;|&nbsp; 🕒 {n['fecha']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No se encontraron titulares recientes para esa consulta. Prueba con el nombre del jugador en inglés.")

    with col_res_espn:
        st.markdown("### 🏀 Titulares Oficiales NBA (ESPN)")
        noticias_espn = obtener_ultimas_noticias_espn_nba()
        for ne in noticias_espn[:7]:
            with st.expander(f"📰 {ne['titulo']}"):
                st.write(ne['descripcion'])
                st.caption(f"Publicado: {ne['fecha']} UTC")
                st.markdown(f"[Leer nota completa ↗]({ne['url']})")

    st.markdown("---")
    st.subheader("🏥 Monitor Oficial de Lesiones de la Liga (Filtrable)")
    df_lesiones_full = obtener_tabla_completa_lesiones_nba()
    if not df_lesiones_full.empty:
        filtro_lesion_txt = st.text_input("Filtrar tabla oficial por equipo, jugador o estatus (ej. Out, Day-To-Day, Lakers, Heat):", "")
        df_mostrar_les = df_lesiones_full.copy()
        if filtro_lesion_txt.strip():
            mask = df_mostrar_les.apply(lambda row: row.astype(str).str.contains(filtro_lesion_txt, case=False, na=False).any(), axis=1)
            df_mostrar_les = df_mostrar_les[mask]
        st.dataframe(df_mostrar_les, use_container_width=True, hide_index=True)
