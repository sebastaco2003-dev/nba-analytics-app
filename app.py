import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from scipy.stats import poisson, norm

# Configuración inicial de la página web
st.set_page_config(page_title="NBA Analytics & Backtesting Engine", page_icon="🏀", layout="wide")

# Estilos CSS avanzados para diseño profesional, insignias de Alta Confianza y barra bicolor
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

UMBRAL_ALTA_CONFIANZA = 62.0

# ==============================================================================
# 1. EXTRACCIÓN DE DATOS BASE (CARTELERA, BACK-TO-BACK, ROSTERS Y RATINGS)
# ==============================================================================

@st.cache_data(ttl=1800)
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
            if odds_list and isinstance(odds_list, list):
                ou_val = odds_list[0].get('overUnder')
                if ou_val is not None:
                    try:
                        linea_mercado_ou = float(ou_val)
                    except (ValueError, TypeError):
                        linea_mercado_ou = None
           
            if len(competidores) == 2:
                t1 = competidores[0]
                t2 = competidores[1]
               
                score_1 = t1.get('score', '0')
                score_2 = t2.get('score', '0')
               
                partidos.append({
                    'event_id': evento.get('id'),
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
                    'estado': evento.get('status', {}).get('type', {}).get('description', 'Programado'),
                    'estado_codigo': evento.get('status', {}).get('type', {}).get('state', 'pre'),
                    'linea_mercado_ou': linea_mercado_ou
                })
       
        partidos = sorted(partidos, key=lambda x: x['dt_orden'])
    except Exception as e:
        print(f"Error cartelera fecha {fecha_str}: {e}")
       
    return partidos


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


@st.cache_data(ttl=86400)
def obtener_mapa_jugadores_completo():
    url_equipos = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams?limit=50"
    headers = {'User-Agent': 'Mozilla/5.0'}
    mapa_jugadores = {}
   
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
                status_info = atleta.get('status', {}).get('name', 'Active')
                if not status_info:
                    status_info = 'Active'
                pos_abrev = atleta.get('position', {}).get('abbreviation', 'F')
               
                if nombre_completo and id_atleta:
                    mapa_jugadores[nombre_completo] = {
                        'id': id_atleta,
                        'team_id': id_equipo,
                        'team_abbrev': abrev_equipo,
                        'team_name': nombre_equipo,
                        'status': status_info,
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

                    filas.append({
                        'FECHA': fecha_raw,
                        'RIVAL': info_partido.get('opponent', {}).get('abbreviation', ''),
                        'MIN': min_val,
                        'PTS': int(stats.get('points', 0)),
                        'REB': int(stats.get('totalRebounds', 0)),
                        'AST': int(stats.get('assists', 0))
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


def calcular_lineas_automaticas_jugador(df_jugador_previo):
    if df_jugador_previo is None or df_jugador_previo.empty:
        return {'pts': 15.5, 'reb': 5.5, 'ast': 3.5}

    df_v = df_jugador_previo[df_jugador_previo['MIN'] > 0]
    if df_v.empty:
        return {'pts': 15.5, 'reb': 5.5, 'ast': 3.5}

    df_10 = df_v.head(10)
    pts_pond = (0.60 * df_10['PTS'].mean()) + (0.40 * df_v['PTS'].mean())
    reb_pond = (0.60 * df_10['REB'].mean()) + (0.40 * df_v['REB'].mean())
    ast_pond = (0.60 * df_10['AST'].mean()) + (0.40 * df_v['AST'].mean())

    def a_medio_punto(val):
        return max(np.floor(val) + 0.5, 0.5)

    return {
        'pts': round(a_medio_punto(pts_pond), 1),
        'reb': round(a_medio_punto(reb_pond), 1),
        'ast': round(a_medio_punto(ast_pond), 1)
    }

# ==============================================================================
# 2. CAPA 5: MOTOR DE ALINEACIONES, REDISTRIBUCIÓN DE MINUTOS Y USAGE BUMP
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
    incluir_bajas_automaticas_espn=True
):
    roster_equipo = [
        (nom, inf) for nom, inf in catalogo_jugadores.items()
        if str(inf.get('team_id')) == str(team_id)
    ]

    jugadores_activos = []
    jugadores_out = []

    for nom, inf in roster_equipo:
        status_lower = inf.get('status', 'Active').lower()
        es_baja_espn = incluir_bajas_automaticas_espn and any(
            w in status_lower for w in ['out', 'injured', 'suspension', 'inactive']
        )

        if nom in bajas_manuales_out:
            esta_out = True
        elif nom in altas_manuales_in:
            esta_out = False
        else:
            esta_out = es_baja_espn

        df_j = consultar_gamelog_jugador(inf['id'])
        if df_j is not None and not df_j.empty:
            df_prev = obtener_historial_previo_sin_leakage(df_j, fecha_limite_str)
            df_v = df_prev[df_prev['MIN'] > 0].head(10)
            if not df_v.empty:
                min_base = float(df_v['MIN'].mean())
                tasa_pts = df_v['PTS'].sum() / df_v['MIN'].sum()
                tasa_ast = df_v['AST'].sum() / df_v['MIN'].sum()
                usg_est = min(max((tasa_pts * 36.0 + tasa_ast * 14.0), 10.0), 38.0)
                grupo_pos = normalizar_grupo_posicion(inf.get('position', 'F'))

                registro = {
                    'nombre': nom,
                    'id': inf['id'],
                    'pos': inf.get('position', 'F'),
                    'grupo_pos': grupo_pos,
                    'min_base': min_base,
                    'usg_est': round(usg_est, 1),
                    'status_raw': inf.get('status', 'Active')
                }
                if esta_out and min_base >= 12.0:
                    jugadores_out.append(registro)
                elif not esta_out and min_base >= 8.0:
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
            'motivos': ['Jugador marcado como OUT']
        }

    if not jugadores_out or not jugadores_activos:
        return ajustes, jugadores_out

    for baja in jugadores_out:
        min_vacantes = baja['min_base']
        grupo_baja = baja['grupo_pos']
        usg_baja = baja['usg_est']

        misma_pos = sorted(
            [j for j in jugadores_activos if j['grupo_pos'] == grupo_baja],
            key=lambda x: x['min_base'],
            reverse=True
        )
        adyacentes = sorted(
            [j for j in jugadores_activos if j['grupo_pos'] != grupo_baja],
            key=lambda x: x['min_base'],
            reverse=True
        )[:3]

        bolsa_directa = min_vacantes * 0.65
        if misma_pos:
            pesos_dir = [0.70] + [0.30 / max(len(misma_pos) - 1, 1)] * (len(misma_pos) - 1) if len(misma_pos) > 1 else [1.0]
            for idx, cand in enumerate(misma_pos):
                nom_c = cand['nombre']
                ganancia = bolsa_directa * pesos_dir[idx]
                nuevo_min = min(ajustes[nom_c]['min_ajustados'] + ganancia, 39.0)
                real_ganado = nuevo_min - ajustes[nom_c]['min_ajustados']
                ajustes[nom_c]['min_ajustados'] = round(nuevo_min, 1)
                ajustes[nom_c]['extra_min'] = round(ajustes[nom_c]['extra_min'] + real_ganado, 1)
                if real_ganado >= 0.5:
                    ajustes[nom_c]['motivos'].append(f"+{real_ganado:.1f} min por baja de {baja['nombre']}")

        bolsa_adyacente = min_vacantes * 0.35
        if adyacentes:
            reparto_ady = bolsa_adyacente / len(adyacentes)
            for cand in adyacentes:
                nom_c = cand['nombre']
                nuevo_min = min(ajustes[nom_c]['min_ajustados'] + reparto_ady, 39.0)
                real_ganado = nuevo_min - ajustes[nom_c]['min_ajustados']
                ajustes[nom_c]['min_ajustados'] = round(nuevo_min, 1)
                ajustes[nom_c]['extra_min'] = round(ajustes[nom_c]['extra_min'] + real_ganado, 1)

        if usg_baja > 25.0:
            exceso_usg = (usg_baja - 25.0) / 100.0
            principales = sorted(jugadores_activos, key=lambda x: x['usg_est'], reverse=True)[:4]
            for cand in principales:
                nom_c = cand['nombre']
                bump = 0.04 + (exceso_usg * 0.45)
                ajustes[nom_c]['factor_usage_bump'] = round(ajustes[nom_c]['factor_usage_bump'] + bump, 3)
                ajustes[nom_c]['motivos'].append(f"Usage Bump +{bump*100:.1f}% (Baja {baja['nombre']} USG {usg_baja}%)")

    for nom_c, dat in ajustes.items():
        if dat['esta_out']:
            continue
        ratio_min = dat['min_ajustados'] / max(dat['min_base'], 1.0)
        mult_bruto = ratio_min * dat['factor_usage_bump']
        mult_tope = min(mult_bruto, 1.25)
        dat['multiplicador_combinado'] = round(mult_tope, 3)
        if mult_bruto > 1.25:
            dat['motivos'].append("Cap de conservadurismo aplicado (+25% máx)")

    return ajustes, jugadores_out

# ==============================================================================
# 3. CAPAS 1, 2 Y 3: MOTOR PROBABILÍSTICO CON BACK-TO-BACK Y RIESGO DE PALIZA
# ==============================================================================

def calcular_contexto_colectivo_capa1(
    id_t1,
    id_t2,
    metricas_equipos,
    promedios_liga,
    linea_mercado_ou=None,
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
    spread_abs = abs(diferencia_esperada)

    if spread_abs >= 12.5:
        riesgo_paliza = "ALTO"
        factor_recorte_paliza = 0.92
    elif spread_abs >= 9.0:
        riesgo_paliza = "MODERADO"
        factor_recorte_paliza = 0.96
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
        'spread_proyectado': round(spread_abs, 1),
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

    if en_b2b and minutos_base >= 26.0:
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

    min_totales = df_valid['MIN'].sum()
    if min_totales == 0:
        return None

    tasa_pts_min = df_valid['PTS'].sum() / min_totales
    tasa_reb_min = df_valid['REB'].sum() / min_totales
    tasa_ast_min = df_valid['AST'].sum() / min_totales
    min_promedio_hist = df_valid['MIN'].mean()

    pace_liga = promedios_liga.get('pace_liga', 99.5) if promedios_liga else 99.5
    def_liga = promedios_liga.get('def_liga', 114.5) if promedios_liga else 114.5

    factor_ritmo = (posesiones_partido / pace_liga) if posesiones_partido else 1.0
    factor_rival = (def_rtg_rival / def_liga) if def_rtg_rival else 1.0
    factor_fatiga_eficiencia = 0.975 if en_b2b else 1.0

    min_ref = min_base_sin_bajas if min_base_sin_bajas else min_promedio_hist
    lambda_pts_sin_bajas = tasa_pts_min * min_ref * factor_ritmo * factor_rival
    lambda_reb_sin_bajas = tasa_reb_min * min_ref * factor_ritmo
    lambda_ast_sin_bajas = tasa_ast_min * min_ref * factor_ritmo * (1.0 + (factor_rival - 1.0) * 0.5)

    lambda_pts_bruto = tasa_pts_min * minutos_proyectados * factor_ritmo * factor_rival * factor_usage_bump * factor_fatiga_eficiencia
    lambda_reb_bruto = tasa_reb_min * minutos_proyectados * factor_ritmo * (1.0 + (factor_usage_bump - 1.0) * 0.4)
    lambda_ast_bruto = tasa_ast_min * minutos_proyectados * factor_ritmo * (1.0 + (factor_rival - 1.0) * 0.5) * factor_usage_bump * factor_fatiga_eficiencia

    lambda_pts = min(lambda_pts_bruto, lambda_pts_sin_bajas * 1.25) if factor_usage_bump > 1.0 or minutos_proyectados > min_ref else lambda_pts_bruto
    lambda_reb = min(lambda_reb_bruto, lambda_reb_sin_bajas * 1.25) if factor_usage_bump > 1.0 or minutos_proyectados > min_ref else lambda_reb_bruto
    lambda_ast = min(lambda_ast_bruto, lambda_ast_sin_bajas * 1.25) if factor_usage_bump > 1.0 or minutos_proyectados > min_ref else lambda_ast_bruto

    std_pts = df_valid['PTS'].std() if len(df_valid) > 1 else 4.5
    if pd.isna(std_pts) or std_pts == 0:
        std_pts = 4.5

    z_score = (linea_pts - lambda_pts) / std_pts
    prob_more_pts = (1.0 - norm.cdf(z_score)) * 100.0
    prob_less_pts = 100.0 - prob_more_pts

    limite_poisson_reb = int(np.floor(linea_reb))
    prob_less_reb = poisson.cdf(limite_poisson_reb, max(lambda_reb, 1e-6)) * 100.0
    prob_more_reb = 100.0 - prob_less_reb

    limite_poisson_ast = int(np.floor(linea_ast))
    prob_less_ast = poisson.cdf(limite_poisson_ast, max(lambda_ast, 1e-6)) * 100.0
    prob_more_ast = 100.0 - prob_less_ast

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
    }

# ==============================================================================
# 4. CAPA 4: EXTRACCIÓN Y MOTOR PROBABILÍSTICO EN VIVO (POLLING 30s)
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


def obtener_datos_en_vivo_espn(event_id):
    if not event_id:
        return None
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={event_id}"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        res = requests.get(url, headers=headers, timeout=8).json()
        header_comp = res.get("header", {}).get("competitions", [{}])[0]
        status_obj = header_comp.get("status", {})
        
        estado_tipo = status_obj.get("type", {}).get("state", "pre")
        cuarto = int(status_obj.get("period", 1))
        reloj_str = status_obj.get("displayClock", "12:00")
        min_reloj = parsear_reloj_a_minutos(reloj_str)
        
        competidores = header_comp.get("competitors", [])
        marcadores = {}
        for comp in competidores:
            abrev = comp.get("team", {}).get("abbreviation", "")
            marcadores[abrev] = int(comp.get("score", 0))
            
        if estado_tipo == "pre":
            min_restantes_partido = 48.0
        elif estado_tipo == "post":
            min_restantes_partido = 0.0
        else:
            if cuarto <= 4:
                min_restantes_partido = ((4 - cuarto) * 12.0) + min_reloj
            else:
                min_restantes_partido = min_reloj

        stats_jugadores_vivo = {}
        boxscore = res.get("boxscore", {}).get("players", [])
        for equipo_box in boxscore:
            abrev_eq = equipo_box.get("team", {}).get("abbreviation", "")
            estadisticas = equipo_box.get("statistics", [])
            if not estadisticas:
                continue
            
            nombres_cols = estadisticas[0].get("names", [])
            atletas = estadisticas[0].get("athletes", [])
            
            for atl in atletas:
                info_atl = atl.get("athlete", {})
                nombre = info_atl.get("displayName", "")
                id_atl = info_atl.get("id", "")
                valores = atl.get("stats", [])
                
                if nombre and valores and len(valores) == len(nombres_cols):
                    mapa_s = dict(zip(nombres_cols, valores))
                    try:
                        min_jugados = float(mapa_s.get("MIN", 0))
                    except ValueError:
                        min_jugados = 0.0
                        
                    stats_jugadores_vivo[nombre] = {
                        "id": id_atl,
                        "equipo": abrev_eq,
                        "MIN_ACTUAL": min_jugados,
                        "PTS_ACTUAL": int(mapa_s.get("PTS", 0)) if str(mapa_s.get("PTS", "0")).isdigit() else 0,
                        "REB_ACTUAL": int(mapa_s.get("REB", 0)) if str(mapa_s.get("REB", "0")).isdigit() else 0,
                        "AST_ACTUAL": int(mapa_s.get("AST", 0)) if str(mapa_s.get("AST", "0")).isdigit() else 0,
                        "FALTAS": int(mapa_s.get("PF", 0)) if str(mapa_s.get("PF", "0")).isdigit() else 0,
                    }
                    
        return {
            "estado": estado_tipo,
            "cuarto": cuarto,
            "reloj": reloj_str,
            "min_reloj": min_reloj,
            "min_restantes_partido": min_restantes_partido,
            "marcadores": marcadores,
            "jugadores": stats_jugadores_vivo
        }
    except Exception as e:
        print(f"Error live summary: {e}")
        return None


def calcular_ajuste_vivo_capa4(
    datos_vivo,
    nombre_jugador,
    df_historial,
    minutos_proyectados_totales=34.0,
    linea_pts=22.5,
    linea_reb=6.5,
    linea_ast=4.5,
    factor_ritmo=1.0,
    factor_rival=1.0,
    factor_usage_bump=1.0,
    penalizacion_faltas=0.20,
    penalizacion_blowout=0.85
):
    if not datos_vivo or nombre_jugador not in datos_vivo["jugadores"]:
        return None
        
    j_vivo = datos_vivo["jugadores"][nombre_jugador]
    cuarto = datos_vivo["cuarto"]
    min_reloj = datos_vivo["min_reloj"]
    min_restantes_partido = datos_vivo["min_restantes_partido"]
    
    df_valid = df_historial[df_historial["MIN"] > 0].head(10)
    min_hist_total = df_valid["MIN"].sum()
    if min_hist_total == 0:
        return None
        
    tasa_pts = (df_valid["PTS"].sum() / min_hist_total) * factor_ritmo * factor_rival * factor_usage_bump
    tasa_reb = (df_valid["REB"].sum() / min_hist_total) * factor_ritmo
    tasa_ast = (df_valid["AST"].sum() / min_hist_total) * factor_ritmo * factor_usage_bump
    std_pts_hist = df_valid["PTS"].std() if len(df_valid) > 1 else 4.5
    if pd.isna(std_pts_hist) or std_pts_hist == 0:
        std_pts_hist = 4.5

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
# 5. COMPONENTES VISUALES (BARRA BICOLOR DE VICTORIA E INSIGNIAS DE CONFIANZA)
# ==============================================================================

def renderizar_barra_probabilidad_dual(abrev_t1, prob_t1, abrev_t2, prob_t2):
    """
    Renderiza una barra horizontal dividida en dos colores (Azul Equipo 1 vs Naranja Equipo 2)
    con etiquetas internas proporcionales a su probabilidad de victoria.
    """
    p1 = min(max(float(prob_t1), 5.0), 95.0)
    p2 = round(100.0 - p1, 1)
    p1 = round(p1, 1)

    badge_ganador = ""
    if max(p1, p2) >= UMBRAL_ALTA_CONFIANZA:
        fav = abrev_t1 if p1 >= p2 else abrev_t2
        badge_ganador = f' <span class="badge-high-conf">🔥 ALTA CONFIANZA ({fav} {max(p1, p2):.1f}%)</span>'

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
    if prob_rec >= UMBRAL_ALTA_CONFIANZA:
        return f'<div class="badge-high-conf">🔥 ALTA CONFIANZA ({prob_rec:.1f}%)</div>'
    return f'<div class="badge-normal-conf">⚖️ Confianza Estándar ({prob_rec:.1f}%)</div>'

# ==============================================================================
# 6. INTERFAZ VISUAL PRINCIPAL Y CONTROLES EN BARRA LATERAL (100% AUTOMATIZADO)
# ==============================================================================

st.title("🏀 NBA Real-Time Analytics & Backtesting Engine")
st.markdown("Motor Probabilístico Autónomo (Capas 1 a 5): Líneas Automáticas, Alta Confianza (≥62%), B2B, Palizas y Usage Bump.")

st.sidebar.header("⚙️ Configuración y Auditoría")

if st.sidebar.button("📅 Ir al Día de Hoy"):
    st.session_state['fecha_input'] = datetime.today().date()
    st.rerun()

fecha_defecto = st.session_state.get('fecha_input', datetime.today().date())
fecha_seleccionada = st.sidebar.date_input("Seleccionar Fecha de Cartelera", value=fecha_defecto, key='fecha_input')
fecha_str_api = fecha_seleccionada.strftime("%Y%m%d")
fecha_busqueda_str = fecha_seleccionada.strftime("%Y-%m-%d")

with st.spinner(f"Sincronizando cartelera, fatiga Back-to-Back, rosters y ratings para el {fecha_seleccionada}..."):
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

    if any(word in status_lower for word in ['active', 'available', 'healthy', 'disponible']):
        st.sidebar.markdown(f'<div class="status-active">✅ <b>Estatus:</b> {status_actual} (Disponible)</div>', unsafe_allow_html=True)
    elif any(word in status_lower for word in ['questionable', 'duda', 'day-to-day', 'probable']):
        st.sidebar.markdown(f'<div class="status-questionable">⚠️ <b>Estatus:</b> {status_actual} (En Duda)</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown(f'<div class="status-injured">❌ <b>Estatus:</b> {status_actual} (Lesionado / Fuera)</div>', unsafe_allow_html=True)

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
st.sidebar.subheader("🤖 Filtros, Fatiga y Automatización")
solo_alta_confianza = st.sidebar.checkbox("🔥 Filtrar solo Alta Confianza (≥62%) en tablas", value=False)
aplicar_b2b_y_paliza = st.sidebar.checkbox("Ajuste Automático Back-to-Back y Palizas", value=True)
usar_min_automaticos = st.sidebar.checkbox("Usar Minutos Ajustados Automáticos", value=True)
min_proj_manual = st.sidebar.slider("Minutos Manuales (si desactivas auto)", min_value=15.0, max_value=42.0, value=34.0, step=0.5)
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
# 7. CARTELERA ORDENADA POR HORARIO DE ECUADOR (100% AUTOMATIZADA)
# ==============================================================================

st.subheader(f"📅 Cartelera de Partidos (Hora de Ecuador - UTC-5) — Fecha: {fecha_seleccionada}")

resumen_equipos = []
resumen_jugadores = []
cache_ajustes_equipos = {}

def obtener_ajustes_equipo_cached(team_id):
    tid = str(team_id)
    if tid not in cache_ajustes_equipos:
        cache_ajustes_equipos[tid] = calcular_ajustes_rotacion_capa5(
            team_id=tid,
            catalogo_jugadores=catalogo_jugadores,
            bajas_manuales_out=bajas_manuales_out,
            altas_manuales_in=altas_manuales_in,
            fecha_limite_str=fecha_busqueda_str,
            incluir_bajas_automaticas_espn=incluir_bajas_espn
        )
    return cache_ajustes_equipos[tid]


def hallar_estrella_activa_real(lista_jugadores, ajustes_equipo):
    mejor_candidato = None
    score_maximo = -1
    for nom, inf in lista_jugadores[:22]:
        aj = ajustes_equipo.get(nom, {})
        if aj.get('esta_out', False):
            continue
        df_t = consultar_gamelog_jugador(inf['id'])
        if df_t is not None and len(df_t) >= 3:
            df_prev = obtener_historial_previo_sin_leakage(df_t, fecha_busqueda_str)
            df_v = df_prev[df_prev['MIN'] > 0].head(10)
            if df_v.empty:
                continue
            min_prom = df_v['MIN'].mean()
            pts_prom = df_v['PTS'].mean()
            reb_prom = df_v['REB'].mean()
            ast_prom = df_v['AST'].mean()
           
            if min_prom >= 22.0:
                score_jerarquia = pts_prom * 1.2 + reb_prom + ast_prom
                if score_jerarquia > score_maximo:
                    score_maximo = score_jerarquia
                    mejor_candidato = (nom, pts_prom, reb_prom, ast_prom, min_prom, inf.get('status', 'Active'), inf['id'], df_t, df_prev)
    if not mejor_candidato and lista_jugadores:
        for nom, inf in lista_jugadores[:8]:
            if ajustes_equipo.get(nom, {}).get('esta_out', False):
                continue
            df_t = consultar_gamelog_jugador(inf['id'])
            if df_t is not None and not df_t.empty:
                df_prev = obtener_historial_previo_sin_leakage(df_t, fecha_busqueda_str)
                if df_prev['MIN'].mean() > 10.0:
                    return (nom, df_prev['PTS'].mean(), df_prev['REB'].mean(), df_prev['AST'].mean(), df_prev['MIN'].mean(), inf.get('status', 'Active'), inf['id'], df_t, df_prev)
    return mejor_candidato


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

            t1_b2b = (str(t1_id) in equipos_b2b_hoy) if aplicar_b2b_y_paliza else False
            t2_b2b = (str(t2_id) in equipos_b2b_hoy) if aplicar_b2b_y_paliza else False

            ctx_partido = calcular_contexto_colectivo_capa1(
                t1_id, t2_id, metricas_equipos, promedios_liga,
                linea_mercado_ou=partido.get('linea_mercado_ou'),
                t1_en_b2b=t1_b2b,
                t2_en_b2b=t2_b2b
            )
            linea_ou_partido = ctx_partido['linea_evaluada']
            origen_ou_partido = ctx_partido['origen_linea']
            factor_paliza_partido = ctx_partido['factor_recorte_paliza'] if aplicar_b2b_y_paliza else 1.0

            ajustes_t1, bajas_t1 = obtener_ajustes_equipo_cached(t1_id)
            ajustes_t2, bajas_t2 = obtener_ajustes_equipo_cached(t2_id)

            badges_ctx = []
            if t1_b2b:
                badges_ctx.append(f"🔋 {partido['abrev_t1']} en B2B")
            if t2_b2b:
                badges_ctx.append(f"🔋 {partido['abrev_t2']} en B2B")
            if ctx_partido['riesgo_paliza'] != "BAJO" and aplicar_b2b_y_paliza:
                badges_ctx.append(f"🚨 Riesgo Paliza {ctx_partido['riesgo_paliza']} (Spread ~{ctx_partido['spread_proyectado']} pts)")
            str_badges = f" | {' · '.join(badges_ctx)}" if badges_ctx else ""

            st.markdown(f"### ⏰ [{partido['hora_ecuador']} ECT] 🏟️ {partido['encuentro']} — *{partido['detalle']}* | Estado: `{partido['estado']}`")
            st.caption(
                f"⚡ **Ritmo Esperado:** {ctx_partido['posesiones_esperadas']} pos | "
                f"**{partido['abrev_t1']}:** Λ = {ctx_partido['lambda_t1']} pts | "
                f"**{partido['abrev_t2']}:** Λ = {ctx_partido['lambda_t2']} pts{str_badges}"
            )

            if bajas_t1 or bajas_t2:
                txt_b1 = f"**{partido['abrev_t1']} OUT:** {', '.join([b['nombre'] for b in bajas_t1])}" if bajas_t1 else ""
                txt_b2 = f"**{partido['abrev_t2']} OUT:** {', '.join([b['nombre'] for b in bajas_t2])}" if bajas_t2 else ""
                separador = " | " if (txt_b1 and txt_b2) else ""
                st.warning(f"🏥 **Bajas detectadas (Capa 5 activa):** {txt_b1}{separador}{txt_b2}")
           
            jugadores_t1 = [(nom, inf) for nom, inf in catalogo_jugadores.items() if str(inf.get('team_id')) == t1_id]
            jugadores_t2 = [(nom, inf) for nom, inf in catalogo_jugadores.items() if str(inf.get('team_id')) == t2_id]
           
            col_p1, col_p2 = st.columns(2)
            partido_finalizado = partido['estado_codigo'] == 'post' or partido['estado'].lower() in ['finalized', 'final', 'closed']
           
            # Procesar Estrella Activa 1
            estrella_1 = hallar_estrella_activa_real(jugadores_t1, ajustes_t1)
            with col_p1:
                if estrella_1:
                    nom_1, pts_1, reb_1, ast_1, min_1, stat_1, id_1, df_j1_full, df_j1_prev = estrella_1
                    lineas_auto_1 = calcular_lineas_automaticas_jugador(df_j1_prev)

                    aj_1 = ajustes_t1.get(nom_1, {})
                    min_aj_capa5_1 = aj_1.get('min_ajustados', min_1)
                    min_aj_ctx_1, notas_min_1 = aplicar_ajustes_fatiga_y_paliza_minutos(
                        min_aj_capa5_1, en_b2b=t1_b2b, factor_recorte_paliza=factor_paliza_partido
                    )
                    bump_1 = aj_1.get('factor_usage_bump', 1.0)
                    min_eval_1 = min_aj_ctx_1 if usar_min_automaticos else min_proj_manual

                    proy_1 = calcular_proyeccion_jugador_capa2_3(
                        df_jugador=df_j1_prev,
                        minutos_proyectados=min_eval_1,
                        linea_pts=lineas_auto_1['pts'],
                        linea_reb=lineas_auto_1['reb'],
                        linea_ast=lineas_auto_1['ast'],
                        posesiones_partido=ctx_partido['posesiones_esperadas'],
                        def_rtg_rival=ctx_partido['eq2_stats']['def_rtg'],
                        promedios_liga=promedios_liga,
                        factor_usage_bump=bump_1,
                        min_base_sin_bajas=min_1,
                        en_b2b=t1_b2b
                    )
                    lambda_p1 = proy_1['lambda_pts'] if proy_1 else pts_1
                    prob_p1 = proy_1['prob_more_pts'] if proy_1 else 50.0
                    rec_p1 = "MÁS" if prob_p1 > 50 else "MENOS"
                    prob_rec_1 = max(prob_p1, 100.0 - prob_p1)
                    es_alta_conf_1 = prob_rec_1 >= UMBRAL_ALTA_CONFIANZA
                   
                    partido_j1_fecha = df_j1_full[df_j1_full['FECHA'] == fecha_busqueda_str]
                    res_j1 = "Pendiente"
                    pts_real_str_1 = "-"
                   
                    if partido_finalizado:
                        if not partido_j1_fecha.empty and partido_j1_fecha.iloc[0]['MIN'] > 0:
                            p_real_j1 = partido_j1_fecha.iloc[0]['PTS']
                            pts_real_str_1 = f"{p_real_j1} pts"
                            res_j1 = evaluar_resultado_linea(p_real_j1, lineas_auto_1['pts'], rec_p1)
                        else:
                            res_j1 = "No jugó / Lesionado"

                    resumen_jugadores.append({
                        'Nivel': '🔥 ALTA CONFIANZA' if es_alta_conf_1 else 'Estándar',
                        'Jugador': nom_1,
                        'Equipo': partido['abrev_t1'],
                        'Contexto': ', '.join(notas_min_1) if notas_min_1 else 'Normal',
                        'Min Proy.': round(min_eval_1, 1),
                        'Λ Pts': lambda_p1,
                        'Línea Auto Pts': lineas_auto_1['pts'],
                        'Predicción': f"{rec_p1} {lineas_auto_1['pts']}",
                        'Prob_Num': round(prob_rec_1, 1),
                        'Prob.': f"{prob_rec_1:.1f}%",
                        'Real': pts_real_str_1,
                        'Resultado': res_j1
                    })

                    etiqueta_extra_1 = f" | {', '.join(notas_min_1)}" if notas_min_1 else ""
                    sello_exp_1 = " 🔥 ALTA CONFIANZA" if es_alta_conf_1 else ""
                    with st.expander(f"⭐ {partido['abrev_t1']} - {nom_1} (Λ: {lambda_p1:.1f} vs Línea {lineas_auto_1['pts']}{etiqueta_extra_1}){sello_exp_1}"):
                        st.markdown(obtener_html_badge_confianza(prob_rec_1), unsafe_allow_html=True)
                        st.markdown(f"**Estatus:** {stat_1} | **Minutos:** {min_1:.1f} → **{min_eval_1:.1f} min**")
                        st.metric(
                            f"Proyección vs Líneas Auto ({lineas_auto_1['pts']}P / {lineas_auto_1['reb']}R / {lineas_auto_1['ast']}A)",
                            f"{lambda_p1:.1f} PTS | {proy_1['lambda_reb']:.1f} REB | {proy_1['lambda_ast']:.1f} AST"
                        )
                        if st.button(f"Analizar a {nom_1}", key=f"btn_{id_1}_{partido['event_id']}"):
                            st.session_state['jugador_seleccionado'] = nom_1
                            st.rerun()
                else:
                    st.info(f"Sin datos activos para {partido['abrev_t1']}")

            # Procesar Estrella Activa 2
            estrella_2 = hallar_estrella_activa_real(jugadores_t2, ajustes_t2)
            with col_p2:
                if estrella_2:
                    nom_2, pts_2, reb_2, ast_2, min_2, stat_2, id_2, df_j2_full, df_j2_prev = estrella_2
                    lineas_auto_2 = calcular_lineas_automaticas_jugador(df_j2_prev)

                    aj_2 = ajustes_t2.get(nom_2, {})
                    min_aj_capa5_2 = aj_2.get('min_ajustados', min_2)
                    min_aj_ctx_2, notas_min_2 = aplicar_ajustes_fatiga_y_paliza_minutos(
                        min_aj_capa5_2, en_b2b=t2_b2b, factor_recorte_paliza=factor_paliza_partido
                    )
                    bump_2 = aj_2.get('factor_usage_bump', 1.0)
                    min_eval_2 = min_aj_ctx_2 if usar_min_automaticos else min_proj_manual

                    proy_2 = calcular_proyeccion_jugador_capa2_3(
                        df_jugador=df_j2_prev,
                        minutos_proyectados=min_eval_2,
                        linea_pts=lineas_auto_2['pts'],
                        linea_reb=lineas_auto_2['reb'],
                        linea_ast=lineas_auto_2['ast'],
                        posesiones_partido=ctx_partido['posesiones_esperadas'],
                        def_rtg_rival=ctx_partido['eq1_stats']['def_rtg'],
                        promedios_liga=promedios_liga,
                        factor_usage_bump=bump_2,
                        min_base_sin_bajas=min_2,
                        en_b2b=t2_b2b
                    )
                    lambda_p2 = proy_2['lambda_pts'] if proy_2 else pts_2
                    prob_p2 = proy_2['prob_more_pts'] if proy_2 else 50.0
                    rec_p2 = "MÁS" if prob_p2 > 50 else "MENOS"
                    prob_rec_2 = max(prob_p2, 100.0 - prob_p2)
                    es_alta_conf_2 = prob_rec_2 >= UMBRAL_ALTA_CONFIANZA
                   
                    partido_j2_fecha = df_j2_full[df_j2_full['FECHA'] == fecha_busqueda_str]
                    res_j2 = "Pendiente"
                    pts_real_str_2 = "-"
                   
                    if partido_finalizado:
                        if not partido_j2_fecha.empty and partido_j2_fecha.iloc[0]['MIN'] > 0:
                            p_real_j2 = partido_j2_fecha.iloc[0]['PTS']
                            pts_real_str_2 = f"{p_real_j2} pts"
                            res_j2 = evaluar_resultado_linea(p_real_j2, lineas_auto_2['pts'], rec_p2)
                        else:
                            res_j2 = "No jugó / Lesionado"

                    resumen_jugadores.append({
                        'Nivel': '🔥 ALTA CONFIANZA' if es_alta_conf_2 else 'Estándar',
                        'Jugador': nom_2,
                        'Equipo': partido['abrev_t2'],
                        'Contexto': ', '.join(notas_min_2) if notas_min_2 else 'Normal',
                        'Min Proy.': round(min_eval_2, 1),
                        'Λ Pts': lambda_p2,
                        'Línea Auto Pts': lineas_auto_2['pts'],
                        'Predicción': f"{rec_p2} {lineas_auto_2['pts']}",
                        'Prob_Num': round(prob_rec_2, 1),
                        'Prob.': f"{prob_rec_2:.1f}%",
                        'Real': pts_real_str_2,
                        'Resultado': res_j2
                    })

                    etiqueta_extra_2 = f" | {', '.join(notas_min_2)}" if notas_min_2 else ""
                    sello_exp_2 = " 🔥 ALTA CONFIANZA" if es_alta_conf_2 else ""
                    with st.expander(f"⭐ {partido['abrev_t2']} - {nom_2} (Λ: {lambda_p2:.1f} vs Línea {lineas_auto_2['pts']}{etiqueta_extra_2}){sello_exp_2}"):
                        st.markdown(obtener_html_badge_confianza(prob_rec_2), unsafe_allow_html=True)
                        st.markdown(f"**Estatus:** {stat_2} | **Minutos:** {min_2:.1f} → **{min_eval_2:.1f} min**")
                        st.metric(
                            f"Proyección vs Líneas Auto ({lineas_auto_2['pts']}P / {lineas_auto_2['reb']}R / {lineas_auto_2['ast']}A)",
                            f"{lambda_p2:.1f} PTS | {proy_2['lambda_reb']:.1f} REB | {proy_2['lambda_ast']:.1f} AST"
                        )
                        if st.button(f"Analizar a {nom_2}", key=f"btn_{id_2}_{partido['event_id']}"):
                            st.session_state['jugador_seleccionado'] = nom_2
                            st.rerun()
                else:
                    st.info(f"Sin datos activos para {partido['abrev_t2']}")
           
            # Resultados colectivos basados en Capa 1 + Línea O/U Automática + B2B
            prob_t1 = ctx_partido['prob_t1']
            prob_t2 = ctx_partido['prob_t2']
            total_proyectado_partido = ctx_partido['total_proyectado']
            is_more_partido = total_proyectado_partido > linea_ou_partido
            verdict_partido_total = "MÁS" if is_more_partido else "MENOS"
            prob_total_rec = ctx_partido['prob_over_total'] if is_more_partido else (100.0 - ctx_partido['prob_over_total'])
            ganador_recomendado = partido['abrev_t1'] if prob_t1 >= prob_t2 else partido['abrev_t2']
            max_prob_val = max(prob_t1, prob_t2)
            es_alta_conf_eq = max_prob_val >= UMBRAL_ALTA_CONFIANZA or prob_total_rec >= UMBRAL_ALTA_CONFIANZA

            if partido_finalizado:
                s1 = partido['score_t1']
                s2 = partido['score_t2']
                total_real = s1 + s2
                ganador_real = partido['abrev_t1'] if s1 > s2 else partido['abrev_t2']
               
                cumplio_total = evaluar_resultado_linea(total_real, linea_ou_partido, verdict_partido_total)
                acerto_ganador = "🟢 Acierto" if ganador_recomendado == ganador_real else "🔴 Fallo"
               
                resumen_equipos.append({
                    'Nivel': '🔥 ALTA CONFIANZA' if es_alta_conf_eq else 'Estándar',
                    'Hora (ECT)': partido['hora_ecuador'],
                    'Encuentro': partido['encuentro'],
                    'Riesgo Paliza': ctx_partido['riesgo_paliza'],
                    'Ganador Sugerido': ganador_recomendado,
                    'Prob_Max_Num': round(max(max_prob_val, prob_total_rec), 1),
                    'Prob. Victoria': max_prob_val,
                    'Resultado Ganador': acerto_ganador,
                    'Línea O/U': f"{linea_ou_partido} ({origen_ou_partido})",
                    'Pts Proyectados': total_proyectado_partido,
                    'Predicción Total': f"{verdict_partido_total} ({prob_total_rec:.1f}%)",
                    'Total Real': total_real,
                    'Resultado Total': cumplio_total,
                    'Estado': 'Finalizado'
                })

                st.info(f"📊 **Resultado Final Real:** {partido['abrev_t1']} **{s1} - {s2}** {partido['abrev_t2']} (Total: {total_real} pts)")
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.markdown(f"**Ganador Recomendado:** {ganador_recomendado} | **Real:** {ganador_real} → {acerto_ganador}")
                with col_res2:
                    st.markdown(
                        f"**Total Partido (Línea {linea_ou_partido} - {origen_ou_partido}):** Proyectado {total_proyectado_partido:.1f} "
                        f"(**{verdict_partido_total} {linea_ou_partido}**) | Real: **{total_real}** → {cumplio_total}"
                    )
            else:
                resumen_equipos.append({
                    'Nivel': '🔥 ALTA CONFIANZA' if es_alta_conf_eq else 'Estándar',
                    'Hora (ECT)': partido['hora_ecuador'],
                    'Encuentro': partido['encuentro'],
                    'Riesgo Paliza': ctx_partido['riesgo_paliza'],
                    'Ganador Sugerido': ganador_recomendado,
                    'Prob_Max_Num': round(max(max_prob_val, prob_total_rec), 1),
                    'Prob. Victoria': max_prob_val,
                    'Resultado Ganador': 'Pendiente',
                    'Línea O/U': f"{linea_ou_partido} ({origen_ou_partido})",
                    'Pts Proyectados': total_proyectado_partido,
                    'Predicción Total': f"{verdict_partido_total} ({prob_total_rec:.1f}%)",
                    'Total Real': '-',
                    'Resultado Total': 'Pendiente',
                    'Estado': partido['estado']
                })

                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    sello_ml = " 🔥 ALTA CONFIANZA" if max_prob_val >= UMBRAL_ALTA_CONFIANZA else ""
                    st.success(f"🏆 **Ganador Recomendado (Moneyline):** **{ganador_recomendado}** ({max_prob_val:.1f}%){sello_ml}")
                with col_r2:
                    sello_ou = " 🔥 ALTA CONFIANZA" if prob_total_rec >= UMBRAL_ALTA_CONFIANZA else ""
                    st.warning(
                        f"📈 **Total Partido (Línea {linea_ou_partido} - {origen_ou_partido}):** "
                        f"Proyección {total_proyectado_partido:.1f} pts → **RECOMENDACIÓN: {verdict_partido_total} ({prob_total_rec:.1f}%)**{sello_ou}"
                    )

            # Barra Bicolor de Probabilidad de Victoria (Equipo 1 Azul vs Equipo 2 Naranja)
            renderizar_barra_probabilidad_dual(partido['abrev_t1'], prob_t1, partido['abrev_t2'], prob_t2)
            st.markdown("---")
else:
    st.info(f"No hay partidos registrados en la cartelera oficial para la fecha: {fecha_seleccionada}.")

# ==============================================================================
# 8. RESUMEN EJECUTIVO DE EQUIPOS Y JUGADORES ESTELARES (CON FILTRO ALTA CONFIANZA)
# ==============================================================================

if resumen_equipos:
    st.subheader("📋 Resumen Ejecutivo: Pronósticos de Equipos (Jornada)")
    st.markdown("Tabla interactiva con insignia **🔥 ALTA CONFIANZA (≥62%)**, líneas Over/Under automáticas, fatiga B2B y riesgo de paliza.")
   
    df_equipos = pd.DataFrame(resumen_equipos)
    if solo_alta_confianza:
        df_equipos = df_equipos[df_equipos['Prob_Max_Num'] >= UMBRAL_ALTA_CONFIANZA]

    df_equipos_export = df_equipos.drop(columns=['Prob_Max_Num'], errors='ignore').copy()
    if not df_equipos_export.empty:
        csv_equipos = df_equipos_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar Resumen de Partidos (CSV)",
            data=csv_equipos,
            file_name=f"nba_equipos_{fecha_seleccionada}.csv",
            mime="text/csv",
        )
        df_equipos_export['Prob. Victoria'] = df_equipos_export['Prob. Victoria'].apply(lambda x: f"{x:.1f}%")
        st.dataframe(df_equipos_export, use_container_width=True, hide_index=True)
    else:
        st.info("Ningún encuentro supera el umbral de Alta Confianza (≥62%) en esta jornada.")
    st.markdown("---")

if resumen_jugadores:
    st.subheader("⭐ Resumen Ejecutivo: Estrellas Activas (Líneas Automáticas)")
    st.markdown("Predicciones individuales con insignia **🔥 ALTA CONFIANZA (≥62%)**, líneas auto-calculadas, ajuste B2B, riesgo de paliza y Usage Bump.")
   
    df_jugadores_res = pd.DataFrame(resumen_jugadores)
    if solo_alta_confianza:
        df_jugadores_res = df_jugadores_res[df_jugadores_res['Prob_Num'] >= UMBRAL_ALTA_CONFIANZA]

    df_jugadores_export = df_jugadores_res.drop(columns=['Prob_Num'], errors='ignore').copy()
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
        st.info("Ningún jugador estelar supera el umbral de Alta Confianza (≥62%) en esta jornada.")
    st.markdown("---")

# ==============================================================================
# 9. ANÁLISIS DETALLADO DEL JUGADOR (CAPAS 2, 3, 4 Y 5 + INSIGNIAS ALTA CONFIANZA)
# ==============================================================================

if 'jugador_seleccionado' in st.session_state and st.session_state['jugador_seleccionado'] in nombres_disponibles:
    jugador_activo = st.session_state['jugador_seleccionado']
elif jugador_seleccionado != "-- Selecciona un jugador --":
    jugador_activo = jugador_seleccionado
else:
    jugador_activo = None

if jugador_activo:
    info_sel = catalogo_jugadores[jugador_activo]
    id_seleccionado = info_sel['id']
    equipo_jugador = info_sel['team_name']
    team_id_jugador = str(info_sel['team_id'])

    ajustes_eq_jugador, bajas_eq_jugador = obtener_ajustes_equipo_cached(team_id_jugador)
    ajuste_indiv_capa5 = ajustes_eq_jugador.get(jugador_activo, {})

    if ajuste_indiv_capa5.get('esta_out', False):
        st.error(f"🚫 **{jugador_activo}** está marcado actualmente como **OUT / BAJA**. Desmárcalo en el sidebar ('Forzar Jugadores como IN') si deseas proyectar su línea.")

    with st.spinner(f"Analizando estadísticas en tiempo real para {jugador_activo}..."):
        df_jugador_full = consultar_gamelog_jugador(id_seleccionado)

    if df_jugador_full is not None and not df_jugador_full.empty:
        df_jugador_previo = obtener_historial_previo_sin_leakage(df_jugador_full, fecha_busqueda_str)

        lineas_auto_j = calcular_lineas_automaticas_jugador(df_jugador_previo)
        linea_pts_eval = linea_pts_manual if sobreescribir_lineas else lineas_auto_j['pts']
        linea_reb_eval = linea_reb_manual if sobreescribir_lineas else lineas_auto_j['reb']
        linea_ast_eval = linea_ast_manual if sobreescribir_lineas else lineas_auto_j['ast']
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

        evento_jugador = next(
            (p for p in cartelera_fecha if team_id_jugador in (str(p["id_t1"]), str(p["id_t2"]))),
            None
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
                t1_en_b2b=t1_b2b_ev,
                t2_en_b2b=t2_b2b_ev
            )
            posesiones_ctx = ctx_j['posesiones_esperadas']
            def_rtg_rival_ctx = metricas_equipos.get(id_rival, {}).get('def_rtg', promedios_liga['def_liga'])
            factor_paliza_jugador = ctx_j['factor_recorte_paliza'] if aplicar_b2b_y_paliza else 1.0
            riesgo_paliza_txt = ctx_j['riesgo_paliza']

        df_valid_min = df_jugador_previo[df_jugador_previo['MIN'] > 0].head(10)
        min_auto_sin_bajas = float(df_valid_min['MIN'].mean()) if not df_valid_min.empty else min_proj_manual
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
                datos_vivo = obtener_datos_en_vivo_espn(evento_jugador["event_id"])
                if datos_vivo and datos_vivo["estado"] == "in":
                    res_vivo = calcular_ajuste_vivo_capa4(
                        datos_vivo=datos_vivo,
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
                    if res_vivo:
                        st.markdown("---")
                        st.subheader(f"🔴 EN VIVO (Q{datos_vivo['cuarto']} - {datos_vivo['reloj']}) | Polling 30s")
                        
                        for alerta in res_vivo["alertas"]:
                            st.warning(alerta)
                            
                        act = res_vivo["actual"]
                        st.caption(
                            f"Acumulado actual: **{act['PTS_ACTUAL']} PTS, {act['REB_ACTUAL']} REB, {act['AST_ACTUAL']} AST** "
                            f"en {act['MIN_ACTUAL']:.0f} min ({act['FALTAS']} faltas) | "
                            f"Minutos restantes proyectados: **{res_vivo['min_restantes_ajustados']} min**"
                        )
                        
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Proyección Final PTS", f"{res_vivo['proj_pts']} pts", f"Over {linea_pts_eval}: {res_vivo['prob_over_pts']}%")
                        c2.metric("Proyección Final REB", f"{res_vivo['proj_reb']} reb", f"Over {linea_reb_eval}: {res_vivo['prob_over_reb']}%")
                        c3.metric("Proyección Final AST", f"{res_vivo['proj_ast']} ast", f"Over {linea_ast_eval}: {res_vivo['prob_over_ast']}%")
                elif datos_vivo and datos_vivo["estado"] == "pre":
                    st.info(f"⏳ Partido aún no inicia ({evento_jugador['hora_ecuador']} ECT vs {rival_abrev_ctx}). El motor en vivo (Capa 4) se activará automáticamente al iniciar el Q1.")

            panel_en_vivo_capa4()

        if proy_detallada:
            lambda_pts = proy_detallada['lambda_pts']
            prob_more_pts = proy_detallada['prob_more_pts']
            prob_less_pts = proy_detallada['prob_less_pts']
            rec_pts_hist = "MÁS" if prob_more_pts > 50 else "MENOS"

            lambda_reb = proy_detallada['lambda_reb']
            prob_more_reb = proy_detallada['prob_more_reb']
            prob_less_reb = proy_detallada['prob_less_reb']
            rec_reb_hist = "MÁS" if prob_more_reb > 50 else "MENOS"

            lambda_ast = proy_detallada['lambda_ast']
            prob_more_ast = proy_detallada['prob_more_ast']
            prob_less_ast = proy_detallada['prob_less_ast']
            rec_ast_hist = "MÁS" if prob_more_ast > 50 else "MENOS"

            partido_en_fecha = df_jugador_full[df_jugador_full['FECHA'] == fecha_busqueda_str]

            if not partido_en_fecha.empty and partido_en_fecha.iloc[0]['MIN'] > 0:
                st.markdown("---")
                st.subheader(f"🔍 Auditoría y Verificación de Objetivos ({origen_lineas_j}) para el {fecha_busqueda_str}")
               
                p_real = partido_en_fecha.iloc[0]['PTS']
                r_real = partido_en_fecha.iloc[0]['REB']
                a_real = partido_en_fecha.iloc[0]['AST']
                min_real = partido_en_fecha.iloc[0]['MIN']

                col_b1, col_b2, col_b3 = st.columns(3)
               
                with col_b1:
                    res_txt_p = evaluar_resultado_linea(p_real, linea_pts_eval, rec_pts_hist)
                    css_p = "card-over" if "Acierto" in res_txt_p else ("card-push" if "Push" in res_txt_p else "card-under")
                    st.markdown(f"""
                        <div class="{css_p}">
                            <h4 style="margin:0; color:white;">Puntos: {p_real} (Línea: {linea_pts_eval})</h4>
                            <p style="margin:5px 0 0 0; font-size:15px; font-weight:bold;">{res_txt_p}</p>
                            <p style="margin:2px 0; font-size:13px; color:#cbd5e1;">Se recomendó: <b>{rec_pts_hist} {linea_pts_eval}</b> (Λ: {lambda_pts})</p>
                            <p style="margin:0; font-size:12px; color:#94a3b8;">Minutos jugados: {min_real:.0f} min</p>
                        </div>
                    """, unsafe_allow_html=True)

                with col_b2:
                    res_txt_r = evaluar_resultado_linea(r_real, linea_reb_eval, rec_reb_hist)
                    css_r = "card-over" if "Acierto" in res_txt_r else ("card-push" if "Push" in res_txt_r else "card-under")
                    st.markdown(f"""
                        <div class="{css_r}">
                            <h4 style="margin:0; color:white;">Rebotes: {r_real} (Línea: {linea_reb_eval})</h4>
                            <p style="margin:5px 0 0 0; font-size:15px; font-weight:bold;">{res_txt_r}</p>
                            <p style="margin:2px 0; font-size:13px; color:#cbd5e1;">Se recomendó: <b>{rec_reb_hist} {linea_reb_eval}</b> (Λ: {lambda_reb})</p>
                            <p style="margin:0; font-size:12px; color:#94a3b8;">Minutos jugados: {min_real:.0f} min</p>
                        </div>
                    """, unsafe_allow_html=True)

                with col_b3:
                    res_txt_a = evaluar_resultado_linea(a_real, linea_ast_eval, rec_ast_hist)
                    css_a = "card-over" if "Acierto" in res_txt_a else ("card-push" if "Push" in res_txt_a else "card-under")
                    st.markdown(f"""
                        <div class="{css_a}">
                            <h4 style="margin:0; color:white;">Asistencias: {a_real} (Línea: {linea_ast_eval})</h4>
                            <p style="margin:5px 0 0 0; font-size:15px; font-weight:bold;">{res_txt_a}</p>
                            <p style="margin:2px 0; font-size:13px; color:#cbd5e1;">Se recomendó: <b>{rec_ast_hist} {linea_ast_eval}</b> (Λ: {lambda_ast})</p>
                            <p style="margin:0; font-size:12px; color:#94a3b8;">Minutos jugados: {min_real:.0f} min</p>
                        </div>
                    """, unsafe_allow_html=True)
                st.markdown("---")

            st.markdown("---")
            st.subheader(f"🎯 Reporte Probabilístico Integral ({origen_lineas_j}): {jugador_activo}")
            st.caption(
                f"⚙️ **Contexto Aplicado:** Minutos Proyectados = **{minutos_proyectados_final:.1f} min** | "
                f"Rival = **{rival_abrev_ctx}** | Ritmo = **{proy_detallada['factor_ritmo']}x** | "
                f"Defensa = **{proy_detallada['factor_rival']}x** | "
                f"B2B = **{'Sí' if en_b2b_jugador else 'No'}** | "
                f"Riesgo Paliza = **{riesgo_paliza_txt}**"
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                is_more_pts = prob_more_pts > 50
                prob_rec_p = max(prob_more_pts, prob_less_pts)
                extra_glow_p = " card-high-conf" if prob_rec_p >= UMBRAL_ALTA_CONFIANZA else ""
                css_box_p = ("card-over" if is_more_pts else "card-under") + extra_glow_p
                badge_p = obtener_html_badge_confianza(prob_rec_p)
                verdict_pts = f"🟢 RECOMENDACIÓN: MÁS {linea_pts_eval}" if is_more_pts else f"🔴 RECOMENDACIÓN: MENOS {linea_pts_eval}"
                st.markdown(f"""
                    <div class="{css_box_p}">
                        {badge_p}
                        <h4 style="margin:0; color:white;">🏀 PUNTOS (Línea {origen_lineas_j}: {linea_pts_eval})</h4>
                        <p style="margin:5px 0 0 0; font-size:18px; font-weight:bold;">Λ Proyectado: {lambda_pts:.2f} pts</p>
                        <p style="margin:2px 0 0 0; font-size:12px; color:#94a3b8;">Tasa Base: {proy_detallada['tasa_pts_min']:.2f} pts/min</p>
                        <p style="margin:5px 0 0 0; font-size:18px; font-weight:bold;">{verdict_pts}</p>
                        <p style="margin:0; font-size:12px; color:#cbd5e1;">Más: {prob_more_pts:.1f}% | Menos: {prob_less_pts:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)
                st.progress(min(max(int(prob_more_pts), 0), 100))

            with col2:
                is_more_reb = prob_more_reb > 50
                prob_rec_r = max(prob_more_reb, prob_less_reb)
                extra_glow_r = " card-high-conf" if prob_rec_r >= UMBRAL_ALTA_CONFIANZA else ""
                css_box_r = ("card-over" if is_more_reb else "card-under") + extra_glow_r
                badge_r = obtener_html_badge_confianza(prob_rec_r)
                verdict_reb = f"🟢 RECOMENDACIÓN: MÁS {linea_reb_eval}" if is_more_reb else f"🔴 RECOMENDACIÓN: MENOS {linea_reb_eval}"
                st.markdown(f"""
                    <div class="{css_box_r}">
                        {badge_r}
                        <h4 style="margin:0; color:white;">📊 REBOTES (Línea {origen_lineas_j}: {linea_reb_eval})</h4>
                        <p style="margin:5px 0 0 0; font-size:18px; font-weight:bold;">Λ Proyectado: {lambda_reb:.2f} reb</p>
                        <p style="margin:2px 0 0 0; font-size:12px; color:#94a3b8;">Tasa Base: {proy_detallada['tasa_reb_min']:.2f} reb/min</p>
                        <p style="margin:5px 0 0 0; font-size:18px; font-weight:bold;">{verdict_reb}</p>
                        <p style="margin:0; font-size:12px; color:#cbd5e1;">Más: {prob_more_reb:.1f}% | Menos: {prob_less_reb:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)
                st.progress(min(max(int(prob_more_reb), 0), 100))

            with col3:
                is_more_ast = prob_more_ast > 50
                prob_rec_a = max(prob_more_ast, prob_less_ast)
                extra_glow_a = " card-high-conf" if prob_rec_a >= UMBRAL_ALTA_CONFIANZA else ""
                css_box_a = ("card-over" if is_more_ast else "card-under") + extra_glow_a
                badge_a = obtener_html_badge_confianza(prob_rec_a)
                verdict_ast = f"🟢 RECOMENDACIÓN: MÁS {linea_ast_eval}" if is_more_ast else f"🔴 RECOMENDACIÓN: MENOS {linea_ast_eval}"
                st.markdown(f"""
                    <div class="{css_box_a}">
                        {badge_a}
                        <h4 style="margin:0; color:white;">🎯 ASISTENCIAS (Línea {origen_lineas_j}: {linea_ast_eval})</h4>
                        <p style="margin:5px 0 0 0; font-size:18px; font-weight:bold;">Λ Proyectado: {lambda_ast:.2f} ast</p>
                        <p style="margin:2px 0 0 0; font-size:12px; color:#94a3b8;">Tasa Base: {proy_detallada['tasa_ast_min']:.2f} ast/min</p>
                        <p style="margin:5px 0 0 0; font-size:18px; font-weight:bold;">{verdict_ast}</p>
                        <p style="margin:0; font-size:12px; color:#cbd5e1;">Más: {prob_more_ast:.1f}% | Menos: {prob_less_ast:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)
                st.progress(min(max(int(prob_more_ast), 0), 100))