import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
import json
import sqlite3
import os
import requests
from io import BytesIO
import plotly.graph_objects as go
from scipy.stats import poisson

st.set_page_config(page_title="ValueBet Lab", page_icon="🔍", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0A1F14; }
    h1, h2, h3 { color: #5DCAA5 !important; }
    .stButton > button {
        background-color: #0F6E56; color: white; border: none;
        border-radius: 8px; padding: 10px 24px; font-weight: 600;
    }
    .stButton > button:hover { background-color: #1D9E75; }
    .card {
        background: linear-gradient(135deg, #0A1F14, #0F6E56);
        border: 1px solid #1D9E75; border-radius: 12px;
        padding: 16px; margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# BASE DE DATOS
# ============================================================

def init_db():
    conn = sqlite3.connect("valuebet.db")
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS encuentros (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fecha DATE NOT NULL,
        equipo_local TEXT NOT NULL,
        equipo_visitante TEXT NOT NULL,
        estado TEXT DEFAULT 'pendiente',
        resultado_analisis TEXT,
        fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit()
    conn.close()

init_db()

def get_db():
    conn = sqlite3.connect("valuebet.db")
    conn.row_factory = sqlite3.Row
    return conn

def add_match(fecha, local, visitante):
    conn = get_db()
    conn.execute("INSERT INTO encuentros (fecha, equipo_local, equipo_visitante) VALUES (?, ?, ?)", (fecha, local, visitante))
    conn.commit()
    id_ = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return id_

def get_matches(fecha):
    conn = get_db()
    m = conn.execute("SELECT * FROM encuentros WHERE fecha = ? ORDER BY equipo_local", (fecha,)).fetchall()
    conn.close()
    return m

def get_match(id_):
    conn = get_db()
    m = conn.execute("SELECT * FROM encuentros WHERE id = ?", (id_,)).fetchone()
    conn.close()
    return m

def update_match(id_, resultado_json):
    conn = get_db()
    conn.execute("UPDATE encuentros SET estado='analizado', resultado_analisis=? WHERE id=?", (resultado_json, id_))
    conn.commit()
    conn.close()

def delete_match(id_):
    conn = get_db()
    conn.execute("DELETE FROM encuentros WHERE id = ?", (id_,))
    conn.commit()
    conn.close()

def get_all_analizados():
    conn = get_db()
    m = conn.execute("SELECT * FROM encuentros WHERE estado='analizado' ORDER BY fecha DESC").fetchall()
    conn.close()
    return m

# ============================================================
# MODELO POISSON (100% local, sin API)
# ============================================================

def analyze_match_poisson(local, visitante):
    # Parámetros basados en estadísticas genéricas de ligas europeas
    # En una versión futura, estos valores vendrán de datos reales
    fuerza_local = 1.0
    fuerza_visitante = 1.0
    
    # Ajuste por nombres de equipos conocidos (pequeña base de conocimientos)
    equipos_fuertes_casa = ["real madrid", "barcelona", "atletico", "man city", "liverpool", 
                           "bayern", "psg", "inter", "juventus", "napoles"]
    equipos_debiles_fuera = ["levante", "alaves", "cadiz", "granada", "elche", "getafe"]
    
    for fuerte in equipos_fuertes_casa:
        if fuerte in local.lower():
            fuerza_local = 1.3
        if fuerte in visitante.lower():
            fuerza_visitante = 1.2
    
    for debil in equipos_debiles_fuera:
        if debil in visitante.lower():
            fuerza_visitante = 0.7
        if debil in local.lower():
            fuerza_local = 0.8
    
    # Goles esperados con ventaja local
    lambda_h = 1.4 * fuerza_local + 0.3  # +0.3 por ventaja local
    lambda_a = 1.1 * fuerza_visitante
    
    # Calcular todas las probabilidades
    max_g = 6
    probs = np.zeros((max_g+1, max_g+1))
    for i in range(max_g+1):
        for j in range(max_g+1):
            probs[i][j] = poisson.pmf(i, lambda_h) * poisson.pmf(j, lambda_a)
    probs = probs / probs.sum()
    
    p_local = float(np.sum(probs * np.tri(max_g+1, max_g+1, -1).T))
    p_empate = float(np.trace(probs))
    p_visitante = float(np.sum(probs * np.tri(max_g+1, max_g+1, -1)))
    p_over_2_5 = float(1 - (probs[0,0]+probs[0,1]+probs[1,0]+probs[1,1]))
    p_btts = float(np.sum(probs[1:, 1:]))
    p_over_1_5 = float(1 - probs[0,0])
    
    # Generar picks inteligentes
    picks = []
    
    # Pick 1: Doble oportunidad (el más seguro)
    if p_local > 0.45:
        prob_1x = p_local + p_empate
        cuota_1x = round(1 / max(prob_1x, 0.01), 2)
        value_1x = round((prob_1x / (1/cuota_1x) - 1) * 100, 1)
        picks.append({
            "mercado": "Doble Oportunidad",
            "pick": "1X (Local o Empate)",
            "probabilidad": round(prob_1x * 100, 1),
            "cuota": cuota_1x,
            "value": f"{value_1x}%",
            "justificacion": f"{local} tiene {round(p_local*100)}% de ganar en casa. Sumando el empate, la probabilidad es del {round(prob_1x*100)}%"
        })
    
    # Pick 2: Goles
    if p_over_2_5 > 0.50:
        cuota_ov = round(1 / max(p_over_2_5, 0.01), 2)
        value_ov = round((p_over_2_5 / (1/cuota_ov) - 1) * 100, 1)
        picks.append({
            "mercado": "Goles",
            "pick": "Over 2.5 Goles",
            "probabilidad": round(p_over_2_5 * 100, 1),
            "cuota": cuota_ov,
            "value": f"{value_ov}%",
            "justificacion": f"Se esperan {round(lambda_h+lambda_a, 1)} goles totales. Probabilidad de Over 2.5: {round(p_over_2_5*100)}%"
        })
    else:
        prob_under = 1 - p_over_2_5
        cuota_un = round(1 / max(prob_under, 0.01), 2)
        picks.append({
            "mercado": "Goles",
            "pick": "Under 2.5 Goles",
            "probabilidad": round(prob_under * 100, 1),
            "cuota": cuota_un,
            "value": "N/A",
            "justificacion": f"Partido con pocos goles esperados ({round(lambda_h+lambda_a, 1)}). Under 2.5 al {round(prob_under*100)}%"
        })
    
    # Pick 3: BTTS
    if p_btts > 0.50:
        cuota_btts = round(1 / max(p_btts, 0.01), 2)
        value_btts = round((p_btts / (1/cuota_btts) - 1) * 100, 1)
        picks.append({
            "mercado": "BTTS",
            "pick": "Si (Ambos anotan)",
            "probabilidad": round(p_btts * 100, 1),
            "cuota": cuota_btts,
            "value": f"{value_btts}%",
            "justificacion": f"Ambos equipos tienen probabilidad de marcar. BTTS estimado: {round(p_btts*100)}%"
        })
    else:
        cuota_no_btts = round(1 / max(1-p_btts, 0.01), 2)
        picks.append({
            "mercado": "BTTS",
            "pick": "No (No anotan ambos)",
            "probabilidad": round((1-p_btts)*100, 1),
            "cuota": cuota_no_btts,
            "value": "N/A",
            "justificacion": f"Algun equipo podria no marcar. No BTTS: {round((1-p_btts)*100)}%"
        })
    
    # Pick 4: Intervalo de goles (el mercado estrella)
    prob_1_3 = float(np.sum(probs[1:4, :]) + np.sum(probs[:, 1:4]) - np.sum(probs[1:4, 1:4]))
    prob_1_3 = min(prob_1_3, 0.95)
    if prob_1_3 > 0.50:
        cuota_1_3 = round(1 / max(prob_1_3, 0.01), 2)
        picks.append({
            "mercado": "Intervalo Goles",
            "pick": "1-3 Goles",
            "probabilidad": round(prob_1_3 * 100, 1),
            "cuota": cuota_1_3,
            "value": "ALTO",
            "justificacion": "Mercado estrella. Partidos entre equipos equilibrados tienden a 1-3 goles"
        })
    
    # Calcular cuota combinada
    cuota_combinada = round(np.prod([p["cuota"] for p in picks[:3]]), 2)
    
    return {
        "goles_local": round(lambda_h, 2),
        "goles_visitante": round(lambda_a, 2),
        "goles_totales": round(lambda_h + lambda_a, 2),
        "prob_local": round(p_local * 100, 1),
        "prob_empate": round(p_empate * 100, 1),
        "prob_visitante": round(p_visitante * 100, 1),
        "prob_over_2_5": round(p_over_2_5 * 100, 1),
        "prob_btts": round(p_btts * 100, 1),
        "prob_1_3_goles": round(prob_1_3 * 100, 1),
        "picks": picks,
        "cuota_combinada": cuota_combinada,
        "modelo": "Poisson Ajustado",
        "confianza": "ALTA" if max(p_local, p_empate, p_visitante) > 0.50 else "MEDIA"
    }

# ============================================================
# INTERFAZ DE USUARIO
# ============================================================

def main():
    st.markdown("<h1 style='text-align:center;'>🔍 ValueBet Lab</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#9FE1CB;'>Analizador Estadistico de Partidos • Modelo Poisson</p>", unsafe_allow_html=True)
    
    # Menú de navegación
    tab1, tab2, tab3 = st.tabs(["📅 Partidos", "📊 Analizar", "📈 Historial"])
    
    with tab1:
        show_matches_tab()
    with tab2:
        show_analyze_tab()
    with tab3:
        show_history_tab()

def show_matches_tab():
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### ➕ Nuevo Partido")
        fecha = st.date_input("Fecha", datetime.now(), key="fecha_add")
        local = st.text_input("🏠 Local", key="local_add")
        visit = st.text_input("✈️ Visitante", key="visit_add")
        
        if st.button("➕ Añadir Partido", use_container_width=True):
            if local and visit and local != visit:
                add_match(fecha, local, visit)
                st.success(f"✅ {local} vs {visit}")
                st.rerun()
            else:
                st.error("Completa todos los campos")
        
        st.markdown("---")
        st.markdown("### 📊 Estadisticas")
        conn = get_db()
        total = conn.execute("SELECT COUNT(*) FROM encuentros").fetchone()[0]
        analizados = conn.execute("SELECT COUNT(*) FROM encuentros WHERE estado='analizado'").fetchone()[0]
        conn.close()
        st.metric("Total Partidos", total)
        st.metric("Analizados", analizados)
    
    with col2:
        st.markdown(f"### 📋 Partidos del {datetime.now().strftime('%d/%m/%Y')}")
        
        fecha_actual = date.today()
        matches = get_matches(fecha_actual)
        
        if not matches:
            st.info("ℹ️ No hay partidos para hoy. Anade uno a la izquierda.")
        else:
            for m in matches:
                estado = "🟢 Analizado" if m["estado"] == "analizado" else "🟡 Pendiente"
                st.markdown(f"""
                <div class="card">
                    <strong style="color:#E1F5EE;font-size:18px;">{m["equipo_local"]}</strong>
                    <span style="color:#9FE1CB;"> vs </span>
                    <strong style="color:#E1F5EE;font-size:18px;">{m["equipo_visitante"]}</strong>
                    <span style="color:#5DCAA5;margin-left:15px;">{estado}</span>
                </div>
                """, unsafe_allow_html=True)
                
                col_a, col_b, col_c = st.columns([2, 2, 1])
                with col_a:
                    if st.button(f"🔍 Analizar {m['equipo_local']} vs {m['equipo_visitante']}", 
                               key=f"go_{m['id']}",
                               disabled=m['estado']=='analizado'):
                        st.session_state['analizar_id'] = m['id']
                        st.session_state['tab'] = 2
                        st.rerun()
                with col_b:
                    if m['estado'] == 'analizado':
                        if st.button(f"📊 Ver resultados", key=f"view_{m['id']}"):
                            st.session_state['ver_id'] = m['id']
                            st.session_state['tab'] = 2
                            st.rerun()
                with col_c:
                    if st.button(f"🗑️", key=f"del_{m['id']}"):
                        delete_match(m['id'])
                        st.rerun()

def show_analyze_tab():
    analizar_id = st.session_state.get('analizar_id', None)
    ver_id = st.session_state.get('ver_id', None)
    
    match_id = analizar_id or ver_id
    
    if not match_id:
        st.info("Selecciona un partido de la pestaña 'Partidos' para analizar")
        return
    
    m = get_match(match_id)
    if not m:
        st.error("Partido no encontrado")
        return
    
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0A1F14,#0F6E56);
         border:2px solid #1D9E75;border-radius:16px;padding:24px;margin:16px 0;">
        <h2 style="text-align:center;color:#5DCAA5;margin:0;">
            {m['equipo_local']} vs {m['equipo_visitante']}
        </h2>
        <p style="text-align:center;color:#9FE1CB;">{m['fecha']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    if analizar_id:
        if st.button("🚀 ANALIZAR PARTIDO", use_container_width=True):
            with st.spinner("Analizando con modelo Poisson..."):
                import time
                time.sleep(1.5)  # Simular proceso
                resultado = analyze_match_poisson(m['equipo_local'], m['equipo_visitante'])
                update_match(match_id, json.dumps(resultado))
            st.success("✅ Analisis completado!")
            st.rerun()
    
    if m['estado'] == 'analizado' or ver_id:
        resultado = json.loads(m['resultado_analisis'])
        mostrar_resultados(resultado, m)

def mostrar_resultados(resultado, m):
    # Goles esperados
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("⚽ Goles Local", resultado["goles_local"], 
                 delta=f"vs {m['equipo_visitante']}")
    with col2:
        st.metric("📊 Total", resultado["goles_totales"])
    with col3:
        st.metric("⚽ Goles Visitante", resultado["goles_visitante"],
                 delta=f"vs {m['equipo_local']}")
    
    st.markdown("---")
    
    # Gráfico de probabilidades
    st.markdown("### 📊 Probabilidades del Resultado")
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[m['equipo_local'], "Empate", m['equipo_visitante']],
        y=[resultado["prob_local"], resultado["prob_empate"], resultado["prob_visitante"]],
        marker_color=["#1D9E75", "#5DCAA5", "#0F6E56"],
        text=[f'{resultado["prob_local"]}%', f'{resultado["prob_empate"]}%', f'{resultado["prob_visitante"]}%'],
        textposition="auto",
    ))
    fig.update_layout(
        title=f"{m['equipo_local']} vs {m['equipo_visitante']}",
        yaxis_title="Probabilidad (%)",
        yaxis_range=[0, 100],
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E1F5EE"),
        height=350,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Métricas clave
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Indicador Over 2.5
        fig2 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=resultado["prob_over_2_5"],
            delta={"reference": 50},
            title={"text": "Over 2.5 Goles", "font": {"color": "#E1F5EE", "size": 14}},
            number={"suffix": "%", "font": {"color": "#5DCAA5", "size": 24}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#5DCAA5"},
                "bar": {"color": "#1D9E75"},
                "bgcolor": "#0A1F14",
                "borderwidth": 2,
                "bordercolor": "#0F6E56",
                "steps": [
                    {"range": [0, 50], "color": "#0A1F14"},
                    {"range": [50, 100], "color": "#0F6E56"}
                ]
            }
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#E1F5EE"},
            height=250,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        fig3 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=resultado["prob_btts"],
            delta={"reference": 50},
            title={"text": "BTTS (Ambos marcan)", "font": {"color": "#E1F5EE", "size": 14}},
            number={"suffix": "%", "font": {"color": "#5DCAA5", "size": 24}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#5DCAA5"},
                "bar": {"color": "#1D9E75"},
                "bgcolor": "#0A1F14",
                "borderwidth": 2,
                "bordercolor": "#0F6E56",
                "steps": [
                    {"range": [0, 50], "color": "#0A1F14"},
                    {"range": [50, 100], "color": "#0F6E56"}
                ]
            }
        ))
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#E1F5EE"},
            height=250,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col3:
        fig4 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=resultado["prob_1_3_goles"],
            delta={"reference": 50},
            title={"text": "Intervalo 1-3 Goles ⭐", "font": {"color": "#E1F5EE", "size": 14}},
            number={"suffix": "%", "font": {"color": "#5DCAA5", "size": 24}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#5DCAA5"},
                "bar": {"color": "#1D9E75"},
                "bgcolor": "#0A1F14",
                "borderwidth": 2,
                "bordercolor": "#0F6E56",
                "steps": [
                    {"range": [0, 50], "color": "#0A1F14"},
                    {"range": [50, 100], "color": "#0F6E56"}
                ]
            }
        ))
        fig4.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#E1F5EE"},
            height=250,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    st.markdown("---")
    
    # Picks sugeridos
    st.markdown("### 🏆 PICKS SUGERIDOS")
    st.markdown(f"<p style='color:#9FE1CB;'>Modelo: {resultado['modelo']} | Confianza: {resultado['confianza']} | Cuota Combinada: <strong style='color:#5DCAA5;'>{resultado['cuota_combinada']}</strong></p>", unsafe_allow_html=True)
    
    if resultado["picks"]:
        for i, pick in enumerate(resultado["picks"], 1):
            st.markdown(f"""
            <div class="card">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <h3 style="color:#5DCAA5;margin:0;">Pick #{i}</h3>
                    <span style="background:#1D9E75;padding:4px 12px;border-radius:12px;color:white;font-size:12px;">
                        {pick['value'] if pick['value'] != 'N/A' else 'Value moderado'}
                    </span>
                </div>
                <p style="color:#E1F5EE;font-size:20px;margin:10px 0;">
                    <strong>{pick['mercado']}:</strong> {pick['pick']}
                </p>
                <div style="display:flex;gap:30px;margin:8px 0;">
                    <span style="color:#9FE1CB;">📊 Probabilidad: <strong>{pick['probabilidad']}%</strong></span>
                    <span style="color:#5DCAA5;">💰 Cuota: <strong>{pick['cuota']}</strong></span>
                    <span style="color:#E1F5EE;">📈 Value: <strong>{pick['value']}</strong></span>
                </div>
                <p style="color:#9FE1CB;font-size:13px;margin-top:8px;border-top:1px solid #0F6E56;padding-top:8px;">
                    📌 {pick['justificacion']}
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Recomendación de stake
    st.markdown("---")
    st.markdown("### 💰 Gestion de Bankroll Recomendada")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(f"""<div class="card" style="text-align:center;">
            <div style="color:#5DCAA5;font-size:12px;">STAKE RECOMENDADO</div>
            <div style="color:#E1F5EE;font-size:28px;font-weight:700;">2%</div>
            <div style="color:#9FE1CB;font-size:11px;">del bankroll total</div>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""<div class="card" style="text-align:center;">
            <div style="color:#5DCAA5;font-size:12px;">NIVEL DE RIESGO</div>
            <div style="color:#E1F5EE;font-size:28px;font-weight:700;">{resultado['confianza']}</div>
            <div style="color:#9FE1CB;font-size:11px;">basado en el modelo</div>
        </div>""", unsafe_allow_html=True)
    with col_c:
        st.markdown(f"""<div class="card" style="text-align:center;">
            <div style="color:#5DCAA5;font-size:12px;">CUOTA COMBINADA ESTIMADA</div>
            <div style="color:#E1F5EE;font-size:28px;font-weight:700;">{resultado['cuota_combinada']}</div>
            <div style="color:#9FE1CB;font-size:11px;">para los 3 primeros picks</div>
        </div>""", unsafe_allow_html=True)

def show_history_tab():
    st.markdown("### 📈 Historial de Analisis")
    
    analizados = get_all_analizados()
    
    if not analizados:
        st.info("Aún no hay partidos analizados. Ve a 'Partidos' y analiza alguno.")
        return
    
    data = []
    for m in analizados:
        r = json.loads(m['resultado_analisis'])
        picks_count = len(r['picks'])
        data.append({
            "Fecha": m['fecha'],
            "Partido": f"{m['equipo_local']} vs {m['equipo_visitante']}",
            "Goles Local": r['goles_local'],
            "Goles Visit": r['goles_visitante'],
            "Total Goles": r['goles_totales'],
            "Prob Local": f"{r['prob_local']}%",
            "Over 2.5": f"{r['prob_over_2_5']}%",
            "BTTS": f"{r['prob_btts']}%",
            "Picks": picks_count,
            "Cuota Comb.": r['cuota_combinada']
        })
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Botón de descarga
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Descargar CSV del historial",
        csv,
        "valuebet_historial.csv",
        "text/csv",
        use_container_width=True
    )

if __name__ == "__main__":
    main()
