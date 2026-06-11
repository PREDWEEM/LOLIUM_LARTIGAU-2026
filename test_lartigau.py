import streamlit as st
import requests
import xml.etree.ElementTree as ET

# ---------------------------------------------------------
# INTERFAZ DE DIAGNÓSTICO METEOBAHÍA EN STREAMLIT
# ---------------------------------------------------------
st.markdown("### 🧪 Panel de Test: Conexión Nodo Lartigau (MeteoBahía)")
st.caption("Utiliza este módulo para verificar si el servidor de producción tiene acceso al XML y procesa las etiquetas correctamente.")

if st.button("🚀 Ejecutar Test de API/XML"):
    URL = "https://meteobahia.com.ar/scripts/forecast/for-cf.xml"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    with st.spinner("Estableciendo conexión con MeteoBahía..."):
        try:
            r = requests.get(URL, headers=headers, timeout=20)
            
            # 1. Código de Estado HTTP
            c1, c2 = st.columns(2)
            with c1:
                if r.status_code == 200:
                    st.success(f"🟢 Conexión Exitosa (Status: {r.status_code})")
                else:
                    st.error(f"🔴 Error de Servidor (Status: {r.status_code})")
            
            # Parseo del XML
            root = ET.fromstring(r.content)
            
            with c2:
                st.metric(label="Etiqueta Raíz XML", value=f"<{root.tag}>")
            
            # 2. Búsqueda de Nodos 'day'
            elementos = root.findall(".//forecast/tabular/day")
            
            st.markdown("---")
            st.metric(label="Nodos '<day>' detectados", value=len(elementos))
            
            # 3. Inspección del Primer Nodo Encontrado
            if len(elementos) > 0:
                st.markdown("#### 📂 Estructura del Primer Registro Hallado:")
                primero = elementos[0]
                
                fecha_val = primero.find('fecha').get('value') if primero.find('fecha') is not None else '❌ No encontrado'
                tmax_val = primero.find('tmax').get('value') if primero.find('tmax') is not None else '❌ No encontrado'
                tmin_val = primero.find('tmin').get('value') if primero.find('tmin') is not None else '❌ No encontrado'
                prec_val = primero.find('precip').get('value') if primero.find('precip') is not None else '❌ No encontrado'
                
                # Renderizado en formato JSON/Diccionario para facilitar la lectura técnica
                muestra_datos = {
                    "Nodo": "forecast/tabular/day[0]",
                    "Atributo Fecha": fecha_val,
                    "Atributo TMAX": tmax_val,
                    "Atributo TMIN": tmin_val,
                    "Atributo Precipitación": prec_val
                }
                st.json(muestra_datos)
                
                if '❌' in [fecha_val, tmax_val, tmin_val, prec_val]:
                    st.warning("⚠️ Atención: El XML responde pero algunas etiquetas internas cambiaron de nombre.")
                else:
                    st.balloons()
                    st.success("🏁 ¡Estructura XML validada! Los nombres de las etiquetas coinciden con el motor analítico de PREDWEEM.")
            else:
                st.error("❌ El XML fue leído pero la ruta `//forecast/tabular/day` devolvió 0 elementos. La estructura interna del XML cambió.")
                
        except requests.exceptions.Timeout:
            st.error("❌ Error: Tiempo de espera agotado (Timeout). El servidor de MeteoBahía tardó más de 20 segundos en responder.")
        except requests.exceptions.ConnectionError:
            st.error("❌ Error de Conexión: No se pudo establecer el enlace. Si estás en Hugging Face, verifica que el host no tenga bloqueadas las peticiones salientes HTTP.")
        except Exception as e:
            st.error(f"❌ Fallo crítico durante el procesamiento analítico: {e}")
