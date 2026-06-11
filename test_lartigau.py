import requests
import xml.etree.ElementTree as ET

URL = "https://meteobahia.com.ar/scripts/forecast/for-cf.xml"
headers = {"User-Agent": "Mozilla/5.0"}

try:
    r = requests.get(URL, headers=headers, timeout=20)
    print(f"1. Status Code: {r.status_code}")
    
    root = ET.fromstring(r.content)
    # Imprimir la estructura raíz para ver si cambió el tag
    print(f"2. Tag Raíz: {root.tag}")
    
    elementos = root.findall(".//forecast/tabular/day")
    print(f"3. Cantidad de nodos 'day' encontrados: {len(elementos)}")
    
    if len(elementos) > 0:
        print("4. Muestra del primer nodo:")
        primero = elementos[0]
        print(f"   Fecha: {primero.find('fecha').get('value') if primero.find('fecha') is not None else 'No encontrado'}")
        print(f"   TMAX: {primero.find('tmax').get('value') if primero.find('tmax') is not None else 'No encontrado'}")
        
except Exception as e:
    print(f"❌ Error durante el test: {e}")
