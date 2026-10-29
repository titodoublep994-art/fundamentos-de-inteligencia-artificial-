"""
===============================================================================
INFORME DE INVESTIGACIÓN: APLICACIONES Y CASOS DE ESTUDIO DE LA IA
Código de Simulación Python para los 3 Casos de Estudio (Salud, Industria, Movilidad)
Ref: García Serrano (Cap. 1); Russell & Norvig (Cap. 1)
===============================================================================
"""

import time
import random
import numpy as np

# =============================================================================
# CASO 1: SALUD - SISTEMA DE TRIAJE OCULAR (Google DeepMind Simulation)
# =============================================================================
class OcularTriageAI:
    """
    Simula la clasificación de imágenes OCT 3D para triaje oftalmológico urgente.
    Entrada: Matriz de densidades tisulares de escaneo retinal.
    Salida: Diagnóstico probabilístico y nivel de urgencia de triaje.
    """
    def __init__(self):
        self.classes = ["Normal", "DMAE Exudativa", "Retinopatía Diabética", "Agujero Macular"]
        self.urgency_map = {
            "Normal": "BAJA (Cita Rutinaria)",
            "DMAE Exudativa": "ALTA (Derivación Urgente <24h)",
            "Retinopatía Diabética": "MEDIA (Seguimiento <7 días)",
            "Agujero Macular": "ALTA (Derivación Urgente <48h)"
        }

    def predict_oct_scan(self, oct_volume_matrix: np.ndarray) -> dict:
        """
        Procesa el volumen OCT 3D y realiza la clasificación en 2 etapas.
        """
        start_time = time.time()
        # Etapa 1: Segmentación Tisular (Extracción de características promedio)
        tissue_density = np.mean(oct_volume_matrix)
        variance = np.var(oct_volume_matrix)
        
        # Etapa 2: Clasificación Probabilística
        if tissue_density > 0.75:
            predicted_class = "DMAE Exudativa"
            confidence = 0.962
        elif variance > 0.15:
            predicted_class = "Retinopatía Diabética"
            confidence = 0.941
        elif tissue_density < 0.25:
            predicted_class = "Agujero Macular"
            confidence = 0.928
        else:
            predicted_class = "Normal"
            confidence = 0.985
            
        processing_time_ms = (time.time() - start_time) * 1000
        
        return {
            "diagnostico": predicted_class,
            "confianza": f"{confidence * 100:.1f}%",
            "nivel_urgencia": self.urgency_map[predicted_class],
            "tiempo_procesamiento_ms": f"{processing_time_ms:.2f} ms"
        }

# =============================================================================
# CASO 2: INDUSTRIA - INSPECCIÓN DE CALIDAD VISUAL (BMW Group AIQX Simulation)
# =============================================================================
class QualityControlAI:
    """
    Simula la inspección visual en tiempo real en una línea de montaje automotriz.
    Entrada: Matriz de imagen de alta definición de la pieza.
    Salida: Estado de conformidad (Aprobado / Defecto) y latencia.
    """
    def __init__(self, defect_threshold: float = 0.18):
        self.threshold = defect_threshold

    def inspect_component(self, image_frame: np.ndarray) -> dict:
        """
        Analiza el frame de la pieza y detecta anomalías micro-estructurales.
        """
        start_time = time.time()
        
        # Simulación de extracción de características por Red Convolucional (CNN)
        anomaly_score = np.std(image_frame) / (np.mean(image_frame) + 1e-5)
        
        is_defective = anomaly_score > self.threshold
        status = "RECHAZADO (Defecto Detectado)" if is_defective else "APROBADO (Conforme)"
        
        latency_ms = (time.time() - start_time) * 1000 + random.uniform(5.0, 12.0) # Latencia <100ms
        
        return {
            "estado_inspeccion": status,
            "score_anomalia": round(float(anomaly_score), 4),
            "umbral_tolerancia": self.threshold,
            "latencia_ms": f"{latency_ms:.2f} ms"
        }

# =============================================================================
# CASO 3: MOVILIDAD - FUSIÓN DE SENSORES Y DECISIÓN (Waymo Driver Simulation)
# =============================================================================
class AutonomousDriverAI:
    """
    Simula la toma de decisiones de un vehículo autónomo mediante fusión de sensores.
    Entrada: Distancia LiDAR (m), velocidad actual (km/h) y detección de cámara.
    Salida: Acción de control (Acelerar, Frenar, Frenado de Emergencia).
    """
    def __init__(self):
        pass

    def evaluate_driving_step(self, lidar_dist_meters: float, current_speed_kmh: float, camera_object: str) -> dict:
        """
        Aplica lógica de agente racional para evitar colisiones y optimizar trayectoria.
        """
        # Distancia de seguridad mínima según velocidad
        min_safety_margin = (current_speed_kmh / 10.0) * 3.0 # metros
        
        if lidar_dist_meters < 5.0 or camera_object in ["Peatón en vía", "Obstáculo crítico"]:
            action = "FRENADO DE EMERGENCIA"
            risk_level = "CRÍTICO"
        elif lidar_dist_meters < min_safety_margin:
            action = "DESACELERAR Y MANTENER DISTANCIA"
            risk_level = "MEDIO"
        else:
            action = "MANTENER VELOCIDAD CRUCERO"
            risk_level = "BAJO"
            
        return {
            "distancia_lidar": f"{lidar_dist_meters:.1f} m",
            "velocidad_actual": f"{current_speed_kmh:.1f} km/h",
            "objeto_detectado_camara": camera_object,
            "nivel_riesgo": risk_level,
            "accion_control": action
        }

# =============================================================================
# EJECUCIÓN Y DEMOSTRACIÓN PRINCIPAL
# =============================================================================
def main():
    print("=" * 80)
    print("DEMOSTRACIÓN DE CÓDIGO PYTHON: CASOS REALES DE APLICACIÓN DE IA")
    print("Sección 1.7 - Fundamentos de Inteligencia Artificial")
    print("=" * 80 + "\n")

    # 1. Ejecución Caso Salud
    print(">>> CASO 1: SALUD - TRIAJE OCULAR AUTOMATIZADO (Google DeepMind)")
    health_ai = OcularTriageAI()
    sample_oct_data = np.random.normal(loc=0.8, scale=0.1, size=(64, 64, 32)) # Simulación de escaneo 3D
    health_result = health_ai.predict_oct_scan(sample_oct_data)
    for k, v in health_result.items():
        print(f"  • {k.replace('_', ' ').capitalize()}: {v}")
    print("\n" + "-" * 80 + "\n")

    # 2. Ejecución Caso Industria
    print(">>> CASO 2: INDUSTRIA - INSPECCIÓN DE CALIDAD VISUAL (BMW Group AIQX)")
    industry_ai = QualityControlAI()
    sample_image = np.random.uniform(low=0, high=255, size=(1080, 1920, 3)) # Frame 1080p
    industry_result = industry_ai.inspect_component(sample_image)
    for k, v in industry_result.items():
        print(f"  • {k.replace('_', ' ').capitalize()}: {v}")
    print("\n" + "-" * 80 + "\n")

    # 3. Ejecución Caso Movilidad
    print(">>> CASO 3: MOVILIDAD - CONDUCCIÓN AUTÓNOMA Y FUSIÓN DE SENSORES (Waymo Driver)")
    driver_ai = AutonomousDriverAI()
    mobility_result = driver_ai.evaluate_driving_step(lidar_dist_meters=4.2, current_speed_kmh=45.0, camera_object="Peatón en vía")
    for k, v in mobility_result.items():
        print(f"  • {k.replace('_', ' ').capitalize()}: {v}")
    print("\n" + "=" * 80)
    print("SIMULACIÓN COMPLETADA CON ÉXITO.")
    print("=" * 80)

if __name__ == "__main__":
    main()