import cv2
import numpy as np
import random

def generar_mapa_manchas(h, w, escala=16):
    """
    Genera un mapa de textura orgánica (manchas de suciedad/polución)
    creando ruido de muy baja resolución y reescalándolo con bicúbico.
    """
    grid_h = max(2, h // escala)
    grid_w = max(2, w // escala)
    ruido_base = np.random.uniform(0.0, 1.0, (grid_h, grid_w)).astype(np.float32)
    mapa = cv2.resize(ruido_base, (w, h), interpolation=cv2.INTER_CUBIC)
    # Normalizar entre 0.0 y 1.0
    mapa = (mapa - mapa.min()) / (mapa.max() - mapa.min() + 1e-6)
    return mapa

def aplicar_degradacion_completa(
    img_bgr,
    seed=None,
    dirt_intensity=0.35,     # Manchas de suciedad/polución (0.0 = limpio, 1.0 = muy sucio)
    wear_fading=0.25,        # Desgaste solar / pérdida de saturación (0.0 = nuevo, 1.0 = descolorido)
    scratch_level=0.15,      # Micro-desconchones y erosión de pintura (0.0 a 1.0)
    shadow_strength=0.20,    # Gradiente de sombra/iluminación no uniforme (0.0 a 1.0)
    noise_level=12.0,        # Ruido de sensor ISO visible (0 a 50)
    blur_amount=1.2,         # Desenfoque óptico/movimiento (0.0 a 5.0)
    jpeg_quality=65          # Compresión JPEG (100 = perfecto, 30 = compresión severa)
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed % (2**32))

    h, w, c = img_bgr.shape
    out = img_bgr.astype(np.float32)

    # 1. DESGASTE SOLAR (Desaturación y aclarado en HSV)
    if wear_fading > 0.0:
        hsv = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        # Reducir saturación de forma irregular (el sol quema más unas zonas que otras)
        mapa_sol = generar_mapa_manchas(h, w, escala=24)
        factor_sat = 1.0 - (wear_fading * (0.5 + 0.5 * mapa_sol))
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor_sat, 0, 255)
        # Quemar ligeramente el brillo en las zonas desgastadas
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + (wear_fading * 30.0 * mapa_sol), 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    # 2. MANCHAS ORGÁNICAS (Suciedad, lluvia seca y hollín)
    if dirt_intensity > 0.0:
        # Mezclamos dos capas de ruido (una gruesa para manchas grandes y una media para salpicaduras)
        mancha_gruesa = generar_mapa_manchas(h, w, escala=20)
        mancha_fina = generar_mapa_manchas(h, w, escala=8)
        mapa_dirt = (0.7 * mancha_gruesa + 0.3 * mancha_fina)
        
        # Umbralizamos para que la suciedad se concentre en parches irregulares
        mapa_dirt = np.power(mapa_dirt, 2.0) * dirt_intensity
        
        # Color de suciedad típico (marrón/grisáceo oscuro en BGR)
        color_suciedad = np.array([35.0, 45.0, 50.0], dtype=np.float32)
        
        for ch in range(3):
            # Oscurece y mancha hacia el tono tierra
            out[:, :, ch] = out[:, :, ch] * (1.0 - mapa_dirt) + (color_suciedad[ch] * mapa_dirt)

    # 3. MICRO-DESCONCHONES Y EROSIÓN DE BORDES
    if scratch_level > 0.0:
        # Ruido de sal y pimienta de alta frecuencia umbralizado
        ruido_puntos = np.random.uniform(0.0, 1.0, (h, w))
        mascara_chips = (ruido_puntos > (1.0 - (scratch_level * 0.05))).astype(np.float32)
        # Los desconchones muestran el metal inferior (gris neutro oxidado)
        for ch in range(3):
            out[:, :, ch] = out[:, :, ch] * (1.0 - mascara_chips) + (110.0 * mascara_chips)

    # 4. GRADIENTE DE SOMBRA / ILUMINACIÓN AMBIENTAL
    if shadow_strength > 0.0:
        # Simula que la luz del sol viene de arriba a la izquierda y el borde opuesto queda en penumbra
        eje_x = np.linspace(-1, 1, w)
        eje_y = np.linspace(-1, 1, h)
        xx, yy = np.meshgrid(eje_x, eje_y)
        # Dirección aleatoria del sol según semilla
        angulo = random.uniform(0, 2 * np.pi)
        gradiente = np.cos(angulo) * xx + np.sin(angulo) * yy
        gradiente = (gradiente - gradiente.min()) / (gradiente.max() - gradiente.min())
        sombra = 1.0 - (shadow_strength * 0.6 * gradiente)
        for ch in range(3):
            out[:, :, ch] = np.clip(out[:, :, ch] * sombra, 0, 255)

    # 5. RUIDO DE SENSOR (ISO alto visible)
    if noise_level > 0.0:
        noise = np.random.normal(0, noise_level, (h, w, c))
        out = np.clip(out + noise, 0, 255)

    out = np.clip(out, 0, 255).astype(np.uint8)

    # 6. DESENFOQUE ÓPTICO
    if blur_amount > 0.0:
        k_size = int(np.ceil(blur_amount * 2)) | 1  # Forzar número impar
        out = cv2.GaussianBlur(out, (k_size, k_size), sigmaX=blur_amount, sigmaY=blur_amount)

    # 7. COMPRESIÓN JPEG (Bloques DCT de 8x8 típicos de cámara de abordo)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    _, enc = cv2.imencode('.jpg', out, encode_param)
    out = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return out
