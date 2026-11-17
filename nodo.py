import torch
import numpy as np
import cv2
import random
from .degradar import aplicar_degradacion_completa

def extraer_mascara_central_opencv(img_bgr):
    """Segmenta la figura central del recorte de 512x512 sin usar IA."""
    h, w = img_bgr.shape[:2]
    scale_factor = 256.0 / max(h, w)
    small_w, small_h = int(w * scale_factor), int(h * scale_factor)
    small_img = cv2.resize(img_bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)

    mask_gc = np.zeros((small_h, small_w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    margin = int(min(small_h, small_w) * 0.12)
    rect = (margin, margin, small_w - 2 * margin, small_h - 2 * margin)
    
    cv2.grabCut(small_img, mask_gc, rect, bgdModel, fgdModel, 2, cv2.GC_INIT_WITH_RECT)
    bin_mask = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask)
    centro_x, centro_y = small_w // 2, small_h // 2
    label_central = labels[centro_y, centro_x]

    if label_central > 0:
        mascara_aislada = np.where(labels == label_central, 255, 0).astype(np.uint8)
    else:
        distancias = [np.hypot(centroids[i][0] - centro_x, centroids[i][1] - centro_y) for i in range(1, num_labels)]
        if distancias:
            mejor_label = np.argmin(distancias) + 1
            mascara_aislada = np.where(labels == mejor_label, 255, 0).astype(np.uint8)
        else:
            mascara_aislada = bin_mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mascara_aislada = cv2.morphologyEx(mascara_aislada, cv2.MORPH_CLOSE, kernel)

    mascara_final = cv2.resize(mascara_aislada, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    return cv2.GaussianBlur(mascara_final, (7, 7), 0)


class RenaultDegradeNode:
    # RANGOS CALIBRADOS PARA MUESTREO ALEATORIO (Sombra y vividez ajustadas)
    RANGOS = {
        "dirt_intensity": (0.20, 0.50),
        "wear_fading": (0.28, 0.55),      # Mínimo más alto para drenar brillo vectorial; máximo moderado
        "scratch_level": (0.10, 0.35),
        "shadow_strength": (0.26, 0.52),  # Caída de luz pronunciada sin cegar la señal
        "noise_level": (16.0, 32.0),
        "blur_amount": (1.2, 2.2),
        "jpeg_quality": (40, 68),
    }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 12345, "min": 0, "max": 0xffffffffffffffff}),
                "autodetect_sign": ("BOOLEAN", {"default": True}),
                "randomize_params": ("BOOLEAN", {"default": False}),
                "dirt_intensity": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
                "wear_fading": ("FLOAT", {"default": 0.40, "min": 0.0, "max": 1.0, "step": 0.05}),
                "scratch_level": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.05}),
                "shadow_strength": ("FLOAT", {"default": 0.38, "min": 0.0, "max": 1.0, "step": 0.05}),
                "noise_level": ("FLOAT", {"default": 22.0, "min": 0.0, "max": 60.0, "step": 1.0}),
                "blur_amount": ("FLOAT", {"default": 1.6, "min": 0.0, "max": 5.0, "step": 0.1}),
                "jpeg_quality": ("INT", {"default": 55, "min": 20, "max": 100, "step": 5}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "aplicar"
    CATEGORY = "Renault"

    def aplicar(self, image, seed, autodetect_sign, randomize_params, dirt_intensity, wear_fading, scratch_level, shadow_strength, noise_level, blur_amount, jpeg_quality, mask=None):
        # Aseguramos determinismo con la semilla del nodo
        rng = random.Random(seed)

        # Si el botón está activo, fluctuar dentro de los rangos definidos
        if randomize_params:
            dirt_intensity = round(rng.uniform(*self.RANGOS["dirt_intensity"]), 2)
            wear_fading = round(rng.uniform(*self.RANGOS["wear_fading"]), 2)
            scratch_level = round(rng.uniform(*self.RANGOS["scratch_level"]), 2)
            shadow_strength = round(rng.uniform(*self.RANGOS["shadow_strength"]), 2)
            noise_level = round(rng.uniform(*self.RANGOS["noise_level"]), 1)
            blur_amount = round(rng.uniform(*self.RANGOS["blur_amount"]), 2)
            jpeg_quality = rng.randint(*self.RANGOS["jpeg_quality"])

            print(f"\n[RenaultDegrade] Muestreo aleatorio (Seed: {seed}):")
            print(f"  > dirt: {dirt_intensity} | wear: {wear_fading} | scratch: {scratch_level}")
            print(f"  > shadow: {shadow_strength} | noise: {noise_level} | blur: {blur_amount} | jpeg: {jpeg_quality}")

        batch = image.cpu().numpy()
        out_batch = []
        masks_out = []

        masks_np = mask.cpu().numpy() if mask is not None else None

        for idx, img in enumerate(batch):
            img_bgr = (img * 255.0).astype(np.uint8)[:, :, ::-1]
            h, w, _ = img_bgr.shape

            # 1. Máscara: externa o autodetección OpenCV
            if masks_np is not None:
                m = masks_np[idx] if len(masks_np) > idx else masks_np[0]
                if m.shape[:2] != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
                m_final = cv2.GaussianBlur(m, (5, 5), 0)
            elif autodetect_sign:
                m_final = extraer_mascara_central_opencv(img_bgr)
            else:
                m_final = np.ones((h, w), dtype=np.float32)

            # 2. Aplicar degradación con los parámetros activos (manuales o fluctuados)
            deg_bgr = aplicar_degradacion_completa(
                img_bgr,
                seed=seed,
                dirt_intensity=dirt_intensity,
                wear_fading=wear_fading,
                scratch_level=scratch_level,
                shadow_strength=shadow_strength,
                noise_level=noise_level,
                blur_amount=blur_amount,
                jpeg_quality=jpeg_quality
            )

            # 3. Composición restringida a la silueta
            m_3ch = np.repeat(m_final[:, :, np.newaxis], 3, axis=2)
            final_bgr = (deg_bgr * m_3ch + img_bgr * (1.0 - m_3ch)).astype(np.uint8)

            img_rgb = final_bgr[:, :, ::-1].astype(np.float32) / 255.0
            out_batch.append(img_rgb)
            masks_out.append(m_final)

        tensor_img = torch.from_numpy(np.array(out_batch))
        tensor_mask = torch.from_numpy(np.array(masks_out))
        
        # Diccionario con los parámetros aplicados para el visor de la UI
        params_info = {
            "dirt_intensity": dirt_intensity,
            "wear_fading": wear_fading,
            "scratch_level": scratch_level,
            "shadow_strength": shadow_strength,
            "noise_level": noise_level,
            "blur_amount": blur_amount,
            "jpeg_quality": jpeg_quality,
        }

        return {
            "ui": {
                "params_info": [params_info]
            },
            "result": (tensor_img, tensor_mask)
        }
