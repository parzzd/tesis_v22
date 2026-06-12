"""Prueba rapida: ¿puede OpenCV leer el stream de la webcam desde MediaMTX?

Uso:
    python test_rtsp_read.py <ruta_o_url_rtsp>

Ejemplos:
    python test_rtsp_read.py 0-yrto4b
    python test_rtsp_read.py rtsp://tesis:tesis@127.0.0.1:8554/0-yrto4b

Mientras este script corre, el navegador debe estar PUBLICANDO la webcam
(boton "Usar mi webcam"). Guarda el primer frame leido en frame_test.jpg.
"""
import os
import sys

# Igual que el backend: fuerza RTSP sobre TCP en el lector FFmpeg de OpenCV.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2  # noqa: E402


def build_url(arg: str) -> str:
    if arg.startswith("rtsp://"):
        return arg
    return f"rtsp://tesis:tesis@127.0.0.1:8554/{arg}"


def main() -> int:
    if len(sys.argv) < 2:
        print("Falta la ruta. Ej: python test_rtsp_read.py 0-yrto4b")
        return 2

    url = build_url(sys.argv[1])
    print(f"Abriendo: {url}")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("FALLO: no se pudo abrir el stream (¿esta el navegador publicando?).")
        return 1

    # Intenta leer hasta 30 frames (los primeros pueden venir vacios).
    for i in range(30):
        ok, frame = cap.read()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            cv2.imwrite("frame_test.jpg", frame)
            print(f"OK: frame {i} leido, resolucion {w}x{h}. Guardado en frame_test.jpg")
            cap.release()
            return 0
    print("FALLO: el stream se abrio pero no se pudo decodificar ningun frame.")
    cap.release()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
