from typing import Any
import insightface

import modules.globals
from modules.typing import Frame
from modules.utilities import resolve_relative_path

FACE_ANALYSER = None

def get_face_analyser() -> Any:
    global FACE_ANALYSER

    if FACE_ANALYSER is None:
        FACE_ANALYSER = insightface.app.FaceAnalysis(
            name='buffalo_l',
            providers=modules.globals.execution_providers
        )
        FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
    return FACE_ANALYSER


def get_one_face(frame: Frame) -> Any:
    face_analyser = get_face_analyser()
    faces = face_analyser.get(frame)
    if faces:
        return faces[0]
    return None


def get_many_faces(frame: Frame) -> Any:
    face_analyser = get_face_analyser()
    faces = face_analyser.get(frame)
    return faces
