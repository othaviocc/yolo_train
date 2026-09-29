from types import SimpleNamespace

from hydrone_vision.gesture_core import classify


def pose(left_arm, right_arm):
    landmarks = [SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=1.0)
                 for _ in range(33)]
    for shoulder, elbow, wrist, points in (
            (11, 13, 15, left_arm), (12, 14, 16, right_arm)):
        landmarks[shoulder], landmarks[elbow], landmarks[wrist] = [
            SimpleNamespace(x=x, y=y, z=0.0, visibility=1.0)
            for x, y in points]
    return landmarks


def test_classifies_both_arms_up_as_climb():
    landmarks = pose(
        ((0.3, 0.5), (0.3, 0.4), (0.3, 0.2)),
        ((0.7, 0.5), (0.7, 0.4), (0.7, 0.2)),
    )

    assert classify(landmarks, 640, 480) == 'SUBIR'


def test_classifies_one_arm_up_and_one_side_as_land():
    landmarks = pose(
        ((0.3, 0.5), (0.3, 0.4), (0.3, 0.2)),
        ((0.7, 0.5), (0.8, 0.5), (0.9, 0.5)),
    )

    assert classify(landmarks, 640, 480) == 'POUSAR'


def test_unreliable_landmarks_do_not_create_a_command():
    landmarks = pose(
        ((0.3, 0.5), (0.3, 0.4), (0.3, 0.2)),
        ((0.7, 0.5), (0.7, 0.4), (0.7, 0.2)),
    )
    landmarks[15].visibility = 0.1

    assert classify(landmarks, 640, 480) == 'NENHUM'