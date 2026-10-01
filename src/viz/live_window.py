"""Вікно перегляду в окремому потоці.

Навіщо окремий потік: OpenCV малює вікно тільки всередині
`waitKey()`. Якщо контур керування викликає його рідко (кадри з
Tello приходять нерівно, а при збоях H.264 — майже ніколи), вікно
лишається неотрисованим, і менеджер вікон вважає його зависшим.

Тут показ відв'язаний від темпу кадрів: потік крутить `waitKey`
постійно й малює останнє, що йому дали. Контур просто кладе кадр
у `show()` і не чекає.

Обмеження OpenCV: створювати вікно й викликати `waitKey` треба в
ОДНОМУ потоці, тому `namedWindow` робиться всередині `_loop`.
"""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np


class LiveWindow:
    """Неблокуючий перегляд. Кадр кладуть, потік малює.

    Приклад:
        win = LiveWindow("Tello")
        win.start()
        win.show(frame)
        if win.quit_requested: ...
        win.close()
    """

    def __init__(self, title: str, width: int = 960, height: int = 720,
                 fps: float = 30.0) -> None:
        self.title = title
        self.width, self.height = width, height
        self.period = 1.0 / fps
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._quit = threading.Event()
        self._key: int | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> "LiveWindow":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        time.sleep(0.4)
        return self

    def show(self, frame: np.ndarray) -> None:
        """Віддати кадр на показ. Не блокує."""
        with self._lock:
            self._frame = frame

    @property
    def quit_requested(self) -> bool:
        """True, якщо натиснуто Q або ESC."""
        return self._quit.is_set()

    def take_key(self) -> int | None:
        """Забрати останню натиснуту клавішу (і очистити).

        Клавіші читає потік показу — викликати `waitKey` ще й ззовні
        не можна, OpenCV цього не дозволяє. Тому потік складає сюди,
        а контур забирає.
        """
        with self._lock:
            k, self._key = self._key, None
        return k

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.title, self.width, self.height)
        placeholder = np.full((self.height, self.width, 3), 25, np.uint8)
        cv2.putText(placeholder, "waiting for video...",
                    (self.width // 2 - 190, self.height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120, 120, 120), 2)
        while not self._stop.is_set():
            with self._lock:
                frame = self._frame
            cv2.imshow(self.title, placeholder if frame is None else frame)
            k = cv2.waitKey(max(1, int(self.period * 1000))) & 0xFF
            if k in (27, ord("q")):
                self._quit.set()
            elif k != 255:
                with self._lock:
                    self._key = k
