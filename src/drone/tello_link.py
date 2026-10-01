"""Зв'язок із DJI Robomaster TT (Tello) через офіційний SDK.

Межа між системою керування й залізом. Контролери оперують
нормованими каналами [-1, 1] (див. `ControlCommand`), а тут вони
перетворюються на rc-команди SDK у [-100, 100] і відправляються
по UDP.

Навмисні рішення:

* `TelloLink` нічого не знає про детектор і позу. Він приймає
  `ControlCommand` і віддає кадри. Це дозволяє ганяти той самий
  контролер у симуляторі й на дроні без змін у коді контролера.
* Watchdog обов'язковий. Tello продовжує виконувати останню
  rc-команду, якщо зв'язок обірвався. Якщо між викликами
  `send()` минуло більше `command_timeout`, канали обнуляються.
* `stream_on` за замовчуванням вимкнений: телеметрію можна читати
  й без відео, а відеопотік з'їдає канал і додає затримку.

Мережа: дрон піднімає власну точку доступу TELLO-XXXXXX,
шлюз 192.168.10.1. Команди — UDP 8889, телеметрія — 8890,
відео — 11111.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from src.control.base import ControlCommand


@dataclass
class Telemetry:
    """Знімок стану дрона."""

    battery: int = -1
    height_cm: int = -1
    tof_cm: int = -1
    temp_c: float = float("nan")
    pitch: int = 0
    roll: int = 0
    yaw: int = 0
    vgx: int = 0
    vgy: int = 0
    vgz: int = 0
    flight_time: int = -1

    @property
    def speed(self) -> float:
        """Модуль горизонтальної швидкості, см/с."""
        return float((self.vgx ** 2 + self.vgy ** 2) ** 0.5)

    def summary(self) -> str:
        return (f"батарея {self.battery}% | висота {self.height_cm} см | "
                f"tof {self.tof_cm} см | темп {self.temp_c:.0f}°C | "
                f"крен {self.roll}° тангаж {self.pitch}° курс {self.yaw}°")


class TelloLink:
    """Тонка обгортка над djitellopy під інтерфейс наших контролерів.

    Приклад:
        link = TelloLink()
        link.connect()
        link.takeoff()
        link.send(ControlCommand(forward=0.3))
        link.land()
    """

    def __init__(self, *, max_channel: int = 60,
                 command_timeout: float = 0.5,
                 min_battery: int = 20) -> None:
        self.max_channel = int(np.clip(max_channel, 10, 100))
        self.command_timeout = command_timeout
        self.min_battery = min_battery
        self._tello = None
        self._flying = False
        self._last_send = 0.0
        self._streaming = False


    def connect(self, *, stream: bool = False) -> Telemetry:
        """Під'єднатися до дрона. Кидає виняток, якщо немає відповіді."""
        from djitellopy import Tello

        self._tello = Tello()
        self._tello.connect()

        tel = self.telemetry()
        if tel.battery < self.min_battery:
            raise RuntimeError(
                f"батарея {tel.battery}% нижче за поріг {self.min_battery}% — "
                "заряди перед польотом")

        if stream:
            self._tello.streamon()
            self._streaming = True
            time.sleep(2.0)
        return tel

    def close(self) -> None:
        """Посадити, якщо в повітрі, і звільнити ресурси."""
        if self._tello is None:
            return
        try:
            if self._flying:
                self.land()
            if self._streaming:
                self._tello.streamoff()
        finally:
            self._tello.end()
            self._tello = None

    def __enter__(self) -> "TelloLink":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


    def telemetry(self) -> Telemetry:
        t = self._require()
        return Telemetry(
            battery=t.get_battery(),
            height_cm=t.get_height(),
            tof_cm=t.get_distance_tof(),
            temp_c=t.get_temperature(),
            pitch=t.get_pitch(),
            roll=t.get_roll(),
            yaw=t.get_yaw(),
            vgx=t.get_speed_x(),
            vgy=t.get_speed_y(),
            vgz=t.get_speed_z(),
            flight_time=t.get_flight_time(),
        )


    def takeoff(self) -> None:
        self._require().takeoff()
        self._flying = True
        self._last_send = time.time()

    def land(self, attempts: int = 4) -> bool:
        """Посадити. НЕ кидає виняток: дрон у повітрі важливіший за помилку.

        Tello відповідає 'error' на 'land', коли зайнятий (типово —
        забитий канал від відеопотоку, що сипеться). Команда при цьому
        часто вже прийнята. Тому: кілька спроб із паузою, а якщо не
        вийшло — тримаємо нульові канали, щоб дрон висів, і повертаємо
        False. Викликач вирішує, чи вдаватися до emergency().
        """
        t = self._require()
        for i in range(attempts):
            try:
                t.send_rc_control(0, 0, 0, 0)
                t.land()
                self._flying = False
                return True
            except Exception as e:
                if i == attempts - 1:
                    print(f"land не вдався ({e}); дрон висить — "
                          "сідай з пульта або викликай emergency()")
                    try:
                        t.send_rc_control(0, 0, 0, 0)
                    except Exception:
                        pass
                    return False
                time.sleep(0.7)
        return False

    def emergency(self) -> None:
        """Негайна зупинка двигунів. Дрон падає — тільки як аварійний вихід."""
        self._require().emergency()
        self._flying = False

    def hover(self) -> None:
        self.send(ControlCommand())

    def send(self, cmd: ControlCommand) -> tuple[int, int, int, int]:
        """Відправити команду керування. Повертає фактичні rc-канали."""
        t = self._require()
        scale = self.max_channel / 100.0
        a, b, c, d = cmd.clip().to_rc()
        rc = (int(round(a * scale)), int(round(b * scale)),
              int(round(c * scale)), int(round(d * scale)))
        t.send_rc_control(*rc)
        self._last_send = time.time()
        return rc

    def watchdog(self) -> bool:
        """Обнулити канали, якщо команд давно не було. True — спрацював.

        Викликати в контурі керування на кожній ітерації: якщо детектор
        затнувся або кадр не прийшов, дрон інакше полетить далі за
        останньою командою.
        """
        if not self._flying:
            return False
        if time.time() - self._last_send > self.command_timeout:
            self.hover()
            return True
        return False


    def frame(self) -> np.ndarray | None:
        """Останній кадр BGR або None, якщо потік ще не піднявся."""
        if not self._streaming:
            raise RuntimeError("потік вимкнено: connect(stream=True)")
        f = self._require().get_frame_read().frame
        if f is None:
            return None
        return f[:, :, ::-1].copy()


    def _require(self):
        if self._tello is None:
            raise RuntimeError("немає з'єднання: спершу connect()")
        return self._tello

    @property
    def flying(self) -> bool:
        return self._flying
