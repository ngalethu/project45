from __future__ import annotations
from collections import deque
from typing import Dict, List
from app.common.constants import EVENT_NO_SEATBELT, EVENT_USING_PHONE
from app.common.types import AlertEvent, CandidateEvent

class AlertManager:
    def __init__(self, config: Dict):
        self.cfg = config
        self.event_types = (EVENT_USING_PHONE, EVENT_NO_SEATBELT)
        self.window_frames = int(self.cfg["rules"].get("temporal_window_frames", 12))
        if not 5 <= self.window_frames <= 15:
            raise ValueError("rules.temporal_window_frames must be between 5 and 15")
        self.histories = {
            event_type: deque(maxlen=self.window_frames) for event_type in self.event_types
        }
        self.cooldown_frames = int(self.cfg["rules"].get("temporal_cooldown_frames", 80))
        # Frame-based cooldown (thay cho time-based last_fired_sec)
        self.last_fired_frame = {
            EVENT_USING_PHONE: -9999,
            EVENT_NO_SEATBELT: -9999,
        }

    def _need_frames(self, event_type: str) -> int:
        mapping = {
            EVENT_USING_PHONE: self.cfg["rules"]["phone_confirm_frames"],
            EVENT_NO_SEATBELT: self.cfg["rules"]["no_seatbelt_confirm_frames"],
        }
        needed = int(mapping[event_type])
        if not 1 <= needed <= self.window_frames:
            raise ValueError(f"Invalid temporal vote requirement for {event_type}: {needed}")
        return needed

    def update(
        self,
        candidates: List[CandidateEvent],
        frame_index: int,
        timestamp_iso: str,
        current_time_sec: float,
        source_device: str,
    ) -> List[AlertEvent]:
        candidate_map: Dict[str, CandidateEvent] = {}
        for c in candidates:
            if c.event_type not in candidate_map or c.score > candidate_map[c.event_type].score:
                candidate_map[c.event_type] = c

        alerts: List[AlertEvent] = []

        for event_type in self.event_types:
            cand = candidate_map.get(event_type)
            history = self.histories[event_type]
            history.append(cand)
            votes = [item for item in history if item is not None]

            if len(votes) >= self._need_frames(event_type):
                if frame_index - self.last_fired_frame[event_type] >= self.cooldown_frames:
                    best = max(votes, key=lambda item: item.score)
                    alerts.append(
                        AlertEvent(
                            event_type=event_type,
                            confidence=best.score,
                            frame_index=frame_index,
                            timestamp=timestamp_iso,
                            bbox=best.bbox,
                            note=f"temporal_votes={len(votes)}/{len(history)}; {best.note}",
                            source_device=source_device,
                        )
                    )
                    self.last_fired_frame[event_type] = frame_index
                    history.clear()
        return alerts
