"""SQLModel + SQLite persistence for simulation data."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlmodel import Field, Session, SQLModel, create_engine, select


class SynthRecord(SQLModel, table=True):
    """Persisted synth profile."""
    id: Optional[int] = Field(default=None, primary_key=True)
    synth_id: str = Field(unique=True, index=True)
    name: str
    simulation_name: str = Field(index=True)
    profile_json: str  # Serialized SynthProfile
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DialogRecord(SQLModel, table=True):
    """Persisted dialog."""
    id: Optional[int] = Field(default=None, primary_key=True)
    dialog_id: str = Field(unique=True, index=True)
    simulation_name: str = Field(index=True)
    tick: int
    participants_json: str
    turns_json: str
    pattern: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)


class EventRecord(SQLModel, table=True):
    """Persisted event."""
    id: Optional[int] = Field(default=None, primary_key=True)
    event_id: str = Field(unique=True, index=True)
    simulation_name: str = Field(index=True)
    tick: int
    event_type: str
    title: str
    description: str
    severity: float
    affected_synths_json: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RelationshipRecord(SQLModel, table=True):
    """Persisted relationship state."""
    id: Optional[int] = Field(default=None, primary_key=True)
    simulation_name: str = Field(index=True)
    synth_a: str = Field(index=True)
    synth_b: str = Field(index=True)
    tick: int
    metrics_json: str
    description: str = ""
    relationship_type: str = ""


class SimulationStore:
    """SQLite-backed persistence layer."""

    def __init__(self, db_path: str = "strangeloop.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        SQLModel.metadata.create_all(self.engine)

    def save_synth(self, simulation_name: str, synth_id: str, name: str, profile_json: str):
        with Session(self.engine) as session:
            existing = session.exec(
                select(SynthRecord).where(SynthRecord.synth_id == synth_id)
            ).first()
            if existing:
                existing.name = name
                existing.profile_json = profile_json
                existing.simulation_name = simulation_name
            else:
                session.add(SynthRecord(
                    synth_id=synth_id, name=name,
                    simulation_name=simulation_name, profile_json=profile_json,
                ))
            session.commit()

    def save_dialog(self, simulation_name: str, dialog_id: str, tick: int,
                    participants: list[str], turns_json: str, pattern: str = ""):
        with Session(self.engine) as session:
            record = DialogRecord(
                dialog_id=dialog_id,
                simulation_name=simulation_name,
                tick=tick,
                participants_json=json.dumps(participants),
                turns_json=turns_json,
                pattern=pattern,
            )
            session.add(record)
            session.commit()

    def save_event(self, simulation_name: str, event_id: str, tick: int,
                   event_type: str, title: str, description: str,
                   severity: float, affected_synths: list[str]):
        with Session(self.engine) as session:
            record = EventRecord(
                event_id=event_id,
                simulation_name=simulation_name,
                tick=tick,
                event_type=event_type,
                title=title,
                description=description,
                severity=severity,
                affected_synths_json=json.dumps(affected_synths),
            )
            session.add(record)
            session.commit()

    def save_relationship(self, simulation_name: str, synth_a: str, synth_b: str,
                          tick: int, metrics_json: str, description: str = "",
                          relationship_type: str = ""):
        with Session(self.engine) as session:
            record = RelationshipRecord(
                simulation_name=simulation_name,
                synth_a=synth_a,
                synth_b=synth_b,
                tick=tick,
                metrics_json=metrics_json,
                description=description,
                relationship_type=relationship_type,
            )
            session.add(record)
            session.commit()

    def get_synths(self, simulation_name: str) -> list[SynthRecord]:
        with Session(self.engine) as session:
            return list(session.exec(
                select(SynthRecord).where(SynthRecord.simulation_name == simulation_name)
            ).all())

    def get_dialogs(self, simulation_name: str, tick: Optional[int] = None) -> list[DialogRecord]:
        with Session(self.engine) as session:
            stmt = select(DialogRecord).where(DialogRecord.simulation_name == simulation_name)
            if tick is not None:
                stmt = stmt.where(DialogRecord.tick == tick)
            return list(session.exec(stmt).all())

    def get_events(self, simulation_name: str, tick: Optional[int] = None) -> list[EventRecord]:
        with Session(self.engine) as session:
            stmt = select(EventRecord).where(EventRecord.simulation_name == simulation_name)
            if tick is not None:
                stmt = stmt.where(EventRecord.tick == tick)
            return list(session.exec(stmt).all())
