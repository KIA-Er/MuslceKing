from operator import add
from typing import Any, Dict, List, Optional

try:  # pragma: no cover - prefer typing_extensions for Pydantic compatibility
    from typing_extensions import Annotated, TypedDict  # type: ignore
except ImportError:  # pragma: no cover - minimal stdlib fallback
    from typing import Annotated, TypedDict

from langchain_core.messages import ToolCall
from pydantic import BaseModel, Field

from muscleking.app.agents.cyper_tools.cypher_utils import CypherOutputState
from muscleking.app.agents.models.visual_state import VisualizationOutputState


class CypherHistoryRecord(TypedDict):
    """A simplified representation of the CypherOutputState"""

    task: str
    statement: str
    records: List[Dict[str, Any]]


class HistoryRecord(TypedDict):
    """Information that may be relevant to future user questions."""

    question: str
    answer: str
    cyphers: List[CypherHistoryRecord]


def update_history(
    history: List[HistoryRecord], new: List[HistoryRecord]
) -> List[HistoryRecord]:
    """
    Update the history record. Allow only a max number of records to be stored at any time.

    Parameters
    ----------
    history : List[HistoryRecord]
        The current history list.
    new : List[HistoryRecord]
        The new record to add. Should be a single entry list.

    Returns
    -------
    List[HistoryRecord]
        A new List with the record added and old records removed to maintain size.
    """

    SIZE: int = 5

    history.extend(new)
    return history[-SIZE:]


class Task(BaseModel):
    question: str = Field(..., description="The question to be addressed.")
    parent_task: str = Field(
        ..., description="The parent task this task is derived from."
    )
    requires_visualization: bool = Field(
        default=False,
        description="Whether this task requires a visual to be returned.",
    )
    data: Optional[CypherOutputState] = Field(
        default=None, description="The Cypher query result details."
    )
    visualization: Optional[VisualizationOutputState] = Field(
        default=None, description="The visualization details."
    )

    @property
    def is_complete(self) -> bool:
        viz_bool = (self.requires_visualization and self.visualization is not None) or (
            not self.requires_visualization and self.visualization is None
        )
        return viz_bool and self.data is not None


class InputState(TypedDict, total=False):
    """The input state for multi agent workflows."""

    question: str
    data: List[Dict[str, Any]]
    history: Annotated[List[HistoryRecord], update_history]
    route_type: Optional[str]


class OverallState(TypedDict, total=False):
    """The main state in multi agent workflows."""

    question: str
    tasks: Annotated[List[Task], add]
    next_action: str
    cyphers: Annotated[List[CypherOutputState], add]
    summary: str
    steps: Annotated[List[str], add]
    history: Annotated[List[HistoryRecord], update_history]
    route_type: Optional[str]


class OutputState(TypedDict, total=False):
    """The final output for multi agent workflows."""

    answer: str
    question: str
    steps: List[str]
    cyphers: List[CypherOutputState]
    visualizations: List[VisualizationOutputState]
    history: Annotated[List[HistoryRecord], update_history]


class TaskState(TypedDict):
    """The state of a task."""

    question: str
    parent_task: str
    requires_visualization: bool
    data: CypherOutputState
    visualization: VisualizationOutputState


class PredefinedCypherInputState(TypedDict):
    """The input state for a predefined Cypher node."""

    task: str
    query_name: str
    query_parameters: Dict[str, Any]
    steps: List[str]


class ToolSelectionInputState(TypedDict):
    """The input state for the Tool Selection node."""

    question: str
    parent_task: str
    context: Any


class ToolSelectionOutputState(TypedDict):
    tool_selection_task: str
    tool_call: Optional[ToolCall]
    steps: List[str]


class ToolSelectionErrorState(TypedDict):
    """The input state to the tool selection error handling node."""

    task: str
    errors: List[str]
    steps: List[str]
