"""Workflow node definitions (placeholder for DAG execution)."""


class WorkflowNode:
    """Base node for workflow graphs."""

    def run(self) -> None:
        raise NotImplementedError("WorkflowNode is not implemented yet.")
