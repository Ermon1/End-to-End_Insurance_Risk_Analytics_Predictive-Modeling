import sys
import traceback
from typing import Optional, Dict


class MLException(Exception):
    """
    Base exception for all ML pipeline failures.
    """

    def __init__(
        self,
        message: str,
        error: Optional[Exception] = None,
        metadata: Optional[Dict] = None
    ):
        self.message = message
        self.error = error
        self.metadata = metadata or {}

        # --- Extract traceback info ---
        _, _, tb = sys.exc_info()

        if tb:
            last_tb = traceback.extract_tb(tb)[-1]
            self.file_name = last_tb.filename
            self.line_no = last_tb.lineno
            self.function_name = last_tb.name
        else:
            self.file_name = "Unknown"
            self.line_no = "Unknown"
            self.function_name = "Unknown"

        super().__init__(self.__str__())

    def __str__(self):
        base = (
            f"[MLException] {self.message}\n"
            f"Location: {self.file_name}:{self.line_no} (function: {self.function_name})"
        )

        if self.metadata:
            base += f"\nMetadata: {self.metadata}"

        if self.error:
            base += f"\nCause: {repr(self.error)}"

        return base

    def to_dict(self):
        return {
            "message": self.message,
            "file": self.file_name,
            "line": self.line_no,
            "function": self.function_name,
            "metadata": self.metadata,
            "cause": repr(self.error) if self.error else None
        }
