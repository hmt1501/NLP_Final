"""
Professional Logger with Rich Console and File Logging

This module provides a centralized logging solution that:
- Beautiful console output using Rich
- Persistent file logging with rotation (unlimited size)
- Consistent formatting across the entire codebase
- Thread-safe logging

Usage:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Training started")
    logger.error("Something went wrong", exc_info=True)
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# Custom theme for Rich console
CUSTOM_THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "critical": "bold white on red",
    "debug": "dim",
    "success": "bold green",
})

# Global console instance
console = Console(theme=CUSTOM_THEME)

# Log file configuration
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_FILE = LOG_DIR / "training.log"

# Track if root logger has been configured
_logger_initialized = False


def setup_log_directory() -> None:
    """Create log directory if it doesn't exist."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_file_handler(log_file: Optional[Path] = None) -> logging.FileHandler:
    """
    Create a file handler for logging.
    
    Uses a standard FileHandler without size limits (as requested).
    Logs are rotated by date in the filename if needed.
    
    Args:
        log_file: Path to log file. Defaults to LOG_FILE.
        
    Returns:
        Configured FileHandler instance.
    """
    setup_log_directory()
    
    if log_file is None:
        log_file = LOG_FILE
    
    # Create file handler (append mode, no size limit)
    handler = logging.FileHandler(
        log_file,
        mode='a',  # Append mode
        encoding='utf-8'
    )
    
    # Detailed format for file logging
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)  # Log everything to file
    
    return handler


def get_rich_handler() -> RichHandler:
    """
    Create a Rich handler for beautiful console output.
    
    Returns:
        Configured RichHandler instance.
    """
    handler = RichHandler(
        console=console,
        show_time=True,
        show_level=True,
        show_path=True,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        markup=True,
        log_time_format="[%X]",
    )
    handler.setLevel(logging.INFO)  # Console shows INFO and above
    
    return handler


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Get a configured logger instance.
    
    This is the main entry point for getting a logger. All modules should use:
        from utils.logger import get_logger
        logger = get_logger(__name__)
    
    Args:
        name: Name of the logger (typically __name__)
        level: Logging level (default: DEBUG)
        
    Returns:
        Configured Logger instance with Rich console and file handlers.
    """
    global _logger_initialized
    
    logger = logging.getLogger(name)
    
    # Only configure root logger handlers once
    if not _logger_initialized:
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        root_logger.handlers.clear()
        
        # Add Rich console handler
        root_logger.addHandler(get_rich_handler())
        
        # Add file handler
        root_logger.addHandler(get_file_handler())
        
        _logger_initialized = True
        
        # Log initialization
        init_logger = logging.getLogger("utils.logger")
        init_logger.info(f"Logger initialized. Log file: {LOG_FILE}")
    
    logger.setLevel(level)
    
    return logger


def log_separator(logger: logging.Logger, title: str = "", char: str = "=", length: int = 70) -> None:
    """
    Log a visual separator line.
    
    Args:
        logger: Logger instance to use
        title: Optional title to embed in separator
        char: Character to use for separator
        length: Length of separator line
    """
    if title:
        padding = (length - len(title) - 2) // 2
        line = f"{char * padding} {title} {char * padding}"
        # Adjust for odd lengths
        if len(line) < length:
            line += char
    else:
        line = char * length
    
    logger.info(line)


def log_dict(logger: logging.Logger, data: dict, title: str = "Configuration") -> None:
    """
    Log a dictionary in a formatted way.
    
    Args:
        logger: Logger instance to use
        data: Dictionary to log
        title: Title for the log section
    """
    logger.info(f"[bold]{title}:[/bold]")
    for key, value in data.items():
        logger.info(f"  {key}: {value}")


def log_metrics(logger: logging.Logger, metrics: dict, epoch: Optional[int] = None) -> None:
    """
    Log training/evaluation metrics in a formatted way.
    
    Args:
        logger: Logger instance to use
        metrics: Dictionary of metric names to values
        epoch: Optional epoch number
    """
    if epoch is not None:
        logger.info(f"[bold cyan]Metrics (Epoch {epoch}):[/bold cyan]")
    else:
        logger.info("[bold cyan]Metrics:[/bold cyan]")
    
    for name, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {name}: {value:.4f}")
        else:
            logger.info(f"  {name}: {value}")


# Convenience function to get console for Rich-specific operations
def get_console() -> Console:
    """
    Get the global Rich Console instance.
    
    Use this for Rich-specific features like tables, panels, progress bars, etc.
    
    Returns:
        Global Console instance with custom theme.
    """
    return console


# Test the logger if run directly
if __name__ == "__main__":
    logger = get_logger("test_logger")
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    log_separator(logger, "Test Section")
    
    log_dict(logger, {
        "model": "Qwen/Qwen3-1.7B",
        "learning_rate": 5e-4,
        "batch_size": 4,
    })
    
    log_metrics(logger, {
        "loss": 0.1234,
        "rouge_l": 45.67,
        "bertscore": 78.90,
    }, epoch=1)
    
    print(f"\nLog file created at: {LOG_FILE}")
