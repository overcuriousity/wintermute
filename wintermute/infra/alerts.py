"""
Alerting infrastructure — stub for future Gotify integration.

Currently logs alerts at WARNING level.  When Gotify support is added,
``send_gotify_alert`` will push notifications to the configured Gotify
server using ``gotify.url`` and ``gotify.token`` from ``config.yaml``.
"""

import logging

logger = logging.getLogger(__name__)


async def send_gotify_alert(title: str, message: str, priority: int = 5) -> None:
    """Send an alert notification.

    Currently a stub that logs at WARNING level.  Future versions will
    forward to a Gotify server when ``gotify.url`` and ``gotify.token``
    are configured in ``config.yaml``.
    """
    logger.warning("ALERT [priority=%d] %s — %s", priority, title, message)
