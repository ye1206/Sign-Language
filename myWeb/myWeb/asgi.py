import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
import webs.routing

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myWeb.settings')

application = ProtocolTypeRouter({
    "http": get_asgi_application(),  # HTTP request
    "websocket": AuthMiddlewareStack(
        URLRouter(
            webs.routing.websocket_urlpatterns  # WebSocket
        )
    ),
})
