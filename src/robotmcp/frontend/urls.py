from django.urls import path, re_path
from django.conf import settings
from django.views.static import serve as static_serve

from . import api, views

app_name = "robotmcp_frontend"

urlpatterns = [
    path("", views.index, name="index"),
    path("api/sessions/", api.sessions_list, name="api-sessions-list"),
    path("api/sessions/<str:session_id>/", api.session_detail, name="api-session-detail"),
    path("api/sessions/<str:session_id>/steps/", api.session_steps, name="api-session-steps"),
    path(
        "api/sessions/<str:session_id>/variables/",
        api.session_variables,
        name="api-session-variables",
    ),
    path(
        "api/sessions/<str:session_id>/state/",
        api.session_state,
        name="api-session-state",
    ),
    path(
        "api/sessions/<str:session_id>/suite/",
        api.suite_preview,
        name="api-suite-preview",
    ),
    path(
        "api/sessions/<str:session_id>/execute/",
        api.execute_keyword,
        name="api-execute-keyword",
    ),
    path("api/events/", api.events_stream, name="api-events-stream"),
    path("api/events/recent/", api.recent_events, name="api-events-recent"),
]

base_prefix = getattr(settings, "ROBOTMCP_FRONTEND_BASE_PATH", "/").strip("/")
static_pattern = r"^static/(?P<path>.*)$" if not base_prefix else rf"^{base_prefix}static/(?P<path>.*)$"

if settings.STATICFILES_DIRS:
    urlpatterns.append(
        re_path(
            static_pattern,
            static_serve,
            {"document_root": settings.STATICFILES_DIRS[0]},
        )
    )
