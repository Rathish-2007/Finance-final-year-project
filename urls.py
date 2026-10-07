from django.urls import path
from . import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("data", views.data_page, name="data"),
    path("sscdv", views.sscdv_page, name="sscdv"),
    path("regimes", views.regimes_page, name="regimes"),
    path("connectedness", views.connectedness_page, name="connectedness"),
    path("risk", views.risk_page, name="risk"),
    path("drl", views.drl_page, name="drl"),
    path("benchmark", views.benchmark_page, name="benchmark"),
    path("api/status", views.api_status, name="api_status"),
    path("api/results/<str:module>", views.api_results, name="api_results"),
    path("api/logs", views.api_logs, name="api_logs"),
    path("api/run", views.api_run, name="api_run"),
    path("api/benchmark/download", views.benchmark_download, name="benchmark_download"),
]
