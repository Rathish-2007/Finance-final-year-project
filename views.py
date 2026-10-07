import json

from django.http import JsonResponse, FileResponse, Http404
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .models import ModuleState, RunLog, MODULES
from .services import pipeline

NAV = [
    ("dashboard", "/", "Dashboard", "◈"),
    ("data", "/data", "M1 · Data Lake", "⬢"),
    ("sscdv", "/sscdv", "M2 · SSCDV", "✦"),
    ("regimes", "/regimes", "M3 · Regimes (CJM)", "◉"),
    ("connectedness", "/connectedness", "M4 · TVP-VAR + QQC", "⬡"),
    ("risk", "/risk", "M5 · Risk Lab", "⚠"),
    ("drl", "/drl", "M6 · DRL Trader", "▲"),
    ("benchmark", "/benchmark", "Benchmark & Paper", "❖"),
]


def _ctx(request, active, title):
    states = {s.name: s for s in ModuleState.objects.all()}
    return {
        "nav": NAV, "active": active, "title": title,
        "states": states,
    }


def dashboard(request):
    ctx = _ctx(request, "dashboard", "Mission Control")
    ctx["logs"] = RunLog.objects.order_by("-ts")[:14]
    ctx["module_list"] = MODULES
    ctx["module_json"] = json.dumps([m for m, _ in MODULES])
    return render(request, "dashboard.html", ctx)


def data_page(request):
    return render(request, "data.html", _ctx(request, "data", "Unified Data Lake"))


def sscdv_page(request):
    return render(request, "sscdv.html", _ctx(request, "sscdv", "SSCDV Embeddings"))


def regimes_page(request):
    return render(request, "regimes.html", _ctx(request, "regimes", "Continuous Jump Model"))


def connectedness_page(request):
    return render(request, "connectedness.html",
                  _ctx(request, "connectedness", "TVP-VAR + QQC Connectedness"))


def risk_page(request):
    return render(request, "risk.html", _ctx(request, "risk", "Threshold & Stability Lab"))


def drl_page(request):
    return render(request, "drl.html", _ctx(request, "drl", "Hierarchical DRL Trader"))


def benchmark_page(request):
    return render(request, "benchmark.html",
                  _ctx(request, "benchmark", "Benchmark & Paper Mapping"))


# ------------------------------------------------------------------ API
def api_status(request):
    out = {}
    for s in ModuleState.objects.all():
        out[s.name] = {"status": s.status, "progress": s.progress,
                       "message": s.message,
                       "updated": s.updated_at.strftime("%H:%M:%S") if s.updated_at else None}
    running = any(v["status"] == "running" for v in out.values())
    return JsonResponse({"modules": out, "running": running})


def api_results(request, module):
    if module not in dict(MODULES):
        raise Http404
    try:
        st = ModuleState.objects.get(name=module)
    except ModuleState.DoesNotExist:
        return JsonResponse({"status": "pending", "results": None})
    return JsonResponse({"status": st.status, "progress": st.progress,
                         "message": st.message, "results": st.results})


def api_logs(request):
    logs = [{"ts": l.ts.strftime("%H:%M:%S"), "module": l.module,
             "level": l.level, "message": l.message}
            for l in RunLog.objects.order_by("-ts")[:40]]
    return JsonResponse({"logs": logs})


@csrf_exempt
@require_http_methods(["POST"])
def api_run(request):
    try:
        body = json.loads(request.body or "{}")
    except json.JSONDecodeError:
        body = {}
    mods = body.get("modules") or ["all"]
    if "all" in mods:
        mods = pipeline.MODULE_ORDER
    running = ModuleState.objects.filter(status="running").exists()
    if running:
        return JsonResponse({"started": False, "reason": "pipeline already running"})
    invalid = [m for m in mods if m not in dict(MODULES)]
    if invalid:
        return JsonResponse({"started": False, "reason": f"unknown modules {invalid}"},
                            status=400)
    pipeline.run_pipeline(mods)
    return JsonResponse({"started": True, "modules": mods})


def benchmark_download(request):
    path, ctype = pipeline.build_benchmark_file()
    return FileResponse(open(path, "rb"), as_attachment=True,
                        filename=path.name, content_type=ctype)
