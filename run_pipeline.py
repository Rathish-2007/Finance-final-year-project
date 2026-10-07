import time

from django.core.management.base import BaseCommand

from core.models import ModuleState
from core.services import pipeline


class Command(BaseCommand):
    help = "Run the full RASTA-QF pipeline headlessly."

    def add_arguments(self, parser):
        parser.add_argument("--modules", nargs="+", default=None)

    def handle(self, *args, **opts):
        mods = opts["modules"] or pipeline.MODULE_ORDER
        pipeline.run_pipeline(mods, single_thread=True)
        for m in mods:
            st = ModuleState.objects.get(name=m)
            self.stdout.write(f"{m:<16} {st.status:<8} {st.message}")
