"""
Management command to seed ExecutionTrace records from data.json.

Usage:
    python manage.py seed_traces                          # uses default frontend/data.json
    python manage.py seed_traces --file /path/to/data.json
    python manage.py seed_traces --clear                  # wipe existing traces first
"""
import json
from pathlib import Path

from django.core.management.base import BaseCommand

from api.models import ExecutionTrace


class Command(BaseCommand):
    help = 'Load journey data from data.json into the ExecutionTrace table'

    def add_arguments(self, parser):
        default_path = Path(__file__).resolve().parent.parent.parent.parent.parent / 'frontend' / 'data.json'
        parser.add_argument(
            '--file',
            default=str(default_path),
            help='Path to data.json (default: frontend/data.json)',
        )
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Delete all existing traces before seeding',
        )

    def handle(self, *args, **options):
        filepath = Path(options['file'])
        if not filepath.exists():
            self.stderr.write(self.style.ERROR(f'File not found: {filepath}'))
            return

        with open(filepath) as f:
            data = json.load(f)

        journeys = data.get('journeys', [])
        if not journeys:
            self.stderr.write(self.style.WARNING('No journeys found in data file'))
            return

        if options['clear']:
            deleted, _ = ExecutionTrace.objects.all().delete()
            self.stdout.write(f'Cleared {deleted} existing records')

        created = 0
        skipped = 0
        for journey in journeys:
            trace_id = journey.get('trace_id', '')
            if not trace_id:
                skipped += 1
                continue

            _, was_created = ExecutionTrace.objects.update_or_create(
                trace_id=trace_id,
                defaults={
                    'journey_name': journey.get('journey_name', ''),
                    'workflow_type': 'imported',
                    'status': 'completed',
                    'raw_journey': journey,
                },
            )
            if was_created:
                created += 1
            else:
                skipped += 1

        self.stdout.write(self.style.SUCCESS(
            f'Seeded {created} traces ({skipped} skipped/updated) from {filepath.name}'
        ))
