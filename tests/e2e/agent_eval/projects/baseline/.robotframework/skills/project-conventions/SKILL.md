---
name: project-conventions
description: Lab project conventions - test naming, tagging and folder layout for rf-lab.
---
# rf-lab conventions
- Every test MUST have at least one tag from: smoke, regression, api, web, calc.
- Web tests live in tests/web and use the Browser library with `headless=${HEADLESS}`.
- API tests use RequestsLibrary sessions named after the service (e.g. `booker`).
- Test names are Title Case, imperative ("Login With Valid User").
