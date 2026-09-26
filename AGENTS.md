# AGENTS.md

Instructions for AI agents working in this repo. Read before editing.

## What this is
media-tools is an internal HTTP service the h4ks n8n workflows call inside the cluster: it cuts objects out of pictures, turns them into pixel art and sprite sheets, draws pose skeletons, shrinks 3D models for browsers and retargets skeleton animations onto rigged humanoid models. Only n8n may reach it, and it runs with no network access of its own, so its routes take files as request bodies and never fetch anything.

## Where things live
- `src/media_tools/app.py`: the HTTP routes. Each POST takes raw file bytes as its body; several files go back to back with their byte lengths in the `lengths` query, since n8n sends one binary body per request.
- `pictures.py`: cutout (ISNet model or flat-colour key), pixel art, sprite sheets and GIFs.
- `poses.py`: preset OpenPose skeletons for side-view sprite animation.
- `mesh.py`: gltfpack simplification and opaque materials.
- `retarget.py`: moves skeleton animations (Kimodo SOMA or a humanoid rig) onto a rigged humanoid GLB as named clips.
- `glb.py`: reading and writing binary glTF.
- `tests/` mirrors the modules; `tests/data` holds two Kimodo motions and a skinned character as gltfpack writes it.

## Commands (Makefile is SSoT)
- `make install` uv sync plus the prek git hooks
- `make quality` the full gate: format, lint, types, imports, dead code, unused deps, security, audit, coverage, build
- `make run` serve the app with reload on port 8080

## Rules
- Run `make quality` before considering work complete; never weaken a check or lower coverage.
- Routes never run code, commands or paths that come from a request; they only pass validated numbers and files to our own functions and to gltfpack.
- The image bakes gltfpack and the ISNet model at pinned checksums, so the pod needs no network. Keep new tools pinned the same way.
- Keep the route contracts stable: the n8n workflows call them with these exact paths and query parameters.
- Code style as in the workflows repo: strict mypy, no `Any` outside glTF documents, imports at the top, comments only for a non-obvious why.
