from __future__ import annotations

import io
import math
import unicodedata
import zipfile
from pathlib import Path
from urllib.parse import urlencode

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from photo_recognizer.config import load_settings
from photo_recognizer.db import FaceRepository
from photo_recognizer.services import (
    IndexService,
    SearchService,
    build_indexed_face_label,
)


def create_app() -> FastAPI:
    settings = load_settings()
    repository = FaceRepository(settings.db_path)
    repository.init_db()

    index_service = IndexService(settings, repository)
    search_service = SearchService(settings, repository)

    app = FastAPI(title="Photo Recognizer", version="0.2.0")
    templates = Jinja2Templates(directory=str(settings.base_dir / "templates"))

    app.state.settings = settings
    app.state.repository = repository
    app.state.index_service = index_service
    app.state.search_service = search_service
    app.state.templates = templates

    app.mount("/static", StaticFiles(directory=str(settings.base_dir / "static")), name="static")
    app.mount("/dataset", StaticFiles(directory=str(settings.images_dir)), name="dataset")
    app.mount("/storage", StaticFiles(directory=str(settings.storage_dir)), name="storage")

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request) -> HTMLResponse:
        events = [decorate_event_row(row, request) for row in repository.list_events()]
        return templates.TemplateResponse(
            request,
            "event_select.html",
            {
                "events": events,
                "has_events": bool(events),
                "library_url": str(request.url_for("library")),
            },
        )

    @app.get("/library", response_class=HTMLResponse)
    async def library(request: Request, page: int = Query(1, ge=1)) -> HTMLResponse:
        return render_library_page(
            request=request,
            repository=repository,
            event_id=None,
            page=page,
        )

    @app.get("/events/{event_id}", response_class=HTMLResponse)
    async def event_detail(request: Request, event_id: int, page: int = Query(1, ge=1)) -> HTMLResponse:
        event = repository.get_event(event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="Event not found")
        return render_library_page(
            request=request,
            repository=repository,
            event_id=event_id,
            page=page,
        )

    @app.post("/events")
    async def create_event(
        request: Request,
        name: str = Form(...),
        event_date: str = Form(""),
    ) -> RedirectResponse:
        event_id = repository.create_event(name=name, event_date=event_date)
        return RedirectResponse(
            url=str(request.url_for("event_detail", event_id=event_id)),
            status_code=303,
        )

    @app.post("/events/{event_id}/rename")
    async def rename_event(
        event_id: int,
        name: str = Form(...),
        event_date: str = Form(""),
        redirect_to: str = Form("/"),
    ) -> RedirectResponse:
        try:
            repository.rename_event(event_id=event_id, name=name, event_date=event_date)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return RedirectResponse(url=redirect_to, status_code=303)

    @app.post("/events/{event_id}/delete")
    async def delete_event(
        event_id: int,
        request: Request,
        redirect_to: str = Form(""),
    ) -> RedirectResponse:
        try:
            repository.delete_event(event_id=event_id)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        target = redirect_to or str(request.url_for("home"))
        return RedirectResponse(url=target, status_code=303)

    @app.post("/events/import")
    async def import_event_photos(
        request: Request,
        files: list[UploadFile] = File(...),
        event_id: int | None = Form(None),
        new_event_name: str = Form(""),
        new_event_date: str = Form(""),
    ) -> RedirectResponse:
        uploaded_files: list[tuple[str, bytes]] = []
        for file in files:
            content = await file.read()
            uploaded_files.append((file.filename or "image.jpg", content))

        target_event_id = event_id
        if target_event_id is None:
            normalized_name = new_event_name.strip()
            if not normalized_name:
                raise HTTPException(status_code=400, detail="Event name is required")
            target_event_id = repository.create_event(
                name=normalized_name,
                event_date=new_event_date.strip(),
            )

        summary = index_service.import_uploaded_files(target_event_id, uploaded_files)
        if summary["photos_indexed"] == 0:
            raise HTTPException(status_code=400, detail="No supported images were uploaded")

        return RedirectResponse(
            url=str(request.url_for("event_detail", event_id=target_event_id)),
            status_code=303,
        )

    @app.get("/search/face", response_class=HTMLResponse)
    async def search_face_by_id(
        request: Request,
        face_id: int = Query(..., ge=1),
        event_id: int | None = Query(None),
    ) -> RedirectResponse:
        query = build_query_string(event_id=event_id)
        base = str(request.url_for("search_indexed_face", face_id=face_id))
        suffix = f"?{query}" if query else ""
        return RedirectResponse(url=f"{base}{suffix}", status_code=303)

    @app.get("/search/person", response_class=HTMLResponse)
    async def search_person_by_id(
        request: Request,
        person_id: int = Query(..., ge=1),
        event_id: int | None = Query(None),
    ) -> RedirectResponse:
        query = build_query_string(event_id=event_id)
        base = str(request.url_for("person_detail", person_id=person_id))
        suffix = f"?{query}" if query else ""
        return RedirectResponse(url=f"{base}{suffix}", status_code=303)

    @app.get("/photo/{photo_id}", response_class=HTMLResponse)
    async def photo_detail(
        request: Request,
        photo_id: int,
        event_id: int | None = Query(None),
    ) -> HTMLResponse:
        photo = repository.get_photo(photo_id)
        if photo is None:
            raise HTTPException(status_code=404, detail="Photo not found")

        current_event_id = event_id if event_id is not None else int(photo["event_id"])
        faces = [
            decorate_face_row(face, int(photo["width"]), int(photo["height"]), request, current_event_id)
            for face in repository.get_photo_faces(photo_id)
        ]
        return templates.TemplateResponse(
            request,
            "photo_detail.html",
            {
                "photo": decorate_photo_row(photo, request),
                "faces": faces,
                "all_persons": decorate_person_rows(repository.list_persons(), request, current_event_id),
                "scope": build_scope_context(request, repository, current_event_id),
                "download_url": str(request.url_for("download_photo", photo_id=photo_id)),
            },
        )

    @app.get("/face/{face_id}/search", response_class=HTMLResponse)
    async def search_indexed_face(
        request: Request,
        face_id: int,
        event_id: int | None = Query(None),
    ) -> HTMLResponse:
        search_result = search_service.search_indexed_face(face_id, event_id=event_id)
        if search_result is None:
            raise HTTPException(status_code=404, detail="Face not found")

        return render_search_results(
            request=request,
            repository=repository,
            search_result=search_result,
            query_image_url=dataset_url(request, search_result["query_path"]),
            query_context=f"Поиск похожих для {search_result['query_faces'][0]['label']}",
            source_photo_id=search_result["source_photo_id"],
            event_id=event_id,
        )

    @app.post("/search", response_class=HTMLResponse)
    async def search(
        request: Request,
        file: UploadFile = File(...),
        event_id: int | None = Form(None),
    ) -> HTMLResponse:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        stored_path = search_service.store_query_file(file.filename or "query.jpg", content)
        search_result = search_service.search(stored_path, event_id=event_id)

        return render_search_results(
            request=request,
            repository=repository,
            search_result=search_result,
            query_image_url=storage_url(request, search_result["query_path"]),
            query_context=build_scope_search_caption(repository, event_id),
            source_photo_id=None,
            event_id=event_id,
        )

    @app.post("/search/folder", response_class=HTMLResponse)
    async def search_folder(
        request: Request,
        files: list[UploadFile] = File(...),
        event_id: int | None = Form(None),
    ) -> HTMLResponse:
        uploaded_files: list[tuple[str, bytes]] = []
        for file in files:
            content = await file.read()
            uploaded_files.append((file.filename or "image.jpg", content))

        search_result = search_service.search_many(uploaded_files, event_id=event_id)
        if search_result["photo_count"] == 0:
            raise HTTPException(status_code=400, detail="No supported images were uploaded")

        photos = [decorate_batch_result(photo, request, event_id) for photo in search_result["photos"]]
        return templates.TemplateResponse(
            request,
            "folder_results.html",
            {
                "photos": photos,
                "photo_count": search_result["photo_count"],
                "scope": build_scope_context(request, repository, event_id),
                "all_persons": decorate_person_rows(repository.list_persons(), request, event_id),
            },
        )

    @app.post("/faces/{face_id}/persons")
    async def create_person_from_face(
        face_id: int,
        display_name: str = Form(...),
        redirect_to: str = Form("/"),
    ) -> RedirectResponse:
        repository.create_person(display_name=display_name, face_id=face_id)
        return RedirectResponse(url=redirect_to, status_code=303)

    @app.post("/faces/{face_id}/persons/attach")
    async def attach_face_to_person_form(
        face_id: int,
        person_id: int = Form(...),
        redirect_to: str = Form("/"),
    ) -> RedirectResponse:
        repository.attach_face_to_person(person_id=person_id, face_id=face_id)
        return RedirectResponse(url=redirect_to, status_code=303)

    @app.post("/faces/{face_id}/persons/{person_id}")
    async def attach_face_to_person(
        face_id: int,
        person_id: int,
        redirect_to: str = Form("/"),
    ) -> RedirectResponse:
        repository.attach_face_to_person(person_id=person_id, face_id=face_id)
        return RedirectResponse(url=redirect_to, status_code=303)

    @app.post("/persons/{person_id}/rename")
    async def rename_person(
        person_id: int,
        display_name: str = Form(...),
        redirect_to: str = Form("/"),
    ) -> RedirectResponse:
        try:
            repository.rename_person(person_id=person_id, display_name=display_name)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return RedirectResponse(url=redirect_to, status_code=303)

    @app.post("/persons/{person_id}/delete")
    async def delete_person(
        person_id: int,
        request: Request,
        redirect_to: str = Form(""),
    ) -> RedirectResponse:
        try:
            repository.delete_person(person_id=person_id)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        target = redirect_to or str(request.url_for("home"))
        return RedirectResponse(url=target, status_code=303)

    @app.get("/persons/{person_id}", response_class=HTMLResponse)
    async def person_detail(
        request: Request,
        person_id: int,
        event_id: int | None = Query(None),
    ) -> HTMLResponse:
        person_result = search_service.search_person(person_id, event_id=event_id)
        if person_result is None:
            raise HTTPException(status_code=404, detail="Person not found")

        confirmed_faces = [
            {
                **face,
                "crop_url": storage_url(request, face["crop_path"]) if face["crop_path"] else None,
                "image_url": dataset_url(request, face["relative_path"]),
                "photo_url": build_photo_url(request, face["photo_id"], event_id),
            }
            for face in person_result["confirmed_faces"]
        ]
        photos = [
            {
                **photo,
                "image_url": dataset_url(request, photo["relative_path"]),
                "crop_url": storage_url(request, photo["crop_path"]) if photo["crop_path"] else None,
                "box": face_box(photo, photo["width"], photo["height"]),
                "photo_url": build_photo_url(request, photo["photo_id"], event_id),
                "download_url": str(request.url_for("download_photo", photo_id=int(photo["photo_id"]))),
            }
            for photo in person_result["photos"]
        ]

        return templates.TemplateResponse(
            request,
            "person_detail.html",
            {
                "person": person_result["person"],
                "confirmed_faces": confirmed_faces,
                "photos": photos,
                "scope": build_scope_context(request, repository, event_id),
                "delete_redirect_url": build_scope_context(request, repository, event_id)["browse_url"],
                "download_zip_url": (
                    build_person_download_url(request, person_id, event_id)
                    if event_id is not None
                    else None
                ),
            },
        )

    @app.get("/photos/{photo_id}/download")
    async def download_photo(photo_id: int) -> FileResponse:
        photo = repository.get_photo(photo_id)
        if photo is None:
            raise HTTPException(status_code=404, detail="Photo not found")

        path = settings.images_dir / str(photo["relative_path"])
        if not path.exists():
            raise HTTPException(status_code=404, detail="Photo file not found")
        return FileResponse(path=path, filename=Path(str(photo["relative_path"])).name)

    @app.get("/persons/{person_id}/download")
    async def download_person_event(person_id: int, event_id: int = Query(...)) -> StreamingResponse:
        person_result = search_service.search_person(person_id, event_id=event_id)
        if person_result is None:
            raise HTTPException(status_code=404, detail="Person not found")

        event = repository.get_event(event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="Event not found")

        archive_buffer = io.BytesIO()
        added_paths: set[str] = set()
        with zipfile.ZipFile(archive_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for photo in person_result["photos"]:
                relative_path = str(photo["relative_path"])
                if relative_path in added_paths:
                    continue
                added_paths.add(relative_path)
                image_path = settings.images_dir / relative_path
                if image_path.exists():
                    archive.write(image_path, arcname=relative_path)
        archive_buffer.seek(0)

        safe_event = slugify_filename(str(event["name"])) or f"event-{event_id}"
        safe_person = slugify_filename(str(person_result["person"]["display_name"])) or f"person-{person_id}"
        filename = f"{safe_person}_{safe_event}.zip"
        return StreamingResponse(
            archive_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    return app


def render_library_page(
    request: Request,
    repository: FaceRepository,
    event_id: int | None,
    page: int,
) -> HTMLResponse:
    templates = request.app.state.templates
    stats = repository.get_stats(event_id=event_id)
    page_size = 60
    total_pages = max(1, math.ceil(stats["photo_count"] / page_size)) if stats["photo_count"] else 1
    current_page = min(page, total_pages)
    offset = (current_page - 1) * page_size
    photos = [
        decorate_photo_row(row, request, event_id)
        for row in repository.list_photos(limit=page_size, offset=offset, event_id=event_id)
    ]
    current_event = decorate_event_row(repository.get_event(event_id), request) if event_id is not None else None
    return templates.TemplateResponse(
        request,
        "library.html",
        {
            "stats": stats,
            "photos": photos,
            "indexed": stats["photo_count"] > 0,
            "pagination": build_pagination(
                total_items=stats["photo_count"],
                page_size=page_size,
                current_page=current_page,
            ),
            "scope": build_scope_context(request, repository, event_id),
            "current_event": current_event,
            "all_persons": decorate_person_rows(repository.list_persons(event_id=event_id), request, event_id),
            "all_events": [decorate_event_row(row, request) for row in repository.list_events()],
        },
    )


def render_search_results(
    request: Request,
    repository: FaceRepository,
    search_result: dict[str, object],
    query_image_url: str,
    query_context: str,
    source_photo_id: int | None,
    event_id: int | None,
) -> HTMLResponse:
    templates = request.app.state.templates
    persons = decorate_person_rows(repository.list_persons(), request, event_id)

    query_faces = [
        {
            **face,
            "box": face_box(face, search_result["query_width"], search_result["query_height"]),
            "crop_url": storage_url(request, face["crop_path"]) if face["crop_path"] else None,
            "matches": [
                {
                    **match,
                    "image_url": dataset_url(request, match["relative_path"]),
                    "crop_url": storage_url(request, match["crop_path"]) if match["crop_path"] else None,
                    "box": face_box(match, match["width"], match["height"]),
                    "photo_url": build_photo_url(request, match["photo_id"], event_id),
                    "download_url": str(request.url_for("download_photo", photo_id=int(match["photo_id"]))),
                    "person_url": (
                        build_person_url(request, int(match["person_id"]), event_id)
                        if match.get("person_id") is not None
                        else None
                    ),
                }
                for match in face["matches"]
            ],
        }
        for face in search_result["query_faces"]
    ]

    return templates.TemplateResponse(
        request,
        "search_results.html",
        {
            "query_image_url": query_image_url,
            "query_width": search_result["query_width"],
            "query_height": search_result["query_height"],
            "query_faces": query_faces,
            "query_context": query_context,
            "source_photo_id": source_photo_id,
            "source_photo_url": (
                build_photo_url(request, source_photo_id, event_id) if source_photo_id is not None else None
            ),
            "scope": build_scope_context(request, repository, event_id),
            "all_persons": persons,
        },
    )


def decorate_batch_result(photo: dict[str, object], request: Request, event_id: int | None) -> dict[str, object]:
    return {
        "filename": photo["filename"],
        "query_image_url": storage_url(request, photo["query_path"]),
        "query_width": photo["query_width"],
        "query_height": photo["query_height"],
        "query_faces": [
            {
                **face,
                "box": face_box(face, photo["query_width"], photo["query_height"]),
                "crop_url": storage_url(request, face["crop_path"]) if face["crop_path"] else None,
                "matches": [
                    {
                        **match,
                        "image_url": dataset_url(request, match["relative_path"]),
                        "crop_url": storage_url(request, match["crop_path"]) if match["crop_path"] else None,
                        "box": face_box(match, match["width"], match["height"]),
                        "photo_url": build_photo_url(request, match["photo_id"], event_id),
                        "download_url": str(request.url_for("download_photo", photo_id=int(match["photo_id"]))),
                        "person_url": (
                            build_person_url(request, int(match["person_id"]), event_id)
                            if match.get("person_id") is not None
                            else None
                        ),
                    }
                    for match in face["matches"]
                ],
            }
            for face in photo["query_faces"]
        ],
    }


def dataset_url(request: Request, relative_path: str) -> str:
    return str(request.url_for("dataset", path=relative_path))


def storage_url(request: Request, relative_path: str) -> str:
    return str(request.url_for("storage", path=relative_path))


def decorate_event_row(row: object, request: Request) -> dict[str, object]:
    return {
        "id": int(row["id"]),
        "name": str(row["name"]),
        "event_date": str(row["event_date"] or ""),
        "is_default": bool(int(row["is_default"])),
        "photo_count": int(row["photo_count"]),
        "face_count": int(row["face_count"]),
        "person_count": int(row["person_count"]),
        "url": str(request.url_for("event_detail", event_id=int(row["id"]))),
    }


def decorate_photo_row(row: object, request: Request, event_id: int | None = None) -> dict[str, object]:
    current_event_id = event_id if event_id is not None else int(row["event_id"])
    return {
        "id": int(row["id"]),
        "event_id": int(row["event_id"]),
        "relative_path": row["relative_path"],
        "width": int(row["width"]),
        "height": int(row["height"]),
        "face_count": int(row["face_count"]),
        "event_name": str(row["event_name"]),
        "event_date": str(row["event_date"] or ""),
        "image_url": dataset_url(request, row["relative_path"]),
        "photo_url": build_photo_url(request, int(row["id"]), current_event_id),
        "download_url": str(request.url_for("download_photo", photo_id=int(row["id"]))),
    }


def decorate_face_row(
    row: object,
    width: int,
    height: int,
    request: Request,
    event_id: int | None,
) -> dict[str, object]:
    crop_url = storage_url(request, row["crop_path"]) if row["crop_path"] else None
    return {
        "id": int(row["id"]),
        "person_index": int(row["person_index"]),
        "label": build_indexed_face_label(int(row["id"])),
        "top_px": int(row["top_px"]),
        "right_px": int(row["right_px"]),
        "bottom_px": int(row["bottom_px"]),
        "left_px": int(row["left_px"]),
        "crop_url": crop_url,
        "person_id": int(row["person_id"]) if row["person_id"] is not None else None,
        "person_name": str(row["person_name"]) if row["person_name"] else None,
        "search_url": build_face_search_url(request, int(row["id"]), event_id),
        "person_url": (
            build_person_url(request, int(row["person_id"]), event_id)
            if row["person_id"] is not None
            else None
        ),
        "box": face_box(row, width, height),
    }


def decorate_person_rows(rows: list[object], request: Request, event_id: int | None) -> list[dict[str, object]]:
    return [
        {
            "id": int(row["id"]),
            "display_name": str(row["display_name"]),
            "confirmed_face_count": int(row["confirmed_face_count"]),
            "photo_count": int(row["photo_count"]),
            "url": build_person_url(request, int(row["id"]), event_id),
        }
        for row in rows
    ]


def face_box(face: object, width: int, height: int) -> dict[str, float]:
    left = int(face["left_px"])
    top = int(face["top_px"])
    right = int(face["right_px"])
    bottom = int(face["bottom_px"])
    return {
        "left_pct": round((left / width) * 100, 4) if width else 0.0,
        "top_pct": round((top / height) * 100, 4) if height else 0.0,
        "width_pct": round(((right - left) / width) * 100, 4) if width else 0.0,
        "height_pct": round(((bottom - top) / height) * 100, 4) if height else 0.0,
    }


def build_pagination(total_items: int, page_size: int, current_page: int) -> dict[str, int | bool]:
    total_pages = max(1, math.ceil(total_items / page_size)) if total_items else 1
    start_item = ((current_page - 1) * page_size) + 1 if total_items else 0
    end_item = min(total_items, current_page * page_size) if total_items else 0
    return {
        "total_items": total_items,
        "page_size": page_size,
        "current_page": current_page,
        "total_pages": total_pages,
        "has_prev": current_page > 1,
        "has_next": current_page < total_pages,
        "prev_page": current_page - 1,
        "next_page": current_page + 1,
        "start_item": start_item,
        "end_item": end_item,
    }


def build_scope_context(
    request: Request,
    repository: FaceRepository,
    event_id: int | None,
) -> dict[str, object]:
    if event_id is None:
        return {
            "event_id": None,
            "title": "Все события",
            "subtitle": "Каталог и поиск по всей базе",
            "browse_url": str(request.url_for("library")),
            "home_url": str(request.url_for("home")),
            "kind": "all",
        }

    event = repository.get_event(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")

    title = str(event["name"])
    event_date = str(event["event_date"] or "")
    subtitle = event_date if event_date else "Без даты"
    return {
        "event_id": event_id,
        "title": title,
        "subtitle": subtitle,
        "browse_url": str(request.url_for("event_detail", event_id=event_id)),
        "home_url": str(request.url_for("home")),
        "kind": "event",
    }


def build_scope_search_caption(repository: FaceRepository, event_id: int | None) -> str:
    if event_id is None:
        return "Результат поиска по всем событиям"
    event = repository.get_event(event_id)
    if event is None:
        return "Результат поиска"
    return f"Результат поиска в событии «{event['name']}»"


def build_query_string(**params: int | str | None) -> str:
    filtered = {key: value for key, value in params.items() if value is not None and value != ""}
    return urlencode(filtered)


def build_photo_url(request: Request, photo_id: int, event_id: int | None) -> str:
    query = build_query_string(event_id=event_id)
    base = str(request.url_for("photo_detail", photo_id=photo_id))
    return f"{base}?{query}" if query else base


def build_face_search_url(request: Request, face_id: int, event_id: int | None) -> str:
    query = build_query_string(event_id=event_id)
    base = str(request.url_for("search_indexed_face", face_id=face_id))
    return f"{base}?{query}" if query else base


def build_person_url(request: Request, person_id: int, event_id: int | None) -> str:
    base = str(request.url_for("person_detail", person_id=person_id))
    query = build_query_string(event_id=event_id)
    return f"{base}?{query}" if query else base


def build_person_download_url(request: Request, person_id: int, event_id: int) -> str:
    base = str(request.url_for("download_person_event", person_id=person_id))
    query = build_query_string(event_id=event_id)
    return f"{base}?{query}" if query else base


def slugify_filename(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    sanitized = "".join(char.lower() if char.isalnum() else "-" for char in ascii_value).strip("-")
    while "--" in sanitized:
        sanitized = sanitized.replace("--", "-")
    return sanitized


app = create_app()
