from __future__ import annotations

import math

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from photo_recognizer.config import load_settings
from photo_recognizer.db import FaceRepository
from photo_recognizer.services import IndexService, SearchService, build_indexed_face_label


def create_app() -> FastAPI:
    settings = load_settings()
    repository = FaceRepository(settings.db_path)
    repository.init_db()

    index_service = IndexService(settings, repository)
    search_service = SearchService(settings, repository)

    app = FastAPI(title="Photo Recognizer", version="0.1.0")
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
    async def home(request: Request, page: int = Query(1, ge=1)) -> HTMLResponse:
        stats = repository.get_stats()
        page_size = 60
        total_pages = max(1, math.ceil(stats["photo_count"] / page_size)) if stats["photo_count"] else 1
        current_page = min(page, total_pages)
        offset = (current_page - 1) * page_size
        photos = [
            decorate_photo_row(row, request)
            for row in repository.list_photos(limit=page_size, offset=offset)
        ]
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "stats": stats,
                "photos": photos,
                "indexed": stats["photo_count"] > 0,
                "pagination": build_pagination(
                    total_items=stats["photo_count"],
                    page_size=page_size,
                    current_page=current_page,
                ),
            },
        )

    @app.get("/search/face", response_class=HTMLResponse)
    async def search_face_by_id(face_id: int = Query(..., ge=1)) -> RedirectResponse:
        return RedirectResponse(url=f"/face/{face_id}/search", status_code=303)

    @app.get("/photo/{photo_id}", response_class=HTMLResponse)
    async def photo_detail(request: Request, photo_id: int) -> HTMLResponse:
        photo = repository.get_photo(photo_id)
        if photo is None:
            raise HTTPException(status_code=404, detail="Photo not found")

        faces = [
            decorate_face_row(face, int(photo["width"]), int(photo["height"]), request)
            for face in repository.get_photo_faces(photo_id)
        ]
        return templates.TemplateResponse(
            request,
            "photo_detail.html",
            {
                "photo": decorate_photo_row(photo, request),
                "faces": faces,
            },
        )

    @app.get("/face/{face_id}/search", response_class=HTMLResponse)
    async def search_indexed_face(request: Request, face_id: int) -> HTMLResponse:
        search_result = search_service.search_indexed_face(face_id)
        if search_result is None:
            raise HTTPException(status_code=404, detail="Face not found")

        return render_search_results(
            request=request,
            search_result=search_result,
            query_image_url=dataset_url(request, search_result["query_path"]),
            query_context=f"Поиск похожих для {search_result['query_faces'][0]['label']} "
            f"на фото {search_result['source_relative_path']}",
            source_photo_id=search_result["source_photo_id"],
        )

    @app.post("/search", response_class=HTMLResponse)
    async def search(request: Request, file: UploadFile = File(...)) -> HTMLResponse:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        stored_path = search_service.store_query_file(file.filename or "query.jpg", content)
        search_result = search_service.search(stored_path)

        return render_search_results(
            request=request,
            search_result=search_result,
            query_image_url=storage_url(request, search_result["query_path"]),
            query_context="Результат поиска по загруженной фотографии",
            source_photo_id=None,
        )

    @app.post("/search/folder", response_class=HTMLResponse)
    async def search_folder(request: Request, files: list[UploadFile] = File(...)) -> HTMLResponse:
        uploaded_files: list[tuple[str, bytes]] = []
        for file in files:
            content = await file.read()
            uploaded_files.append((file.filename or "image.jpg", content))

        search_result = search_service.search_many(uploaded_files)
        if search_result["photo_count"] == 0:
            raise HTTPException(status_code=400, detail="No supported images were uploaded")

        templates = request.app.state.templates
        photos = [decorate_batch_result(photo, request) for photo in search_result["photos"]]
        return templates.TemplateResponse(
            request,
            "folder_results.html",
            {
                "photos": photos,
                "photo_count": search_result["photo_count"],
            },
        )

    return app


def render_search_results(
    request: Request,
    search_result: dict[str, object],
    query_image_url: str,
    query_context: str,
    source_photo_id: int | None,
) -> HTMLResponse:
    templates = request.app.state.templates

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
            "primary_engine_label": search_result["primary_engine_label"],
            "verify_engine_label": search_result["verify_engine_label"],
            "primary_threshold": search_result["primary_threshold"],
            "verify_threshold": search_result["verify_threshold"],
            "query_context": query_context,
            "source_photo_id": source_photo_id,
        },
    )


def decorate_batch_result(photo: dict[str, object], request: Request) -> dict[str, object]:
    return {
        "filename": photo["filename"],
        "query_image_url": storage_url(request, photo["query_path"]),
        "query_width": photo["query_width"],
        "query_height": photo["query_height"],
        "primary_engine_label": photo["primary_engine_label"],
        "verify_engine_label": photo["verify_engine_label"],
        "primary_threshold": photo["primary_threshold"],
        "verify_threshold": photo["verify_threshold"],
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


def decorate_photo_row(row: object, request: Request) -> dict[str, object]:
    return {
        "id": int(row["id"]),
        "relative_path": row["relative_path"],
        "width": int(row["width"]),
        "height": int(row["height"]),
        "face_count": int(row["face_count"]),
        "image_url": dataset_url(request, row["relative_path"]),
    }


def decorate_face_row(row: object, width: int, height: int, request: Request) -> dict[str, object]:
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
        "search_url": str(request.url_for("search_indexed_face", face_id=int(row["id"]))),
        "box": face_box(row, width, height),
    }


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


app = create_app()
