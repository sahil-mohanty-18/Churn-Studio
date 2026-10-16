"""Churn Studio application factory."""

import logging

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.exceptions import HTTPException

from app.config import ROOT, Config
from app.db import close_db

PAGES = {
    "dashboard": ("Overview", "Your retention command center."),
    "predict": ("Predict churn", "Understand the customer behind the probability."),
    "batch": (
        "Batch prediction",
        "Turn a customer list into a focused retention queue.",
    ),
    "customers": ("Customers", "Find the right people. Start the right conversations."),
    "insights": ("Model insights", "Transparent performance. Measurable confidence."),
    "simulator": ("What-if simulator", "Explore scenarios before you make a move."),
}


def create_app(config: dict | None = None) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(ROOT / "templates"),
        static_folder=str(ROOT / "static"),
    )
    app.config.from_object(Config)
    if config:
        app.config.update(config)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    from app.services import Predictor

    app.extensions["predictor"] = Predictor(app.config["MODEL_PATH"])
    app.extensions["downloads"] = {}
    CORS(app, resources={r"/api/*": {"origins": app.config["CORS_ORIGINS"]}})
    Limiter(
        get_remote_address,
        app=app,
        default_limits=["120 per minute"],
        storage_uri=app.config["RATELIMIT_STORAGE_URI"],
    )
    app.teardown_appcontext(close_db)
    from app.api import api

    app.register_blueprint(api)

    @app.get("/")
    def home():
        return render_template(
            "index.html",
            page="dashboard",
            title=PAGES["dashboard"][0],
            subtitle=PAGES["dashboard"][1],
            pages=PAGES,
        )

    @app.get("/<page>")
    def page_view(page):
        if page not in PAGES:
            return jsonify(error="Page not found."), 404
        return render_template(
            "index.html",
            page=page,
            title=PAGES[page][0],
            subtitle=PAGES[page][1],
            pages=PAGES,
        )

    @app.errorhandler(ValueError)
    def invalid(error):
        return jsonify(error=str(error)), 400

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify(error=error.description), error.code

    @app.errorhandler(Exception)
    def unexpected(error):
        app.logger.exception("Unhandled request failure")
        return jsonify(error="An internal error occurred. Please try again."), 500

    @app.after_request
    def headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
        )
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    return app
