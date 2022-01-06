from flask import Blueprint, session, request, flash, redirect, url_for
from flask.templating import render_template
from app.authentication.forms import LoginForm
from app.authentication.models import User

authentication_bp = Blueprint("authentication", __name__, url_prefix="/auth")

# Routes
@authentication_bp.route("/signin", methods=["GET", "POST"])
def signin_page():
    login_form = LoginForm(request.form)
    if login_form.validate_on_submit():
        attempted_user = User.query.filter_by(username=login_form.username.data).first()
        if attempted_user:
            if attempted_user.check_password(login_form.password.data):
                session["user_id"] = attempted_user.id
                flash(f"Welcome, {attempted_user.username}.", category="success")
                return redirect(url_for("main.home_page"))
            else:
                flash(f"Wrong password.", category="danger")
        else:
            flash("User does not exist.")

    return render_template("authentication/signin.html", login_form=login_form)
